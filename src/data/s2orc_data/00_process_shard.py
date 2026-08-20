#!/usr/bin/env python3
"""
S2ORC Full-Release Download & Sentence Extraction Pipeline
==========================================================

Downloads one full S2ORC release and builds a sentence-level corpus of
Computer Science papers.

Phase 0: Build CS metadata lookup from 'papers' dataset shards
         (keeps CS papers with publicationdate >= --date_cutoff).
Phase 1: Process 's2orc' shards in parallel — extract sentences for the papers
         selected in Phase 0, join metadata, save parquet shards, merge.

Reproducibility: --release_id defaults to 2026-05-05, the snapshot used for
the paper. Pass --release_id latest to run against the current release.

Memory-efficient design:
    - CS corpus IDs stored as a memory-mapped NumPy array (shared across workers via OS page cache)
    - Metadata stored as a Feather (Arrow IPC) file, memory-mapped by each worker
    - This avoids duplicating ~10 GB of Python dicts per worker process

Usage:
    # Full pipeline (Phase 0 + Phase 1):
    python 00_process_shard.py --api_key YOUR_KEY --output_dir ./data_s2orc/output --temp_dir ./data_s2orc/temp

    # Phase 0 only (metadata build):
    python 00_process_shard.py --api_key YOUR_KEY --phase 0

    # Phase 1 only (assumes metadata already built):
    python 00_process_shard.py --api_key YOUR_KEY --phase 1

    # Test run (limit lines per shard):
    python 00_process_shard.py --api_key YOUR_KEY --test_lines 500 --workers 2
"""

import argparse
import gc
import gzip
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf
import requests

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def ts():
    """Compact timestamp for log lines."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, flush=True):
    print(f"[{ts()}] {msg}", flush=flush)


def log_worker(worker_id, msg, flush=True):
    print(f"[{ts()}] [worker-{worker_id:02d}] {msg}", flush=flush)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

API_BASE = "https://api.semanticscholar.org/datasets/v1"
_last_api_call = 0.0  # module-level rate-limit tracker


def _rate_limit():
    """Enforce 1 req/sec between API calls (not S3 downloads)."""
    global _last_api_call
    elapsed = time.time() - _last_api_call
    if elapsed < 1.0:
        time.sleep(1.0 - elapsed)
    _last_api_call = time.time()


def api_get(endpoint, api_key, params=None):
    """GET from Semantic Scholar datasets API with rate limiting + retries."""
    url = f"{API_BASE}/{endpoint.lstrip('/')}"
    headers = {"x-api-key": api_key}
    for attempt in range(5):
        _rate_limit()
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
            if resp.status_code == 429:
                wait = 2 ** attempt * 5
                log(f"  Rate-limited (429). Retrying in {wait}s... (attempt {attempt+1}/5)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            wait = 2 ** attempt * 3
            log(f"  API error: {e}. Retrying in {wait}s... (attempt {attempt+1}/5)")
            time.sleep(wait)
    raise RuntimeError(f"Failed after 5 attempts: {endpoint}")


def resolve_release_id(api_key, release_id):
    """Return a concrete release ID, resolving the literal 'latest' via the API."""
    if release_id != "latest":
        return release_id
    info = api_get("release/latest", api_key)
    rid = info.get("release_id", "") if isinstance(info, dict) else str(info)
    if not rid:
        raise RuntimeError(f"Could not resolve 'latest' release ID from: {info}")
    return rid


def download_file(url, dest_path, desc="file", worker_id=None):
    """
    Download a (possibly large) file from a pre-signed S3 URL with resume support.
    Returns True if download succeeded, False if file already complete.
    """
    _log = (lambda m: log_worker(worker_id, m)) if worker_id is not None else log

    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # --- Check for completed marker ---
    done_marker = dest.with_suffix(dest.suffix + ".done")
    if done_marker.exists() and dest.exists():
        _log(f"  Skipping {desc} — already downloaded (marker exists)")
        return False

    # --- Resume support: check partial download size ---
    existing_size = dest.stat().st_size if dest.exists() else 0
    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        _log(f"  Resuming {desc} from {existing_size / 1e6:.1f} MB...")

    for attempt in range(5):
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=300)
            # If server doesn't support range or file is complete
            if resp.status_code == 416:
                _log(f"  {desc} already fully downloaded ({existing_size / 1e6:.1f} MB)")
                done_marker.touch()
                return False
            resp.raise_for_status()

            total = resp.headers.get("Content-Length")
            total_str = f"{int(total) / 1e6:.1f} MB" if total else "unknown size"

            mode = "ab" if existing_size > 0 and resp.status_code == 206 else "wb"
            if mode == "wb":
                existing_size = 0  # full restart

            downloaded = existing_size
            chunk_count = 0
            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks
                    f.write(chunk)
                    downloaded += len(chunk)
                    chunk_count += 1
                    if chunk_count % 25 == 0:
                        _log(f"  {desc}: {downloaded / 1e6:.1f} MB / {total_str}")

            _log(f"  {desc}: done ({downloaded / 1e6:.1f} MB)")
            done_marker.touch()
            return True

        except (requests.exceptions.RequestException, IOError) as e:
            wait = 2 ** attempt * 5
            _log(f"  Download error for {desc}: {e}. Retry in {wait}s (attempt {attempt+1}/5)")
            time.sleep(wait)
            # Update existing_size for next resume attempt
            existing_size = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

    raise RuntimeError(f"Failed to download {desc} after 5 attempts")


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------

def load_spacy_model():
    """
    Load the scispaCy model with only the sentencizer pipe enabled.
    Falls back to blank English + sentencizer if scispaCy is not installed
    (the paper run used en_core_sci_sm — install it for exact reproduction).
    """
    import spacy

    try:
        nlp = spacy.load("en_core_sci_sm", disable=[
            "tagger", "ner", "entity_linker", "lemmatizer",
            "morphologizer", "parser", "tok2vec"
        ])
        # Make sure sentencizer is active
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        log("  Loaded en_core_sci_sm with sentencizer")
    except OSError:
        log("  WARNING: en_core_sci_sm not found — using blank English + sentencizer")
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

    nlp.max_length = 2_000_000  # some papers are very long
    return nlp


def parse_annotations(annotations_dict):
    """
    Parse the double-encoded annotation structure from new S2ORC schema.

    Returns:
        paragraphs: list of {"start": int, "end": int}
        section_headers: list of {"start": int, "end": int}
    """
    paragraphs = []
    section_headers = []

    if not annotations_dict:
        return paragraphs, section_headers

    # --- Paragraphs ---
    raw_para = annotations_dict.get("paragraph", "[]")
    if isinstance(raw_para, str):
        try:
            parsed = json.loads(raw_para)
        except (json.JSONDecodeError, TypeError):
            parsed = []
    else:
        parsed = raw_para if isinstance(raw_para, list) else []

    for p in parsed:
        try:
            paragraphs.append({"start": int(p["start"]), "end": int(p["end"])})
        except (KeyError, ValueError, TypeError):
            continue

    # --- Section headers ---
    raw_sec = annotations_dict.get("sectionheader", "[]")
    if isinstance(raw_sec, str):
        try:
            parsed = json.loads(raw_sec)
        except (json.JSONDecodeError, TypeError):
            parsed = []
    else:
        parsed = raw_sec if isinstance(raw_sec, list) else []

    for s in parsed:
        try:
            section_headers.append({"start": int(s["start"]), "end": int(s["end"])})
        except (KeyError, ValueError, TypeError):
            continue

    # Sort by start offset
    paragraphs.sort(key=lambda x: x["start"])
    section_headers.sort(key=lambda x: x["start"])

    return paragraphs, section_headers


def assign_sections(paragraphs, section_headers, full_text):
    """
    For each paragraph, assign the most recent section header whose end < paragraph start.
    Returns list of section name strings (one per paragraph).
    """
    sections = []
    for para in paragraphs:
        section_name = ""
        for sh in section_headers:
            if sh["end"] <= para["start"]:
                section_name = full_text[sh["start"]:sh["end"]].strip()
            else:
                break  # headers are sorted, no need to continue
        sections.append(section_name)
    return sections


def extract_sentences_from_record(record, nlp, sent_len_cutoff=20):
    """
    Extract sentences from a single s2orc record using the new annotation schema.

    Returns list of dicts with keys: corpusid, single_text, multi_text, section
    """
    corpusid = record.get("corpusid")
    content = record.get("content")
    if not content or not isinstance(content, dict):
        return []

    full_text = content.get("text", "")
    annotations = content.get("annotations", {})

    if not full_text or not annotations:
        return []

    paragraphs, section_headers = parse_annotations(annotations)
    if not paragraphs:
        return []

    section_names = assign_sections(paragraphs, section_headers, full_text)

    # Extract paragraph texts
    para_texts = []
    for para in paragraphs:
        txt = full_text[para["start"]:para["end"]].strip()
        if txt:
            para_texts.append(txt)
        else:
            para_texts.append("")

    # Sentencize all non-empty paragraphs in one nlp.pipe call for efficiency
    nonempty_indices = [i for i, t in enumerate(para_texts) if t]
    nonempty_texts = [para_texts[i] for i in nonempty_indices]

    # Process with spaCy
    para_sentences = {i: [] for i in range(len(para_texts))}
    if nonempty_texts:
        for idx, doc in zip(nonempty_indices, nlp.pipe(nonempty_texts, batch_size=256)):
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            para_sentences[idx] = sents

    # Build output rows
    rows = []
    for para_idx in range(len(para_texts)):
        sents = para_sentences[para_idx]
        section = section_names[para_idx] if para_idx < len(section_names) else ""
        n_sents = len(sents)

        for sent_idx, sent_text in enumerate(sents):
            # --- Filter: minimum length + starts with uppercase or digit ---
            if len(sent_text) < sent_len_cutoff:
                continue
            if sent_text and not (sent_text[0].isupper() or sent_text[0].isdigit()):
                continue

            # --- Build multi_text: prev + curr + next (paragraph-scoped) ---
            if n_sents < 3:
                multi_text = None
            elif sent_idx == 0 or sent_idx == n_sents - 1:
                multi_text = None
            else:
                prev_sent = sents[sent_idx - 1]
                next_sent = sents[sent_idx + 1]
                multi_text = f"{prev_sent} {sent_text} {next_sent}"

            rows.append({
                "corpusid": corpusid,
                "single_text": sent_text,
                "multi_text": multi_text,
                "section": section,
            })

    return rows


# ---------------------------------------------------------------------------
# Shared metadata helpers (memory-mapped via Arrow/Feather + NumPy)
# ---------------------------------------------------------------------------

def build_shared_files(df_meta, output_dir):
    """
    Build memory-mappable shared lookup files from the metadata DataFrame:
      1. _shared_cs_ids.npy     — sorted int64 array for fast np.searchsorted membership test
      2. _shared_metadata.feather — Arrow IPC file with all metadata columns, sorted by corpusid

    CRITICAL: Both files must have the same sorted order with no duplicate corpusids,
    because workers use np.searchsorted on the IDs array to get the row index into
    the Feather DataFrame directly.

    These files are read-only and shared across all worker processes via the OS page cache.
    Returns (ids_path, feather_path).
    """
    ids_path = Path(output_dir) / "_shared_cs_ids.npy"
    feather_path = Path(output_dir) / "_shared_metadata.feather"

    log("Building shared memory-mapped files for workers...")
    t0 = time.time()

    # --- Deduplicate and sort by corpusid ---
    df_deduped = df_meta.drop_duplicates(subset="corpusid", keep="first")
    df_deduped = df_deduped.sort_values("corpusid").reset_index(drop=True)
    if len(df_deduped) < len(df_meta):
        log(f"  Deduplicated: {len(df_meta):,} → {len(df_deduped):,} unique corpusids")

    # --- 1. Sorted corpus ID array for O(log n) membership + row-index lookups ---
    cs_ids = df_deduped["corpusid"].values.astype(np.int64)
    np.save(ids_path, cs_ids)
    log(f"  Saved {len(cs_ids):,} corpus IDs → {ids_path} "
        f"({ids_path.stat().st_size / 1e6:.1f} MB)")

    # --- 2. Feather file for metadata lookups (same row order as cs_ids) ---
    df_feather = df_deduped.copy()
    # Convert list columns to JSON strings (Arrow can't memory-map nested types efficiently)
    # Always use json.dumps to guarantee valid JSON, regardless of input type
    def _to_json_str(x):
        if isinstance(x, list):
            return json.dumps(x)
        if isinstance(x, np.ndarray):
            return json.dumps(x.tolist())
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "[]"
        return json.dumps([x])

    for col in ["field", "authors"]:
        if col in df_feather.columns:
            df_feather[col] = df_feather[col].apply(_to_json_str)
    pf.write_feather(df_feather, feather_path)
    log(f"  Saved metadata feather → {feather_path} "
        f"({feather_path.stat().st_size / 1e6:.1f} MB)")

    elapsed = time.time() - t0
    log(f"  Shared files built in {elapsed:.1f}s")

    del df_deduped, df_feather
    gc.collect()

    return str(ids_path), str(feather_path)


def cleanup_shared_files(output_dir):
    """Remove shared memory-mapped files after processing is complete."""
    for fname in ["_shared_cs_ids.npy", "_shared_metadata.feather"]:
        p = Path(output_dir) / fname
        try:
            p.unlink(missing_ok=True)
            log(f"  Cleaned up {p}")
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Worker-level shared state (loaded once per process via globals)
# ---------------------------------------------------------------------------

# These globals are populated once per worker process on first use.
# _worker_cs_ids is memory-mapped (shared via OS page cache, zero per-worker cost).
# _worker_meta_df is memory-mapped via Feather (shared via OS page cache).
# No per-worker Python dicts are built — lookups use np.searchsorted on the
# sorted corpusid array, which doubles as both membership check and row-index lookup
# (since the Feather file is sorted by corpusid in the same order).
_worker_cs_ids = None       # np.ndarray, sorted int64, memory-mapped
_worker_meta_df = None      # pd.DataFrame (memory-mapped via feather, sorted by corpusid)
_worker_nlp = None          # spaCy model


def _load_worker_shared(ids_path, feather_path, worker_id):
    """
    Load shared data into worker-process globals. Called once per worker.
    Uses memory-mapped files so the OS shares physical pages across processes.

    No per-worker Python dicts are created — the sorted numpy array serves as
    both the membership set and the row-index lookup (since the Feather DataFrame
    is sorted in the same corpusid order).
    """
    global _worker_cs_ids, _worker_meta_df

    log_worker(worker_id, "Loading shared data (memory-mapped)...")
    t0 = time.time()

    # --- Memory-mapped numpy array for ID membership + row-index lookups ---
    _worker_cs_ids = np.load(ids_path, mmap_mode="r")

    # --- Memory-mapped feather for metadata (sorted by corpusid, same order as _worker_cs_ids) ---
    _worker_meta_df = pf.read_feather(feather_path, memory_map=True)

    elapsed = time.time() - t0
    log_worker(worker_id,
               f"  Shared data loaded in {elapsed:.1f}s "
               f"({len(_worker_cs_ids):,} IDs, {len(_worker_meta_df):,} metadata rows)")


def _check_membership(corpusid):
    """O(log n) membership check using sorted numpy array + searchsorted."""
    idx = np.searchsorted(_worker_cs_ids, corpusid)
    return idx < len(_worker_cs_ids) and _worker_cs_ids[idx] == corpusid


def _safe_json_loads(val):
    """Parse a JSON string back to a Python object, falling back to the raw value."""
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return val


def _get_metadata(corpusid):
    """
    Look up metadata for a corpusid. Returns dict or empty dict if not found.

    Since both _worker_cs_ids and _worker_meta_df are sorted by corpusid,
    np.searchsorted gives us the row index directly — no per-worker dict needed.
    """
    idx = np.searchsorted(_worker_cs_ids, corpusid)
    if idx >= len(_worker_cs_ids) or _worker_cs_ids[idx] != corpusid:
        return {}

    # idx is the row position in the feather DataFrame (same sort order)
    row = _worker_meta_df.iloc[idx]
    meta = {
        "field": _safe_json_loads(row["field"]),
        "journal": row["journal"] if pd.notna(row["journal"]) else "",
        "venue": row["venue"] if pd.notna(row["venue"]) else "",
        "year": row["year"] if pd.notna(row["year"]) else None,
        "authors": _safe_json_loads(row["authors"]),
        "in_citations": row["in_citations"] if pd.notna(row["in_citations"]) else 0,
        "out_citations": row["out_citations"] if pd.notna(row["out_citations"]) else 0,
        "publicationdate": row["publicationdate"] if pd.notna(row["publicationdate"]) else "",
    }
    return meta


# ---------------------------------------------------------------------------
# Phase 0 — Metadata Build
# ---------------------------------------------------------------------------

def _process_papers_shard(args_tuple):
    """
    Worker: download one papers shard, filter for CS + date, save intermediate parquet.
    """
    shard_idx, shard_url, meta_shards_dir, temp_dir, test_lines, date_cutoff = args_tuple

    shard_meta_path = Path(meta_shards_dir) / f"cs_meta_shard_{shard_idx:03d}.parquet"
    temp_file = Path(temp_dir) / f"temp_papers_{shard_idx}.jsonl.gz"

    try:
        download_file(shard_url, temp_file, desc=f"papers shard {shard_idx}")
    except RuntimeError as e:
        log(f"  Papers shard {shard_idx}: download failed — {e}")
        return shard_idx, "failed", 0

    shard_metadata = []
    scanned = 0
    try:
        with gzip.open(temp_file, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if test_lines and line_no >= test_lines:
                    break
                scanned += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # --- Filter: CS field ---
                fields = rec.get("s2fieldsofstudy") or rec.get("s2FieldsOfStudy") or []
                is_cs = False
                field_list = []
                for f_obj in fields:
                    if isinstance(f_obj, dict):
                        cat = f_obj.get("category", "")
                        field_list.append(cat)
                        if cat == "Computer Science":
                            is_cs = True
                    elif isinstance(f_obj, str):
                        field_list.append(f_obj)
                        if f_obj == "Computer Science":
                            is_cs = True
                if not is_cs:
                    continue

                # --- Filter: publication date (keep papers on/after cutoff) ---
                pub_date = rec.get("publicationdate") or rec.get("publicationDate") or ""
                if not pub_date or pub_date < date_cutoff:
                    continue

                # --- Extract metadata ---
                corpusid = rec.get("corpusid")
                if corpusid is None:
                    corpusid = rec.get("corpusId")
                if corpusid is None:
                    continue

                in_cit = rec.get("citationcount")
                if in_cit is None:
                    in_cit = rec.get("citationCount")
                if isinstance(in_cit, list):
                    in_cit = len(in_cit)
                elif in_cit is None:
                    in_cit = 0

                out_cit = rec.get("referencecount")
                if out_cit is None:
                    out_cit = rec.get("referenceCount")
                if isinstance(out_cit, list):
                    out_cit = len(out_cit)
                elif out_cit is None:
                    out_cit = 0

                authors_raw = rec.get("authors") or []
                if authors_raw and isinstance(authors_raw[0], dict):
                    authors = [a.get("name", "") for a in authors_raw]
                else:
                    authors = authors_raw

                year_val = rec.get("year")
                if year_val is not None:
                    try:
                        year_val = int(year_val)
                    except (ValueError, TypeError):
                        year_val = None

                journal_raw = rec.get("journal")
                if isinstance(journal_raw, dict):
                    journal_val = journal_raw.get("name", "")
                elif isinstance(journal_raw, str):
                    journal_val = journal_raw
                else:
                    pv = rec.get("publicationvenue")
                    journal_val = pv.get("name", "") if isinstance(pv, dict) else ""

                venue_raw = rec.get("venue", "")
                venue_val = venue_raw.get("name", "") if isinstance(venue_raw, dict) else str(venue_raw)

                shard_metadata.append({
                    "corpusid": int(corpusid),
                    "field": field_list,
                    "journal": journal_val,
                    "venue": venue_val,
                    "year": year_val,
                    "authors": authors,
                    "in_citations": in_cit,
                    "out_citations": out_cit,
                    "publicationdate": pub_date,
                })

    except Exception as e:
        log(f"  Papers shard {shard_idx}: parse error — {e}")
        traceback.print_exc()

    kept = len(shard_metadata)
    log(f"  Papers shard {shard_idx}: scanned {scanned:,} → kept {kept:,}")

    if shard_metadata:
        df_shard = pd.DataFrame(shard_metadata)
        df_shard.to_parquet(shard_meta_path, index=False)

    # Cleanup
    try:
        temp_file.unlink(missing_ok=True)
        temp_file.with_suffix(temp_file.suffix + ".done").unlink(missing_ok=True)
    except OSError:
        pass
    gc.collect()

    return shard_idx, "done", kept


def phase0_build_metadata(api_key, output_dir, temp_dir, test_lines=None, workers=10,
                          release_id="2026-05-05", date_cutoff="2015-01-01"):
    """
    Download all 'papers' shards from the given release, filter for CS papers
    with publicationdate >= date_cutoff, and save as cs_metadata_lookup.parquet.

    Each shard's filtered results are saved as cs_meta_shard_{i}.parquet so that
    if the job crashes, completed shards are preserved and resumed on re-run.
    """
    metadata_path = Path(output_dir) / "cs_metadata_lookup.parquet"
    if metadata_path.exists():
        n = len(pd.read_parquet(metadata_path))
        log(f"Phase 0: Metadata already exists ({n:,} rows). Skipping.")
        log(f"  Delete {metadata_path} to rebuild.")
        return metadata_path

    log("=" * 70)
    log("PHASE 0: Building CS metadata lookup")
    log("=" * 70)

    # --- Resolve release ---
    release_id = resolve_release_id(api_key, release_id)
    log(f"  Release: {release_id}")

    # --- Get papers shard URLs ---
    log("Fetching papers dataset shard URLs...")
    papers_info = api_get(f"release/{release_id}/dataset/papers", api_key)
    shard_urls = papers_info.get("files", [])
    log(f"  Found {len(shard_urls)} papers shards")

    # Save shard URLs for reference
    urls_path = Path(output_dir) / "papers_shard_urls.json"
    urls_path.parent.mkdir(parents=True, exist_ok=True)
    with open(urls_path, "w") as f:
        json.dump(shard_urls, f, indent=2)
    log(f"  Saved shard URLs to {urls_path}")

    # --- Also fetch and save s2orc shard URLs (needed for Phase 1) ---
    log("Fetching s2orc dataset shard URLs...")
    s2orc_info = api_get(f"release/{release_id}/dataset/s2orc", api_key)
    s2orc_urls = s2orc_info.get("files", [])
    log(f"  Found {len(s2orc_urls)} s2orc shards")
    s2orc_urls_path = Path(output_dir) / "s2orc_shard_urls.json"
    with open(s2orc_urls_path, "w") as f:
        json.dump(s2orc_urls, f, indent=2)
    log(f"  Saved s2orc shard URLs to {s2orc_urls_path}")

    # --- Process each papers shard ---
    log(f"  Date cutoff: publicationdate >= {date_cutoff}")
    temp = Path(temp_dir)
    temp.mkdir(parents=True, exist_ok=True)
    meta_shards_dir = Path(output_dir) / "meta_shards"
    meta_shards_dir.mkdir(parents=True, exist_ok=True)

    # In test mode, only process a few papers shards
    if test_lines:
        max_test_shards = 3
        shard_urls = shard_urls[:max_test_shards]
        log(f"  TEST MODE: limiting to {max_test_shards} papers shards, {test_lines} lines each")

    # --- Check which shards still need processing ---
    remaining = []
    already_done = 0
    for shard_idx, shard_url in enumerate(shard_urls):
        shard_meta_path = meta_shards_dir / f"cs_meta_shard_{shard_idx:03d}.parquet"
        if shard_meta_path.exists():
            try:
                n = len(pd.read_parquet(shard_meta_path, columns=["corpusid"]))
                log(f"  Papers shard {shard_idx}: SKIP — already filtered ({n:,} rows)")
                already_done += 1
                continue
            except Exception:
                shard_meta_path.unlink(missing_ok=True)
        remaining.append((shard_idx, shard_url))

    log(f"  Already completed: {already_done}, Remaining: {len(remaining)}")

    if remaining:
        p0_workers = min(workers, len(remaining))
        log(f"  Launching {p0_workers} parallel workers for {len(remaining)} papers shards...")
        worker_args = [
            (idx, url, str(meta_shards_dir), str(temp_dir), test_lines, date_cutoff)
            for idx, url in remaining
        ]

        with ProcessPoolExecutor(max_workers=p0_workers) as executor:
            futures = {executor.submit(_process_papers_shard, args): args[0]
                       for args in worker_args}
            for future in as_completed(futures):
                shard_idx = futures[future]
                try:
                    result = future.result(timeout=3600)
                    log(f"  Papers shard {result[0]}: {result[1]} — {result[2]:,} kept")
                except Exception as e:
                    log(f"  Papers shard {shard_idx}: FAILED — {e}")

    # --- Merge all shard parquets into final lookup ---
    import glob as glob_mod
    shard_files = sorted(glob_mod.glob(str(meta_shards_dir / "cs_meta_shard_*.parquet")))
    if not shard_files:
        log("ERROR: No CS papers found! Check API schema / field names.")
        sys.exit(1)

    log(f"Merging {len(shard_files)} shard files...")
    dfs = [pd.read_parquet(f) for f in shard_files]
    df_meta = pd.concat(dfs, ignore_index=True)
    del dfs

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    df_meta.to_parquet(metadata_path, index=False)
    log(f"Phase 0 COMPLETE: {len(df_meta):,} CS papers → {metadata_path}")
    log(f"  corpusid range: {df_meta['corpusid'].min()} – {df_meta['corpusid'].max()}")
    log(f"  Year range: {df_meta['year'].min()} – {df_meta['year'].max()}")

    return metadata_path


# ---------------------------------------------------------------------------
# Phase 1 — Process a single s2orc shard (called by worker)
# ---------------------------------------------------------------------------

def process_single_shard(args_tuple):
    """
    Worker function: download one s2orc shard, extract sentences, join metadata,
    save output parquet. Designed to run in a separate process.

    Uses memory-mapped shared files instead of per-process pickle copies.

    args_tuple: (shard_idx, shard_url, ids_path, feather_path,
                 output_dir, temp_dir, test_lines, sent_len_cutoff, worker_id)
    """
    (shard_idx, shard_url, ids_path, feather_path,
     output_dir, temp_dir, test_lines, sent_len_cutoff, worker_id) = args_tuple

    output_path = Path(output_dir) / f"output_shard_{shard_idx:04d}.parquet.gz"
    temp_file = Path(temp_dir) / f"temp_s2orc_{shard_idx:04d}.jsonl.gz"

    # --- Skip if output already exists ---
    if output_path.exists():
        try:
            n = len(pd.read_parquet(output_path, columns=["corpusid"]))
            log_worker(worker_id, f"Shard {shard_idx}: SKIP — output exists ({n:,} rows)")
            return shard_idx, n, "skipped"
        except Exception:
            log_worker(worker_id, f"Shard {shard_idx}: Output exists but corrupt, reprocessing...")
            output_path.unlink(missing_ok=True)

    log_worker(worker_id, f"Shard {shard_idx}: Starting")
    t0 = time.time()

    try:
        # --- Download ---
        download_file(shard_url, temp_file, desc=f"s2orc shard {shard_idx}", worker_id=worker_id)

        # --- Load shared data (once per worker process, via memory-mapped files) ---
        global _worker_cs_ids
        if _worker_cs_ids is None:
            _load_worker_shared(ids_path, feather_path, worker_id)

        # --- Load spaCy (cached per worker process via global) ---
        global _worker_nlp
        if _worker_nlp is None:
            _worker_nlp = load_spacy_model()

        # --- Stream parse ---
        all_rows = []
        scanned = 0
        matched = 0
        sentences_extracted = 0

        with gzip.open(temp_file, "rt", encoding="utf-8") as f:
            for line_no, line in enumerate(f):
                if test_lines and line_no >= test_lines:
                    break
                scanned += 1

                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                corpusid = rec.get("corpusid")
                if corpusid is None:
                    corpusid = rec.get("corpusId")
                if corpusid is None:
                    continue
                corpusid = int(corpusid)

                if not _check_membership(corpusid):
                    continue
                matched += 1

                # --- Extract sentences ---
                rows = extract_sentences_from_record(rec, _worker_nlp, sent_len_cutoff)
                if rows:
                    # --- Join metadata ---
                    meta = _get_metadata(corpusid)
                    for row in rows:
                        row.update(meta)
                    all_rows.extend(rows)
                    sentences_extracted += len(rows)

                if matched % 500 == 0 and matched > 0:
                    log_worker(worker_id,
                               f"  Shard {shard_idx}: scanned {scanned:,}, "
                               f"matched {matched:,}, sentences {sentences_extracted:,}")

        elapsed = time.time() - t0
        log_worker(worker_id,
                   f"  Shard {shard_idx}: Parse done in {elapsed:.0f}s — "
                   f"scanned {scanned:,}, matched {matched:,}, sentences {sentences_extracted:,}")

        # --- Save output ---
        if all_rows:
            df = pd.DataFrame(all_rows)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False, compression="gzip")
            log_worker(worker_id, f"  Shard {shard_idx}: Saved {len(df):,} rows → {output_path}")
            n_rows = len(df)
        else:
            log_worker(worker_id, f"  Shard {shard_idx}: No sentences extracted (0 rows)")
            # Write empty parquet so we don't reprocess
            df = pd.DataFrame(columns=[
                "corpusid", "single_text", "multi_text", "section",
                "field", "journal", "venue", "year", "authors",
                "in_citations", "out_citations", "publicationdate"
            ])
            df.to_parquet(output_path, index=False, compression="gzip")
            n_rows = 0

    except Exception as e:
        log_worker(worker_id, f"  Shard {shard_idx}: FAILED — {e}")
        traceback.print_exc()
        return shard_idx, 0, f"error: {e}"

    finally:
        # Cleanup temp file
        try:
            temp_file.unlink(missing_ok=True)
            temp_file.with_suffix(temp_file.suffix + ".done").unlink(missing_ok=True)
        except OSError:
            pass
        gc.collect()

    total_elapsed = time.time() - t0
    log_worker(worker_id,
               f"Shard {shard_idx}: COMPLETE in {total_elapsed:.0f}s ({n_rows:,} rows)")
    return shard_idx, n_rows, "done"


def phase1_process_shards(api_key, output_dir, temp_dir, workers=4,
                          test_lines=None, sent_len_cutoff=20,
                          shard_start=None, shard_end=None,
                          release_id="2026-05-05"):
    """
    Process all s2orc shards in parallel using ProcessPoolExecutor.
    Each worker: download shard → extract sentences → join metadata → save parquet.

    Uses memory-mapped shared files (NumPy + Feather) so all workers share
    one copy of the metadata in physical RAM via the OS page cache.
    """
    log("=" * 70)
    log("PHASE 1: Processing s2orc shards")
    log("=" * 70)

    output_path = Path(output_dir)
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load metadata ---
    metadata_file = output_path / "cs_metadata_lookup.parquet"
    if not metadata_file.exists():
        log(f"ERROR: Metadata file not found: {metadata_file}")
        log("Run Phase 0 first!")
        sys.exit(1)

    log("Loading metadata lookup...")
    t0 = time.time()
    df_meta = pd.read_parquet(metadata_file)
    log(f"  Loaded {len(df_meta):,} rows in {time.time()-t0:.1f}s")

    # --- Build shared memory-mapped files ---
    ids_path, feather_path = build_shared_files(df_meta, str(output_dir))

    # Free the DataFrame from the main process — workers will mmap the files
    del df_meta
    gc.collect()

    # --- Load s2orc shard URLs (always re-fetch for fresh pre-signed URLs) ---
    release_id = resolve_release_id(api_key, release_id)
    log(f"Fetching fresh s2orc shard URLs from API (release {release_id})...")
    s2orc_info = api_get(f"release/{release_id}/dataset/s2orc", api_key)
    shard_urls = s2orc_info.get("files", [])
    s2orc_urls_path = output_path / "s2orc_shard_urls.json"
    with open(s2orc_urls_path, "w") as f:
        json.dump(shard_urls, f, indent=2)
    log(f"  Saved fresh URLs to {s2orc_urls_path}")

    total_shards = len(shard_urls)
    log(f"  Total s2orc shards: {total_shards}")

    # --- Apply shard range filter ---
    start = shard_start if shard_start is not None else 0
    end = shard_end if shard_end is not None else total_shards - 1
    end = min(end, total_shards - 1)
    shard_indices = list(range(start, end + 1))
    log(f"  Processing shards {start} – {end} ({len(shard_indices)} shards)")

    # --- Check which shards are already done ---
    remaining = []
    already_done = 0
    for idx in shard_indices:
        out_file = output_path / f"output_shard_{idx:04d}.parquet.gz"
        if out_file.exists():
            already_done += 1
        else:
            remaining.append(idx)

    log(f"  Already completed: {already_done}, Remaining: {len(remaining)}")

    # --- Track failures for merge decision ---
    failures = 0

    if not remaining:
        log("All shards already processed!")
    else:
        # --- Prepare worker arguments (lightweight — just paths and indices) ---
        worker_args = []
        for i, idx in enumerate(remaining):
            worker_id = i % workers
            worker_args.append((
                idx, shard_urls[idx], ids_path, feather_path,
                str(output_dir), str(temp_dir), test_lines, sent_len_cutoff, worker_id
            ))

        # --- Launch parallel workers ---
        log(f"Launching {workers} parallel workers for {len(remaining)} shards...")
        t_start = time.time()
        results = []

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single_shard, args): args[0]
                       for args in worker_args}

            for future in as_completed(futures):
                shard_idx = futures[future]
                try:
                    result = future.result(timeout=7200)  # 2h timeout per shard
                    results.append(result)
                    done_count = len(results)
                    elapsed = time.time() - t_start
                    avg_per_shard = elapsed / done_count
                    eta = avg_per_shard * (len(remaining) - done_count)
                    log(f"  Progress: {done_count}/{len(remaining)} "
                        f"({done_count/len(remaining)*100:.1f}%) — "
                        f"ETA: {eta/60:.0f} min")
                except Exception as e:
                    log(f"  Shard {shard_idx}: Worker EXCEPTION — {e}")
                    traceback.print_exc()
                    results.append((shard_idx, 0, f"exception: {e}"))

        # --- Summary ---
        total_time = time.time() - t_start
        total_rows = sum(r[1] for r in results)
        successes = sum(1 for r in results if r[2] in ("done", "skipped"))
        failures = sum(1 for r in results if r[2] not in ("done", "skipped"))

        log("=" * 70)
        log(f"PHASE 1 COMPLETE")
        log(f"  Total time:    {total_time/60:.1f} min")
        log(f"  Shards:        {successes} succeeded, {failures} failed")
        log(f"  Total rows:    {total_rows:,}")
        log(f"  Output dir:    {output_dir}")
        log("=" * 70)

        if failures > 0:
            failed_shards = [r[0] for r in results if r[2] not in ("done", "skipped")]
            log(f"  Failed shards: {failed_shards}")
            log("  Re-run the script to retry failed shards (resume logic will skip completed ones)")

    # --- Cleanup shared files ---
    cleanup_shared_files(output_dir)

    # --- Merge all output shards into a single file ---
    if failures == 0:
        import glob as glob_mod
        shard_files = sorted(glob_mod.glob(str(output_path / "output_shard_*.parquet.gz")))
        if shard_files:
            log(f"Merging {len(shard_files)} output shards...")
            t0 = time.time()

            # Get date range from metadata
            metadata_file = output_path / "cs_metadata_lookup.parquet"
            df_dates = pd.read_parquet(metadata_file, columns=["publicationdate"])
            min_date = df_dates["publicationdate"].min().replace("-", "")
            max_date = df_dates["publicationdate"].max().replace("-", "")
            del df_dates

            merged_name = f"s2orc_cs_sentences_{min_date}_{max_date}.parquet.gz"
            merged_path = output_path / merged_name

            dfs = [pd.read_parquet(f) for f in shard_files]
            df_all = pd.concat(dfs, ignore_index=True)
            del dfs
            df_all.to_parquet(merged_path, index=False, compression="gzip")
            log(f"Merged {len(df_all):,} rows in {time.time() - t0:.0f}s → {merged_path}")
            df_all.head(1000).to_csv(
                output_path / f"s2orc_cs_sentences_{min_date}_{max_date}_sample_sentences.csv",
                index=False
            )
            del df_all
            gc.collect()
    else:
        log("Skipping merge — fix failed shards first, then re-run.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="S2ORC CS Paper Download & Sentence Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline:
  python 00_process_shard.py --api_key KEY --output_dir ./output --temp_dir ./temp

  # Phase 0 only (build metadata):
  python 00_process_shard.py --api_key KEY --phase 0

  # Phase 1 only (process shards), 8 workers:
  python 00_process_shard.py --api_key KEY --phase 1 --workers 8

  # Test run (first 500 lines of first 3 shards):
  python 00_process_shard.py --api_key KEY --test_lines 500 --shard_end 2 --workers 2

  # Resume from shard 100:
  python 00_process_shard.py --api_key KEY --phase 1 --shard_start 100
        """
    )

    parser.add_argument("--api_key", type=str, default=None,
                        help="Semantic Scholar API key (or set S2_API_KEY env var)")
    parser.add_argument("--release_id", type=str, default="2026-05-05",
                        help="S2 release to download (default: 2026-05-05, the snapshot "
                             "used for the paper; 'latest' resolves dynamically)")
    parser.add_argument("--date_cutoff", type=str, default="2015-01-01",
                        help="Keep papers with publicationdate on/after this date, "
                             "YYYY-MM-DD (default: 2015-01-01)")
    parser.add_argument("--output_dir", type=str, default="./data_s2orc/output",
                        help="Directory for output parquet files")
    parser.add_argument("--temp_dir", type=str, default="./data_s2orc/temp",
                        help="Directory for temp downloads")
    parser.add_argument("--phase", type=int, default=None, choices=[0, 1],
                        help="Run only Phase 0 or Phase 1 (default: both)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel worker processes for Phase 1 (default: 4)")
    parser.add_argument("--test_lines", type=int, default=None,
                        help="Limit lines read per shard (for testing)")
    parser.add_argument("--sent_len_cutoff", type=int, default=20,
                        help="Minimum sentence character length (default: 20)")
    parser.add_argument("--shard_start", type=int, default=None,
                        help="First s2orc shard index to process (default: 0)")
    parser.add_argument("--shard_end", type=int, default=None,
                        help="Last s2orc shard index to process (default: last)")

    args = parser.parse_args()

    # --- Resolve API key ---
    api_key = args.api_key or os.environ.get("S2_API_KEY")
    if not api_key:
        print("ERROR: Provide --api_key or set S2_API_KEY environment variable.", flush=True)
        sys.exit(1)

    # --- Paths ---
    output_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    log("S2ORC Pipeline Starting")
    log(f"  Release: {args.release_id}")
    log(f"  Cutoff:  publicationdate >= {args.date_cutoff}")
    log(f"  Output:  {output_dir.resolve()}")
    log(f"  Temp:    {temp_dir.resolve()}")
    log(f"  Workers: {args.workers}")
    if args.test_lines:
        log(f"  TEST MODE: {args.test_lines} lines per shard")
    if args.shard_start is not None or args.shard_end is not None:
        log(f"  Shard range: {args.shard_start or 0} – {args.shard_end or '(end)'}")

    # --- Run phases ---
    if args.phase is None or args.phase == 0:
        log("\n" + "=" * 70)
        log("PHASE 0: Building metadata lookup")
        log("=" * 70)
        phase0_build_metadata(api_key, str(output_dir), str(temp_dir),
                              args.test_lines, args.workers,
                              release_id=args.release_id,
                              date_cutoff=args.date_cutoff)

    if args.phase is None or args.phase == 1:
        log("\n" + "=" * 70)
        log("PHASE 1: Processing s2orc shards")
        log("=" * 70)
        phase1_process_shards(
            api_key, str(output_dir), str(temp_dir),
            workers=args.workers,
            test_lines=args.test_lines,
            sent_len_cutoff=args.sent_len_cutoff,
            shard_start=args.shard_start,
            shard_end=args.shard_end,
            release_id=args.release_id,
        )

    log("Pipeline finished.")


if __name__ == "__main__":
    main()