# S2ORC CS sentence corpus (Stage 0)

Downloads one full [Semantic Scholar S2ORC](https://www.semanticscholar.org/product/api) release and extracts every body-text sentence of Computer Science papers with its ±1-sentence context and paper metadata, saved as compressed parquet. The corpus used in the paper was built from S2 release **2026-05-05**, keeping CS papers with `publicationdate >= 2015-01-01` — the scripts default to exactly these settings. All commands below run from the repository root.

## Requirements

- Python 3.9+ and the packages in `requirements.txt`
- The scispaCy sentencizer model (used for sentence splitting in the paper run):

```bash
pip install -r src/data/s2orc_data/requirements.txt
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

(Pick the `en_core_sci_sm` version matching your spaCy from the [scispaCy releases](https://github.com/allenai/scispacy). If the model is missing, the pipeline falls back to spaCy's blank-English sentencizer, which splits slightly differently.)

- A Semantic Scholar API key with **bulk dataset access**, exported as `S2_API_KEY` (or passed via `--api_key`). The datasets API is rate-limited to 1 request/second; the script handles this. Actual shard downloads are pre-signed S3 URLs and are not rate-limited.
- Disk: s2orc shards are ~4 GB compressed each and papers shards ~1.5 GB; the full 2026-05-05 release is several hundred GB of downloads in total. Temp files are deleted after each shard.

## Stage 0 — Build the sentence corpus

```bash
export S2_API_KEY=your_key
```

**1. Fetch shard URLs and sample data**

```bash
python src/data/s2orc_data/00_get_releases.py
```

Verifies API access for release 2026-05-05, saves the shard URL lists to `shard_urls.json`, and downloads one sample shard per dataset so the record schema can be inspected.

**2. Test run**

```bash
python src/data/s2orc_data/00_process_shard.py --test_lines 500 --workers 2
```

Runs both phases end-to-end on the first 500 lines of a few shards, to verify the schema, filters, and output format before committing to the full download.

**3. Full run (Phase 0 + Phase 1)**

Edit the repo path, venv path, and API key in `run_00_process_shard.sh` (its resource requests match the paper run: 63 CPUs, 400 GB RAM, 7-day limit) and submit:

```bash
sbatch src/data/s2orc_data/run_00_process_shard.sh
```

The script runs Phase 0 (metadata build, `--workers 60`) then Phase 1 (sentence extraction, `--workers 50`). Without SLURM, run the same two `python` commands from the script directly.

Outputs, written to `./data_s2orc/output/`:

- `cs_metadata_lookup.parquet` — one row per selected CS paper (Phase 0)
- `output_shard_XXXX.parquet.gz` — sentences per s2orc shard (Phase 1)
- `s2orc_cs_sentences_{min_date}_{max_date}.parquet.gz` — merged corpus, plus a 1000-row sample CSV

Useful flags: `--phase {0,1}` to run one phase, `--test_lines N` for a small test run, `--sent_len_cutoff` (default 20), `--date_cutoff` (default 2015-01-01), and `--release_id latest` to run against the current release instead of the pinned snapshot.

Stage 0 is replaceable: Stage 1 works on any sentence corpus, not just S2ORC. To use your own sentences, skip Stage 0 and place parquet file(s) in `--shards-dir` following the same final structure — filename `s2orc_cs_sentences_<YYYYMMDD>_<YYYYMMDD>.parquet.gz` and at minimum the columns `single_text` (one sentence), `multi_text` (previous + sentence + next), and `corpusid` (any unique document id). `publicationdate` is recommended (used for the dated output filename), and all other columns are carried through to the filtered output.

## Output schema (Stage 0)

| Column            | Type      | Description                                                                 |
|-------------------|-----------|-----------------------------------------------------------------------------|
| `corpusid`        | int       | Semantic Scholar corpus ID                                                  |
| `single_text`     | str       | One sentence from the paper body                                            |
| `multi_text`      | str/None  | prev + current + next sentence; `None` for the first/last sentence of a paragraph or if the paragraph has <3 sentences |
| `section`         | str       | Nearest preceding section header (may be empty)                             |
| `field`           | list[str] | All `s2fieldsofstudy` categories (papers kept iff they include "Computer Science") |
| `journal`         | str       | Journal name                                                                |
| `venue`           | str       | Venue name                                                                  |
| `year`            | int       | Publication year                                                            |
| `authors`         | list[str] | Author names                                                                |
| `in_citations`    | int       | `citationcount` from the papers record                                      |
| `out_citations`   | int       | `referencecount` from the papers record                                     |
| `publicationdate` | str       | `YYYY-MM-DD`                                                                |

Sentence filter: length ≥ 20 characters and first character is an uppercase letter or digit. Paragraph and section-header spans come from the s2orc `content.annotations` offsets; context never crosses paragraph boundaries.

## Notes

- **Resumable:** in Stages 0 and 1, per-shard outputs are written once at the end; on re-run, existing outputs (including empty markers) are skipped, so a crashed job can simply be restarted.
- **Memory:** Phase 1 workers share the metadata via a memory-mapped sorted-ID numpy array plus a Feather file, so RAM usage stays flat as `--workers` grows.
