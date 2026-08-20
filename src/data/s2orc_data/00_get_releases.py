#!/usr/bin/env python3
"""
00_get_releases.py — step 1: verify access and inspect the data

Fetches the shard URL lists for the 'papers' and 's2orc' datasets of one
Semantic Scholar release, saves them to shard_urls.json, and downloads one
sample shard of each so the record schema can be inspected before running
the full pipeline. (00_process_shard.py fetches its own shard URLs, so this
script only verifies access and schema.)

Inputs:
    --api_key      Semantic Scholar API key with bulk dataset access
                   (or set the S2_API_KEY environment variable)
    --release_id   Release to inspect (default: 2026-05-05, the snapshot used
                   for the paper; pass 'latest' to resolve dynamically)
    --data_dir     Output directory (default: ./data_s2orc)

Outputs:
    {data_dir}/shard_urls.json
    {data_dir}/sample/sample_papers_0.jsonl.gz
    {data_dir}/sample/sample_s2orc_0.jsonl.gz

Usage:
    python 00_get_releases.py --api_key YOUR_KEY
    python 00_get_releases.py --release_id latest --skip_download
"""

import argparse
import gzip
import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "https://api.semanticscholar.org/datasets/v1"


def get_headers(api_key: str) -> dict:
    return {"x-api-key": api_key}


def resolve_release_id(release_id: str) -> str:
    """Return a concrete release ID, resolving the literal 'latest' via the API.
    The releases list endpoint is public — no auth needed."""
    if release_id != "latest":
        return release_id
    print(f"Resolving latest release from {BASE_URL}/release/ ...")
    response = requests.get(f"{BASE_URL}/release/")
    response.raise_for_status()
    all_releases = sorted(response.json())
    print(f"  Total releases: {len(all_releases)}")
    print(f"  Latest release: {all_releases[-1]}")
    return all_releases[-1]


def get_shard_urls(api_key: str, release_id: str, dataset_name: str) -> list[str]:
    """Return the list of shard download URLs for a dataset in a release.
    Requires an API key with bulk dataset access. Rate limit: 1 request/second."""
    url = f"{BASE_URL}/release/{release_id}/dataset/{dataset_name}"
    print(f"\nFetching shard URLs for '{dataset_name}' at release {release_id} ...")
    time.sleep(1)  # respect 1 req/sec rate limit
    response = requests.get(url, headers=get_headers(api_key))
    response.raise_for_status()

    data = response.json()
    files = data.get("files", [])
    print(f"  Shards available: {len(files)}")
    print(f"  Description: {data.get('description', '')}")
    return files


def download_file(url: str, dest_path: Path, label: str = "") -> None:
    """Download a pre-signed S3 URL to dest_path (no API key needed)."""
    print(f"\nDownloading {label}: {dest_path.name}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {downloaded / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.1f}%)",
                          end="\r", flush=True)
    print(f"\n  Done: {dest_path}")


def peek_records(path: Path, n: int = 3) -> None:
    """Print the first N JSON records from a .jsonl.gz file for schema inspection."""
    print(f"\n--- First {n} records from {path.name} ---")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            record = json.loads(line)
            print(json.dumps(record, indent=2)[:2000])  # cap at 2000 chars per record
            print("---")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch S2ORC shard URLs and download one sample shard per dataset")
    parser.add_argument("--api_key", type=str, default=os.environ.get("S2_API_KEY"),
                        help="Semantic Scholar API key (or set S2_API_KEY env var)")
    parser.add_argument("--release_id", type=str, default="2026-05-05",
                        help="Release ID to inspect (default: 2026-05-05; 'latest' resolves dynamically)")
    parser.add_argument("--data_dir", type=str, default="./data_s2orc",
                        help="Output directory for shard_urls.json and sample shards")
    parser.add_argument("--skip_download", action="store_true",
                        help="Only fetch shard URLs, skip downloading sample files")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("API key required. Pass --api_key or set S2_API_KEY env var.")

    data_dir = Path(args.data_dir)
    sample_dir = data_dir / "sample"
    data_dir.mkdir(parents=True, exist_ok=True)

    release_id = resolve_release_id(args.release_id)

    shard_urls = {}
    for dataset_name in ["papers", "s2orc"]:
        try:
            urls = get_shard_urls(args.api_key, release_id, dataset_name)
            shard_urls[dataset_name] = urls
        except requests.HTTPError as e:
            print(f"  ERROR fetching {dataset_name} shards: {e}")
            print("  -> Bulk dataset access may not be enabled for this API key.")
            print("     Request access at: https://www.semanticscholar.org/product/api")
            shard_urls[dataset_name] = []

    shard_urls_path = data_dir / "shard_urls.json"
    with open(shard_urls_path, "w") as f:
        json.dump({"release": release_id, "shards": shard_urls}, f, indent=2)
    print(f"\nSaved shard URLs to {shard_urls_path}")
    print(f"  papers shards: {len(shard_urls.get('papers', []))}")
    print(f"  s2orc shards:  {len(shard_urls.get('s2orc', []))}")

    if args.skip_download:
        print("Skipping sample download.")
        return

    for dataset_name in ["papers", "s2orc"]:
        urls = shard_urls.get(dataset_name, [])
        if not urls:
            print(f"\nNo URLs available for {dataset_name}, skipping sample download.")
            continue
        dest = sample_dir / f"sample_{dataset_name}_0.jsonl.gz"
        download_file(urls[0], dest, label=dataset_name)
        peek_records(dest, n=3)


if __name__ == "__main__":
    main()
