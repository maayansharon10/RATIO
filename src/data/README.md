# S2ORC CS Sentence Extraction Pipeline

Builds a sentence-level corpus of Computer Science papers from [Semantic Scholar S2ORC](https://www.semanticscholar.org/product/api) and filters it down to sentences that open with an ideation-relation queryTerm (e.g. "in contrast,", "more generally,"). Three stages:

0. **Corpus build** — `00_process_shard.py` downloads one full S2ORC release and extracts every body-text sentence with its ±1-sentence context and paper metadata, saved as compressed parquet.
1. **QueryTerm filtering** — `1_filter_sentences_by_queryTerms.py` keeps sentences starting with a known queryTerm, splits them into querySentence / goldSentence with prompt-variant columns, and mines unmatched sentences for potential new queryTerms.
2. **Dataset construction** — `2_split_filterGroups_construct_datasets_temporal.py` deduplicates pairs, splits temporally into train/val/test with leakage checks, and filters per-relation-group sub-datasets.

The corpus used in the paper was built from S2 release **2026-05-05**, keeping CS papers with `publicationdate >= 2015-01-01`. The scripts default to exactly these settings.

Model training on the resulting datasets (step 3) and test evaluation (step 4) are documented separately in [src/train/TRAINING.md](../train/TRAINING.md).

## Repository layout

```
ratio/
├── config/
│   ├── ProjectDataConfig.py              # config loader used by Stages 1-2
│   ├── data/configData_cs2_qt5.json      # Stage 1 configuration (columns, paths, queryTerm groups)
│   └── models/configModel_cs2_qt5.json   # model definitions + prompt/column setups (steps 3-4)
├── data/raw/queryTerms/qt5/all_query_terms5.csv   # queryTerms list (word, group, special_word)
├── src/
│   ├── data/
│   │   ├── s2orc_data/                   # Stage 0: S2ORC download + sentence extraction (own README)
│   │   ├── 1_filter_sentences_by_queryTerms.py    # Stage 1: filter by queryTerms, split query/gold
│   │   ├── 2_split_filterGroups_construct_datasets_temporal.py   # Stage 2: dedup, temporal split, per-group datasets
│   │   ├── run_1_filter_sentences.sh     # SLURM job for Stage 1 (paper-run resources)
│   │   └── run_2_create_datasets.sh      # SLURM job for Stage 2 (paper-run resources)
│   ├── train/3_train_models_hpo_ddp.py   # Step 3: HPO + DDP training — see train/TRAINING.md
│   ├── eval/4_evaluate_trained_on_test.py   # Step 4: test-set eval of a trained final model
│   ├── eval/eval_utils.py                # IR evaluator + eval-CSV helpers (steps 3-4)
│   └── utils.py                          # shared helpers
├── requirements-train.txt                # training-only deps (pinned)
└── src/data/s2orc_data/requirements.txt  # data-pipeline deps (Stages 0-2)
```

Run everything from the repository root with `PYTHONPATH` set to it. Data-pipeline dependencies: `pip install -r src/data/s2orc_data/requirements.txt` (see [s2orc_data/README.md](s2orc_data/README.md) for the scispaCy model, API key, and disk requirements).

## Stage 0 — Build the sentence corpus

Lives in [`s2orc_data/`](s2orc_data/README.md) — see its README for the full workflow (verify access → test run → full SLURM run), the output schema, and how to substitute your own sentence corpus.

## Stage 1 — Filter sentences by queryTerms

Configuration lives in `config/data/configData_cs2_qt5.json` (column names, output dirs, queryTerms CSV path), loaded via `config/ProjectDataConfig.py` with helpers in `src/utils.py` — run from the repo root with `PYTHONPATH` set. The queryTerms list lives at `data/raw/queryTerms/qt5/all_query_terms5.csv`, with columns `word`, `group`, `special_word`; the `blacklist` group marks excluded terms.

```bash
export PYTHONPATH=$PWD
python src/data/1_filter_sentences_by_queryTerms.py \
    --config-path config/data/configData_cs2_qt5.json \
    --shards-dir ./data_s2orc/output --num_workers 60
```

On SLURM, edit the paths in `src/data/run_1_filter_sentences.sh` (68 CPUs, 350 GB RAM, 2-day limit — the paper run) and `sbatch` it.

The script discovers the corpus in `--shards-dir` (the merged `.parquet.gz` from Stage 0, or `.parquet.zst` shards with `.done` markers). A single file is streamed row-group by row-group across workers; multiple files run one worker per file. Outputs:

- `s2orc_filtered_sentences_..._{minpub}_{maxpub}.parquet.gz` (path from config) — matched rows with `queryTerm`, `queryTerm_group`, `querySentence`, `goldSentence`, and prompt-variant columns
- `{shards-dir}/potential_queryTerms_distribution.csv` / `.parquet` — corpus-wide frequencies of unmatched leading phrases
- `{shards-dir}/potential_queryTerms_for_llm.parquet` — thresholded candidates (`--potential-min-count`, default 50) with sample sentences, input for the downstream LLM queryTerm classifier

Other flags: `--potential-samples-per-term` (default 5), `--keep-temps` to keep per-shard temp files. Resume: completed shards (filtered parquet + extraction marker) are skipped on re-run.

## Stage 2 — Construct train/val/test datasets

Splits the Stage 1 output into temporally disjoint train/val/test sets and per-relation-group sub-datasets. The paper run uses `temporal_cutoff_2026_valQ4`: train = publications through 2025 Q3, val = 2025 Q4, test = 2026 onward. Before splitting, pairs are deduplicated on (`querySentence`, `goldSentence`), and pair-level leakage checks assert that no pair appears in more than one split (globally and per group).

```bash
export PYTHONPATH=$PWD
python src/data/2_split_filterGroups_construct_datasets_temporal.py \
  --config-data-path config/data/configData_cs2_qt5.json \
  --split-mode temporal_cutoff_2026_valQ4 \
  --date-from 20150101 --date-to 20260601
```

`--date-from`/`--date-to` resolve the Stage 1 output filename (its publication-date suffix). On SLURM, edit the paths in `src/data/run_2_create_datasets.sh` (1 CPU, 300 GB RAM, 2-day limit — the paper run) and `sbatch` it.

Outputs, under `data/original_split/cs2_qt5/temporal_cutoff_2026_valQ4/`:

- `<group>/<split>/s2orc_filtered__...__<group>_<split-mode>__<split>.parquet.gz` — train/val/test for "all" plus every queryTerm group defined in the config (e.g. `explicit_contrast`, `similarity`, `allButContrast`)
- `<group>/queryTermCompareTable/...` — per-queryTerm counts and proportions across the three splits
- `all/datasets_info_table/...` — paths and sizes for every (transductive/inductive, group, split) combination, consumed by the training scripts

Other options: `--split-mode` also supports `random` (GroupShuffleSplit by corpusid), `temporal_midyear2025`, and `temporal_cutoff_2026_valH2`; `--resume` reloads an existing global split from disk and skips the expensive load + dedup + split step.
