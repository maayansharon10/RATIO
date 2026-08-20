# Model training (step 3)

`3_train_models_hpo_ddp.py` fine-tunes sentence-transformer embedding models on the datasets built in step 2.

## Why it's built this way

The paper trains the cross product of several base encoders (`all-mpnet-base-v2`, `modernbert-embed-large`, `stella_en_1.5B_v5`) × several relation groups × query setups, and the models range from 100M to 1.5B parameters — so no single set of hyperparameters, batch sizes, or launch modes fits all of them. The script is therefore organized around configuration rather than code:

- **One invocation = one (base model, group) pair.** `--model_name` and `--group_name` select the pair; the full matrix is launched as one SLURM job per pair. Nothing in the code loops over models or groups.
- **Model definitions live in `config/models/configModel_cs2_qt5.json`** — base model path, prompt/column setup per `--query_setup`, similarity function. It contains the three published (encoder, setup) pairs, each matching that encoder's input-prefix scheme: `all-mpnet-base-v2` / `querySentence_prompt` (`query:` / `document:`), `modernbert-embed-large` / `querySentence_prompt_search_query` (`search_query:` / `search_document:`), and `stella_en_1.5B_v5` / `querySentence_group_s2p_query` (`s2p_query`). Add entries here to train other models or setups.
- **Hyperparameters live in `config/config_hpo_cs2_qt5.json`** (`--hpo_configs_path`) as curated trial lists per model, organized in size tiers (`small`/`medium`/`large`/`default`). A top-level `group_to_tier` map routes each group to a tier, so small groups get cheaper trial grids than large ones. The script runs every trial for the pair, scores each on **val cosine-MRR@10**, then re-trains the best config as the final model.
- **Datasets come from step 2's `datasets_info_table`** (`--dataset-info-table`), filtered by `--format` (transductive/inductive) and `--group_name` — the script never touches raw parquet paths directly.

This setup exists because we run many models with many different parameter types; for a smaller use case it can be adjusted with no code changes — point `--hpo_configs_path` at a JSON with a single trial config (a one-element list under `"default"`), and the "HPO" collapses to a single training run.

## Usage

```bash
export PYTHONPATH=$PWD

# Single GPU
python src/train/3_train_models_hpo_ddp.py \
  --model_name all-mpnet-base-v2 --group_name more_generally \
  --query_setup querySentence_prompt --format transductive --no-ddp

# Multi-GPU (DDP), e.g. 4 GPUs
torchrun --nproc_per_node=4 src/train/3_train_models_hpo_ddp.py \
  --model_name modernbert-embed-large --query_setup querySentence_prompt_search_query \
  --group_name more_specific --ddp
```

Defaults reproduce the paper run: `--dataset-info-table` points at the qt5 `temporal_cutoff_2026_valQ4` info table, `--hpo_configs_path` at `config/config_hpo_cs2_qt5.json`.

## What a run does

1. Loads train/val for the (model, group) pair from the info table.
2. For each HPO trial: trains with MultipleNegativesRankingLoss (in-batch negatives), evaluates on val, records the trial in `best_params.jsonl`. Completed trials are detected on disk and skipped, so a resubmitted job resumes where it stopped.
3. Re-trains the best trial as the final model.

Outputs land under the model output path (see `model_utils.get_model_output_path`): `hpo_trail_<n>/` per trial with its `eval/trained/val/` CSV, the final model under `final/`, and `best_params.jsonl` (`pd.read_json(path, lines=True)`).

## Test evaluation (step 4)

Test-set evaluation is a separate script with the same selection args, producing identical outputs to the previous in-training evaluation — per-group `..._results_with_headers.csv` files and a combined `all_model_evals_on_test_*.csv` under `final/eval/trained/test/`. It loads the final model from step 3 and evaluates it on the reported groups' test sets (the three relation groups plus `all` — cross-group transfer):

```bash
python src/eval/4_evaluate_trained_on_test.py \
  --model_name all-mpnet-base-v2 --group_name more_generally \
  --query_setup querySentence_prompt --format transductive
```

## DDP notes (short version)

The in-code docstrings are the authoritative reference; headlines:

- Loss uses `gather_across_devices=True`, so effective in-batch negatives per step = `per_device_train_batch_size × world_size`. Size per-device batch as `desired_negatives / world_size`.
- `modernbert` + gradient checkpointing + DDP needs `_set_static_graph()`, applied via `SetStaticGraphCallback` (transformers 4.48.2 has no `ddp_static_graph` argument). `mpnet` needs `ddp_find_unused_parameters=True` (unused pooler params); `modernbert` requires it `False`.
- Under DDP the per-epoch checkpoint criterion is `eval_loss` (the IR evaluator runs on rank 0 only); single-GPU runs select epochs by val MRR@10 directly. Trial selection is by val MRR@10 in both modes.
- Gradient checkpointing is enabled only for `modernbert`/`stella` (mpnet doesn't need it).

The pinned versions in `requirements-train.txt` matter: the DDP workarounds are specific to sentence-transformers 5.4.0 / transformers 4.48.2 / accelerate 1.4.0.

## Adjusting for your own setup

- **One model, no HPO**: single trial config under `"default"` in your HPO JSON.
- **No cluster**: everything runs single-process with `--no-ddp`; DDP is opt-in.
- **No wandb**: set `WANDB_MODE=offline` (or `disabled`) — the script logs to wandb by default.
- **Different data**: any parquet datasets work as long as they're described by an info table with the step 2 schema (`groupName`, `format`, `*_queries_gold_path`, `*_candidates_path`, `temporal_version`).
