"""
3_train_models_hpo_ddp.py — step 3 of the pipeline: single-GPU and multi-GPU (DDP)
training of sentence-transformer embedding models with manual HPO over curated
trial configs loaded from a JSON file.

One invocation trains one (base model, queryTerm group) pair with one query setup:
it runs every HPO trial for that pair, selects the best by val cosine-MRR@10,
and final-trains it. Test-set evaluation runs separately (4_evaluate_trained_on_test.py).
The paper's models are the cross product of base models and relation groups,
launched as one job per pair.

Inputs:
    --config-model-path    Model definitions + prompt/column setups (JSON)
    --hpo_configs_path     Curated HPO trial configs per model and group tier (JSON)
    --dataset-info-table   datasets_info_table parquet from step 2
    --model_name / --group_name / --query_setup / --format   which pair to train

Outputs (under the model output path from model_utils.get_model_output_path):
    hpo_trail_<n>/         checkpoint + val eval per trial
    final/                 re-trained best trial (test eval: 4_evaluate_trained_on_test.py)
    best_params.jsonl      one record per trial + final (pd.read_json(lines=True))

DDP usage:
    Single GPU (default):
        python src/train/3_train_models_hpo_ddp.py --group_name more_generally --no-ddp ...

    Multi-GPU (DDP):
        torchrun --nproc_per_node=4 src/train/3_train_models_hpo_ddp.py --group_name more_specific --ddp ...

Cross-device negatives under DDP (sentence-transformers 5.4.0):
    `MultipleNegativesRankingLoss` is constructed with gather_across_devices=True,
    so embeddings ARE all-gathered across GPUs before the loss (all_gather_with_grad).
    Effective in-batch negatives per step = per_device_train_batch_size * world_size.
    e.g. per_device_batch=16 on 4 GPUs => 64 negatives (NOT 16). Set per-device batch
    to (desired_negatives / world_size). NOTE: HPO config "effective batch" comments
    predate gathering and may be stated per-device — multiply by world_size for the
    true negatives count.

Gradient checkpointing + DDP (transformers 4.48.2, accelerate 1.4.0):
    modernbert-embed-large with gradient_checkpointing=True under DDP raises either
    `CheckpointError` (non-reentrant recompute tensor mismatch) or "marked ready twice"
    (reentrant double hook fire). Both PyTorch error messages point to the same
    workaround: call `_set_static_graph()` on the DDP-wrapped model. This transformers
    version has NO `ddp_static_graph` TrainingArguments flag (added in a later release),
    so we set it via SetStaticGraphCallback.on_train_begin instead, right after HF
    Trainer wraps the model in DistributedDataParallel. Static graph requires the graph
    be identical every iteration (true here: NO_DUPLICATES sampler, dataloader_drop_last
    set by DDP, trials differ only in hyperparams) and no unused params (true for
    modernbert -> ddp_find_unused_parameters=False).

HPO configs:
    Curated trial configs live in `--hpo_configs_path` (default: config/config_hpo_cs2_qt5.json).
    Structure: {model_name: {"small"|"medium"|"large"|"default": [config_dict, ...]}}.
    The lookup key is `--model_name` (same convention as configModel_cs2_qt5.json).
    `group_to_tier` (a top-level JSON map {group_name: tier}) is the single source of
    truth for which tier each group uses; a group not listed there falls back to "large".
"""

import argparse
import gc
import json
import os
import sys
import pandas as pd
from datetime import datetime
import wandb
from transformers import TrainerCallback
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainer
from sentence_transformers.sentence_transformer.training_args import BatchSamplers
from transformers.trainer_utils import get_last_checkpoint
from src.dataset_utils import load_parquet_to_dataset
from src.eval.eval_utils import get_ir_evaluator_set
import src.utils as utils
import src.model_utils as model_utils


# ============================================================
# DDP helpers
# ============================================================

def get_world_size():
    """Return WORLD_SIZE from env, default 1 (non-DDP)."""
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    """Return RANK from env, default 0 (non-DDP or main process)."""
    return int(os.environ.get("RANK", 0))


def get_local_rank():
    """Return LOCAL_RANK from env, default 0."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process():
    """True if this is rank 0 (or single-process run)."""
    return get_rank() == 0


def is_ddp_active():
    """True if running under torchrun with WORLD_SIZE > 1 and dist is initialized."""
    return get_world_size() > 1 and torch.distributed.is_initialized()


def ddp_barrier():
    """Synchronization barrier. No-op if DDP is not active."""
    if is_ddp_active():
        torch.distributed.barrier()


def ddp_broadcast_object(obj, src=0):
    """Broadcast a python object from src rank to all ranks. No-op if DDP inactive."""
    if not is_ddp_active():
        return obj
    obj_list = [obj if get_rank() == src else None]
    torch.distributed.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def print_main(*args_, **kwargs_):
    """Print only on rank 0."""
    if is_main_process():
        print(*args_, **kwargs_)


def validate_ddp_environment(use_ddp):
    """Validate the DDP environment matches the --ddp flag. Exit on mismatch."""
    visible_gpus = torch.cuda.device_count()
    world_size = get_world_size()
    rank = get_rank()
    local_rank = get_local_rank()

    print_main(f"========================================")
    print_main(f"GPU / DDP environment check")
    print_main(f"========================================")
    print_main(f"  --ddp flag:                {use_ddp}")
    print_main(f"  torch.cuda.device_count(): {visible_gpus}")
    print_main(f"  WORLD_SIZE (env):          {world_size}")
    print_main(f"  RANK (env):                {rank}")
    print_main(f"  LOCAL_RANK (env):          {local_rank}")
    print_main(f"========================================", flush=True)

    if use_ddp:
        if world_size < 2:
            print_main(
                f"\n[FATAL] --ddp was set but WORLD_SIZE={world_size}.\n"
                f"  Launch with torchrun, e.g.:\n"
                f"      torchrun --nproc_per_node=4 src/train/3_train_models_hpo_ddp.py --ddp ...\n"
                f"  And ensure SLURM allocated multiple GPUs (#SBATCH --gres=gpu:N, N>=2).\n",
                flush=True
            )
            sys.exit(1)
        if visible_gpus < 2:
            print_main(
                f"\n[FATAL] --ddp was set but only {visible_gpus} GPU(s) visible to this process.\n"
                f"  Check SLURM allocation and CUDA_VISIBLE_DEVICES.\n",
                flush=True
            )
            sys.exit(1)
        print_main(f"[OK] DDP environment validated. Running with world_size={world_size}.", flush=True)
    else:
        if world_size > 1:
            print_main(
                f"\n[WARNING] --ddp was NOT set but WORLD_SIZE={world_size}.\n"
                f"  This means torchrun was used without --ddp. Add --ddp or launch with\n"
                f"  plain `python` instead of torchrun.\n",
                flush=True
            )
        print_main(f"[OK] Running in single-process mode.", flush=True)


# ============================================================
# HPO config loading
# ============================================================

def load_hpo_configs(hpo_configs_path, model_name, group_name):
    """
    Load curated HPO trial configs for (model_name, group_name) from the JSON file.

    Lookup rule:
      1. `model_name` is used as a direct key into the JSON (same convention as
         configModel_cs2_qt5.json["models"][model_name]).
      2. Within the model entry, pick the size branch:
           - If "default" exists, use it.
           - Else if group_name is in group_to_tier, use the tier it maps to.
           - Else use "large".

    `group_to_tier` is the single top-level JSON map {group_name: tier} that defines
    every group's tier explicitly. A group not present in the map falls back to "large".
    (This replaces the older small_group/medium_group sets, which routed groups
    implicitly; all group->tier assignments now live in this one map.)

    Returns: list of trial config dicts.
    """
    with open(hpo_configs_path, 'r') as f:
        all_configs = json.load(f)

    group_to_tier = all_configs.get("group_to_tier", {})

    if model_name not in all_configs:
        available = [k for k in all_configs
                     if not k.startswith("_") and k != "group_to_tier"]
        raise ValueError(
            f"No HPO configs found for model_name={model_name!r}. "
            f"Available model keys in {hpo_configs_path}: {available}"
        )

    model_entry = all_configs[model_name]
    if "default" in model_entry:
        size_key = "default"
    elif group_name in group_to_tier:
        size_key = group_to_tier[group_name]
    else:
        size_key = "large"

    if size_key not in model_entry:
        raise ValueError(
            f"HPO configs for model_name={model_name!r} have no '{size_key}' branch. "
            f"Available branches: {list(model_entry.keys())}"
        )

    configs = model_entry[size_key]
    print_main(f"[HPO] loaded {len(configs)} trial configs for model_name={model_name!r}, "
               f"size_key={size_key!r}, group_name={group_name!r}", flush=True)
    return configs


# ============================================================
# HPO record keeping
# ============================================================

def save_params(trail_params, trail_num, best_params, best_score, output_model_path,
                comment, current_score=None, is_final=False):
    """Rank-0 only: append an HPO trial record to best_params.jsonl. The rank-0
    guard prevents the DDP write race.

    Load later with: pd.read_json(path, lines=True)
    """
    if not is_main_process():
        return
    if not os.path.isdir(output_model_path):
        return

    # --- best_params.jsonl ---
    try:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "trial_num": trail_num,
            "comment": comment,
            "is_new_best": (comment == "found new best!"),
            "is_final": is_final,
            "trial_score": current_score,
            "trial_params": trail_params,
            "best_score": best_score,
            "best_params": best_params,
        }
        out_path = utils.create_path_from_dir_filename(output_model_path, "best_params.jsonl")
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"----- params saved to: {output_model_path} (jsonl)\n"
              f"trail_num {trail_num}\nbest_score {best_score}", flush=True)
    except Exception as e:
        print(f"error saving best_params.jsonl to {output_model_path}: {e}", flush=True)


# ============================================================
# Training helpers
# ============================================================

class SetStaticGraphCallback(TrainerCallback):
    """Apply _set_static_graph() to the DDP-wrapped model after HF Trainer wraps it.

    !!! GOTCHA — DO NOT revert to checking the `model` kwarg !!!
    The on_train_begin `model` kwarg is the UNWRAPPED SentenceTransformer, NOT the
    DistributedDataParallel module. A previous version did
    `isinstance(model, DistributedDataParallel)` on that kwarg, which is ALWAYS False,
    so _set_static_graph() silently never ran (logs showed "model is not DDP-wrapped
    (type=SentenceTransformer)" on every rank) and the safeguard was dead code. The DDP
    wrapper lives at `trainer.model_wrapped`, which only exists after train() wraps it —
    so we close over the trainer (passed to __init__) and read model_wrapped here.
    If model_wrapped is also not DDP, accelerate isn't DDP-wrapping at all and static
    graph is impossible regardless; in that case disable gradient_checkpointing instead.

    Why static graph at all: transformers 4.48.2 has NO `ddp_static_graph` TrainingArguments
    flag (added in a later release), so we set it directly on the DDP module instead.
    gradient_checkpointing=True + DDP on modernbert-embed-large raised either CheckpointError
    (non-reentrant recompute tensor mismatch, 36 vs 26) or "Parameter ... marked as ready
    twice" (reentrant double hook fire). Both PyTorch messages recommend _set_static_graph():
    it records the autograd graph once on iter 1 and replays it, so the reducer stops
    dynamically re-discovering ready params (which is what double-counted).

    Preconditions (all hold here): graph identical every iteration (NO_DUPLICATES sampler,
    dataloader_drop_last set by DDP, trials differ only in hyperparams) and no unused params
    (modernbert -> ddp_find_unused_parameters=False; do NOT combine static graph with
    find_unused_parameters=True).

    Register AFTER the trainer is built: trainer.add_callback(SetStaticGraphCallback(trainer)).
    No-op when model_wrapped is not a DDP instance (single-process run), so safe to register.
    """

    def __init__(self, trainer=None):
        self._trainer = trainer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        wrapped = self._trainer.model_wrapped if self._trainer is not None else model
        if isinstance(wrapped, torch.nn.parallel.DistributedDataParallel):
            wrapped._set_static_graph()
            print(f"[rank {get_rank()}] SetStaticGraphCallback: _set_static_graph() applied "
                  f"to DDP model", flush=True)
        else:
            print(f"[rank {get_rank()}] SetStaticGraphCallback: model_wrapped is "
                  f"{type(wrapped).__name__}, not DDP; static graph not applied", flush=True)


def safe_train(trainer, output_model_path):
    last_checkpoint = None
    if os.path.isdir(output_model_path):
        last_checkpoint = get_last_checkpoint(output_model_path)

    # Validate the checkpoint has model weights, not just trainer state
    if last_checkpoint is not None:
        weight_files = ["model.safetensors", "pytorch_model.bin",
                        "model.safetensors.index.json", "pytorch_model.bin.index.json"]
        has_weights = any(os.path.exists(os.path.join(last_checkpoint, f)) for f in weight_files)
        if not has_weights:
            print_main(f"safe_train: checkpoint {last_checkpoint} is corrupt (no weights). "
                       f"Ignoring and training from scratch.", flush=True)
            last_checkpoint = None

    if last_checkpoint:
        print_main(f"safe_train: Resuming from last checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print_main("safe_train: No valid checkpoint found. Training from scratch.")
        trainer.train()


def train_model(model_info, model_name, hyperparams,
                val_dataset, train_dataset, ir_evaluator_dev,
                output_model_path):
    """
    Train a single (model, hyperparams) configuration.

    Device placement: each rank loads the model onto its own LOCAL_RANK GPU. Under DDP,
    HF Trainer wraps the model in DistributedDataParallel automatically.

    Loss: MultipleNegativesRankingLoss with gather_across_devices=True (ST 5.4.0),
    so in-batch negatives are gathered across all GPUs: effective negatives per step
    = per_device_train_batch_size * world_size.
    """
    run_name = model_name

    if is_main_process():
        print(f"init run_name {run_name}")
        wandb.init(name=run_name, reinit=True)

    torch.cuda.empty_cache()

    warmup_kwargs = {}
    if "warmup_steps" in hyperparams and hyperparams["warmup_steps"] > 0:
        warmup_kwargs["warmup_steps"] = hyperparams["warmup_steps"]
    else:
        warmup_kwargs["warmup_ratio"] = hyperparams.get("warmup_ratio", 0.1)

    print_main(f"----- load TrainingArguments", flush=True)

    if "mpnet" in model_name:
        ddp_find_unused_parameters = True  # all-mpnet-base-v2 has a final pooler (params 197/198) that doesn't receive grad under MultipleNegativesRankingLoss, so DDP's reducer hangs/errors unless we expect unused params. HF Trainer ignores this in single-process mode.
    elif "modernbert" in model_name:
        ddp_find_unused_parameters = False  # modernbert-embed-large has no such pooler, so all params receive grad and DDP reducer is happy with find_unused_params=False. REQUIRED False for static graph (static graph is incompatible with find_unused_parameters=True).
    else:
        ddp_find_unused_parameters = True  # default to True for safety; if the model has no unused params, DDP ignores this and runs fine.
    use_grad_checkpointing = "modernbert" in model_name or "stella" in model_name
    # Precision: BF16 for all models — the L40S GPUs support bf16 natively, and the
    # encoders (modernbert-embed-large, stella_en_1.5B) are bf16-pretrained, so bf16
    # avoids the fp16 overflow/NaN risk with no memory cost.

    # Inner (per-epoch) checkpoint selection criterion:
    # - Under DDP this CANNOT be the IR metric: the dev IR evaluator runs on rank 0
    #   only, so 'eval_val_cosine_mrr@10' is absent from the metrics dict that HF's
    #   _determine_best_metric inspects on the other ranks -> KeyError. DDP therefore
    #   keeps the default criterion (eval_loss, logged on every rank).
    # - Single-process runs have the IR metric available, so the best epoch is picked
    #   by val MRR@10 directly — consistent with the outer HPO criterion.
    # Env-based DDP detection: torch.distributed is NOT initialized yet at this
    # point (the process group is created inside TrainingArguments.__post_init__),
    # so is_ddp_active() would wrongly return False here under torchrun.
    ddp_run = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp_run:
        best_metric_kwargs = {}
    else:
        best_metric_kwargs = {
            "metric_for_best_model": "eval_val_cosine_mrr@10",
            "greater_is_better": True,
        }
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_model_path,
        num_train_epochs=hyperparams["num_train_epochs"],
        per_device_train_batch_size=hyperparams["batch_size"],
        per_device_eval_batch_size=hyperparams["batch_size"],
        learning_rate=hyperparams["learning_rate"],
        **warmup_kwargs,
        max_grad_norm=hyperparams.get("max_grad_norm", 1.0),
        weight_decay=hyperparams.get("weight_decay", 0.0),
        fp16=False,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        gradient_checkpointing=use_grad_checkpointing,
        # use_reentrant=False is the sentence-transformers-blessed setting for
        # checkpointing + DDP + MNRL (see UKPLab/sentence-transformers#2844). For
        # modernbert it additionally needs static graph, applied via
        # SetStaticGraphCallback below (no ddp_static_graph arg in transformers 4.48.2).
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=hyperparams["gradient_accumulation_steps"],
        save_strategy="epoch",
        save_total_limit=2,
        run_name=run_name,
        report_to="wandb",
        disable_tqdm=True,
        load_best_model_at_end=True,
        # Per-epoch best-checkpoint criterion: val MRR@10 when single-process,
        # eval_loss under DDP (IR metric is rank-0-only there; see best_metric_kwargs
        # above). Outer HPO selects across trials by VAL cosine_mrr@10.
        **best_metric_kwargs,
        ddp_find_unused_parameters=ddp_find_unused_parameters,
        ddp_timeout=28800,
    )

    # Per-rank device placement: each rank uses its own LOCAL_RANK GPU.
    local_rank = get_local_rank()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    print(f"[rank {get_rank()}] loading model onto {device}", flush=True)

    print_main(f"----- load model", flush=True)
    model = model_utils.load_model_sentence_transformers(
        model_name=model_name,
        model_path=model_info['base_model_path'],
        similarity_fn_name="cosine",
        device=device,
        is_eval=False
    )

    # sentence-transformers 5.4.0: gather embeddings across devices so in-batch
    # negatives are pooled from all GPUs (effective negatives = batch * world_size).
    loss = MultipleNegativesRankingLoss(model, gather_across_devices=True)

    print_main(f"----- load trainer", flush=True)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=loss,
        evaluator=ir_evaluator_dev,
    )
    # The static-graph callback is registered AFTER the trainer is built (see
    # below) — a trainer-less SetStaticGraphCallback() is a guaranteed no-op.

    if is_ddp_active() and use_grad_checkpointing and "modernbert" in model_name:
        trainer.add_callback(SetStaticGraphCallback(trainer))

    utils.clean_cache()
    print_main(f"training")

    print_main("safe_train(...) => either resume or fresh-train")
    print(f"---- start time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    safe_train(trainer, output_model_path)

    # Save manually only on rank 0 (HF Trainer's own save_model is already rank-0-guarded).
    if is_main_process():
        model.save_pretrained(output_model_path)
        print(f"----- saved model to output_model_path {output_model_path}", flush=True)

    # Wait so all ranks reach the same point before moving on.
    ddp_barrier()

    return model


def train_single_model_on_single_dataset_and_evaluate(model_info, dataset_info, hpo_configs_path,
                                                       model_name_arg, ddp=False):
    """
    Run HPO over curated configs for (model, group), then final-train the best config.

    HPO loop runs on all ranks in lockstep. Each trial's training is DDP-coordinated.
    Evaluator runs on rank 0 only (with a barrier so other ranks wait).

    `model_name_arg` is the CLI --model_name value (used as the JSON key for HPO configs).
    `model_name` (computed below via model_utils.get_model_name) is the run name used for
    output paths and wandb run names — typically a richer string than the CLI arg.
    """
    utils.clean_cache()
    model_name = model_utils.get_model_name(model_info, dataset_info)
    group_name = dataset_info["groupName"]
    data_version = model_info["dataset_version"]
    columns_setup_dict = model_info["columns_setup_dict"]
    # Temporal split name written by the split script into the info table
    # (self-describing column). Missing on old/random-mode tables -> "" ->
    # get_model_output_path skips the temporal directory level (original layout).
    temporal_version = dataset_info.get("temporal_version", "")

    print_main(f"----- run inputs:\n"
               f"-- model_name {model_name}\n"
               f"-- load model from path {model_info['base_model_path']}\n"
               f"-- group_name {group_name}\n"
               f"-- data_version {data_version}\n"
               f"-- temporal_version {temporal_version!r}\n"
               f"-- columns_setup_dict {columns_setup_dict}\n"
               f"-- ddp {ddp} world_size {get_world_size()}", flush=True)

    output_model_path = model_utils.get_model_output_path(
        base_model=model_info['base_model'],
        version=data_version,
        format_type=model_info["format"],
        query_setup=model_info["query_setup"],
        group_name=f"{group_name}",
        temporal_version=temporal_version
    )
    print_main(f"output_model_path {output_model_path}")

    final_model_dir = os.path.join(output_model_path, "final")
    if any(os.path.exists(os.path.join(final_model_dir, f))
           for f in ("model.safetensors", "pytorch_model.bin")):
        print_main("#### final trained model already exists")
        return None, None, None

    print_main(f"----- load val dataset", flush=True)
    val_dataset = load_parquet_to_dataset(
        dataset_info['val_queries_gold_path'],
        utils.MULTIPLE_NEGATIVE_RANKING_LOSS,
        columns_setup_dict
    )

    print_main(f"----- load train_dataset", flush=True)
    train_dataset = load_parquet_to_dataset(
        dataset_info['train_queries_gold_path'],
        utils.MULTIPLE_NEGATIVE_RANKING_LOSS,
        columns_setup_dict
    )

    hpo_params_trail = load_hpo_configs(hpo_configs_path, model_name_arg, group_name)
    if not hpo_params_trail:
        print_main(
            f"[FATAL] no HPO configs for model={model_name_arg!r} group={group_name!r} "
            f"(resolved size tier is empty). Fill the tier in {hpo_configs_path} before running.",
            flush=True
        )
        return None, None, None

    best_result = None
    best_hyperparams = None

    for trail_num, trail_params in enumerate(hpo_params_trail):
        print_main(f"start trail_num {trail_num}")

        cur_output_model_path = os.path.join(output_model_path, f"hpo_trail_{trail_num}")
        # Trial scoring is on VAL — test is reserved for the final model
        # (selecting trials on test leaks the test set into model selection).
        eval_val_path = os.path.join(cur_output_model_path, "eval", utils.TRAINED, utils.VAL)

        # Skip-if-done check: deterministic filesystem read, all ranks agree.
        if os.path.isdir(cur_output_model_path):
            print_main(f"cur_output_model_path exists {cur_output_model_path}")
            if os.path.isdir(eval_val_path) and any(
                    f.startswith("Information-Retrieval_evaluation_val_results.csv")
                    for f in os.listdir(eval_val_path)):
                print_main(f"found csv file for trail")
                df = pd.read_csv(os.path.join(
                    eval_val_path,
                    "Information-Retrieval_evaluation_val_results.csv"
                ))
                current_result = df["cosine-MRR@10"].tolist()[-1]
                print_main(f"with current_result {current_result}")
                if best_result is None or current_result > best_result:
                    best_result = current_result
                    best_hyperparams = trail_params
                    print_main(f"update best result")
                continue

        # Only rank 0 creates the dir; other ranks wait then proceed.
        if is_main_process():
            utils.create_dir(cur_output_model_path)
        ddp_barrier()

        print_main(f"Training with hyperparameters: {trail_params}\n"
                   f"cur_output_model_path : {cur_output_model_path}")

        # Dev evaluator: only rank 0 builds it (it runs on rank 0 only during eval).
        ir_evaluator_dev = None
        if is_main_process():
            print(f"----- load ir_evaluator_dev", flush=True)
            ir_evaluator_dev = get_ir_evaluator_set(
                dataset_info, utils.VAL, utils.VAL, model_info['setup_column_dict']
            )

        trained_model = train_model(
            model_info=model_info,
            model_name=f"{model_name}_trail_{trail_num}",
            hyperparams=trail_params,
            val_dataset=val_dataset,
            train_dataset=train_dataset,
            ir_evaluator_dev=ir_evaluator_dev,
            output_model_path=cur_output_model_path,
        )

        # Trial scoring on VAL: rank 0 only. Reuses the dev evaluator built above;
        # test is evaluated only for the final model (in main's cross-dataset eval).
        current_result = None
        if is_main_process():
            utils.create_dir(eval_val_path)
            print(f"---- start VAL eval for trial scoring "
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"eval_val_path {eval_val_path}", flush=True)
            eval_val_res = ir_evaluator_dev(trained_model, eval_val_path)
            print(f"---- trial {trail_num} evaluated on {utils.VAL} set, saved to "
                  f"{eval_val_path}:\nwith results:\n{eval_val_res}")
            current_result = eval_val_res['val_cosine_mrr@10']
            print(f"current_result (val) {current_result} {type(current_result)}")

        del ir_evaluator_dev
        gc.collect()

        del trained_model
        gc.collect()
        # Return freed CUDA blocks to the allocator before the next HPO trial
        # allocates (trials vary batch_size; without this, fragmentation from a
        # large-batch trial can OOM the next one).
        torch.cuda.empty_cache()

        # Broadcast current_result so all ranks update best_result in sync.
        current_result = ddp_broadcast_object(current_result, src=0)
        ddp_barrier()

        if current_result is not None and (best_result is None or current_result > best_result):
            best_result = current_result
            best_hyperparams = trail_params
            save_params(trail_params, trail_num, best_hyperparams, best_result,
                        output_model_path, "found new best!", current_score=current_result)
            print_main(
                f"!! New best result found! trail_num {trail_num} {best_result} "
                f"with hyperparameters: {best_hyperparams}"
            )
        else:
            save_params(trail_params, trail_num, best_hyperparams, best_result,
                        output_model_path, "not the best", current_score=current_result)

    print_main("\n=== Done with all HPO trials ===")
    print_main(f"Best result found: {best_result} with hyperparameters: {best_hyperparams}")
    save_params(best_hyperparams, "final_best", best_hyperparams, best_result,
                output_model_path, "final_best", current_score=best_result, is_final=True)

    # Make sure all ranks agree on best_hyperparams before final training.
    best_hyperparams = ddp_broadcast_object(best_hyperparams, src=0)
    ddp_barrier()

    # Final training with the best hyperparameters.
    ir_evaluator_dev = None
    if is_main_process():
        print(f"----- load ir_evaluator_dev to train best", flush=True)
        ir_evaluator_dev = get_ir_evaluator_set(
            dataset_info, utils.VAL, utils.VAL, model_info['setup_column_dict']
        )

    output_model_path_best = utils.concat_directories(output_model_path, "final")
    print_main(f"Training with best hyperparameters: {best_hyperparams}")
    best_trained_model = train_model(
        model_info=model_info,
        model_name=model_name,
        hyperparams=best_hyperparams,
        val_dataset=val_dataset,
        train_dataset=train_dataset,
        ir_evaluator_dev=ir_evaluator_dev,
        output_model_path=output_model_path_best,
    )
    print_main(f"DONE TRAINING MODEL")
    return best_trained_model, model_name, output_model_path_best


def main(args):
    print_main(f"=========================================================\n"
               f"================= start 3_train_models ==================\n"
               f"=========================================================\n"
               f"---- start time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print_main(f"args: {args}", flush=True)

    ds_info = pd.read_parquet(args.dataset_info_table)
    ds_info = ds_info[(ds_info['format'] == args.format) | (ds_info['format'] == "all")]
    model_info = utils.load_json(args.config_model_path)["models"][args.model_name][args.query_setup]
    print_main(f"query setup: {args.query_setup}\nmodel_info: \n{model_info}")
    print_main(f"total of {len(ds_info)} datasets")

    for index, dataset_info in ds_info.iterrows():
        print_main(f" index{index}, groupname {dataset_info['groupName']}")
        if dataset_info['groupName'] != args.group_name:
            continue

        print_main(f"group_name: {args.group_name}\n"
                   f"--------dataset_info: {dataset_info}")
        print_main(model_info["columns_setup_dict"])

        model, model_name_to_eval, output_model_path = \
            train_single_model_on_single_dataset_and_evaluate(
                model_info, dataset_info,
                hpo_configs_path=args.hpo_configs_path,
                model_name_arg=args.model_name,
                ddp=args.ddp,
            )

        if model is None and model_name_to_eval is None and output_model_path is None:
            break

        print_main(f"----- trained model ready at {output_model_path}\n"
                   f"----- run src/eval/4_evaluate_trained_on_test.py for the test-set evaluation",
                   flush=True)

    print_main(f"----- end time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-model-path',
                        default="config/models/configModel_cs2_qt5.json",
                        help='Path to the model JSON configuration file')
    parser.add_argument('--dataset-info-table',
                        default='data/original_split/cs2_qt5/temporal_cutoff_2026_valQ4/all/datasets_info_table/'
                                's2orc_filtered__vcs2_qt5_byQueryTermV5__all_temporal_cutoff_2026_valQ4__datasets_info_table.parquet.gz',
                        help='Path to the datasets_info_table parquet written by step 2 '
                             '(default: the paper run, qt5 + temporal_cutoff_2026_valQ4)')
    parser.add_argument('--hpo_configs_path',
                        default='config/config_hpo_cs2_qt5.json',
                        help='Path to HPO trial configs JSON file')
    parser.add_argument('--model_name',
                        default='all-mpnet-base-v2',
                        help='name of model to train and evaluate')
    parser.add_argument('--query_setup',
                        default='querySentence_prompt',
                        help="setup for training model")
    parser.add_argument('--format',
                        default='transductive',
                        help='transductive or inductive format')
    parser.add_argument('--group_name',
                        help='group name of the sub dataset to train', type=str)
    parser.add_argument('--ddp',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='enable DistributedDataParallel training. Must be launched '
                             'with torchrun and WORLD_SIZE>=2. Exits if validation fails.')
    args = parser.parse_args()

    # Validate DDP environment before doing anything else.
    validate_ddp_environment(use_ddp=args.ddp)

    print_main(f"======= start script check =======\n"
               f"Allocated Memory: {torch.cuda.memory_allocated() / (1024 ** 3)} GB\n"
               f"Cached Memory: {torch.cuda.memory_reserved() / (1024 ** 3)} GB\n"
               f"======= empty_cache =======")
    torch.cuda.empty_cache()

    main(args)
