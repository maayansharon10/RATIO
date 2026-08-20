#!/usr/bin/env python3
"""
4_evaluate_trained_on_test.py — step 4: test-set evaluation of a trained final model.

Refactored out of 3_train_models_hpo_ddp.py's main with identical outputs: loads the
final model trained by step 3 for one (base model, group, setup), evaluates it on the
reported groups' test sets (the three relation groups plus `all` — cross-group
transfer), and writes the same CSVs the in-training evaluation used to write.

Inputs:
    Same selection args as step 3 (--model_name / --group_name / --query_setup /
    --format), the model config, and step 2's datasets_info_table.

Outputs (under <model output path>/final/eval/trained/test/):
    Information-Retrieval_evaluation_test_<group>_results_with_headers.csv  per group
    all_model_evals_on_test_<timestamp>.csv                                 combined

Usage:
    python src/eval/4_evaluate_trained_on_test.py \
        --model_name all-mpnet-base-v2 --group_name more_generally \
        --query_setup querySentence_prompt --format transductive
"""

import argparse
import gc
import os
from datetime import datetime

import pandas as pd
import torch

import src.model_utils as model_utils
import src.utils as utils
from src.eval.eval_utils import auto_eval_add_model_info_and_save, get_ir_evaluator_set

# Cross-dataset eval scope: the 3 relation groups + the full `all` corpus.
# Other groups (allButContrast, similarity/specificity subtypes, ...) are skipped:
# their test files lack the derived query columns some setups need and they are
# not reported.
EVAL_GROUPS = {"all", "Solving_problems_improvements_mitigation",
               "more_generally", "more_specific"}


def main(args):
    print(f"=========================================================\n"
          f"=========== start 4_evaluate_trained_on_test ============\n"
          f"=========================================================\n"
          f"---- start time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
          f"args: {args}", flush=True)

    ds_info = pd.read_parquet(args.dataset_info_table)
    ds_info = ds_info[(ds_info['format'] == args.format) | (ds_info['format'] == "all")]
    model_info = utils.load_json(args.config_model_path)["models"][args.model_name][args.query_setup]
    print(f"query setup: {args.query_setup}\nmodel_info: \n{model_info}", flush=True)

    for index, dataset_info in ds_info.iterrows():
        if dataset_info['groupName'] != args.group_name:
            continue

        model_name_to_eval = model_utils.get_model_name(model_info, dataset_info)
        temporal_version = dataset_info.get("temporal_version", "")
        output_model_path = model_utils.get_model_output_path(
            base_model=model_info['base_model'],
            version=model_info["dataset_version"],
            format_type=model_info["format"],
            query_setup=model_info["query_setup"],
            group_name=f"{args.group_name}",
            temporal_version=temporal_version
        )
        # Step 3 saves the final model here; provenance columns record this path,
        # matching the previous in-training evaluation exactly.
        output_model_path_final = utils.concat_directories(output_model_path, "final")
        print(f"loading trained model from {output_model_path_final}", flush=True)
        model = model_utils.load_model_sentence_transformers(
            model_name=model_name_to_eval,
            model_path=output_model_path_final,
            similarity_fn_name="cosine",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            is_eval=True
        )

        model_eval_path_on_dataset_lst = []
        eval_test_path = model_utils.get_model_output_sub_path(
            output_model_path_final, "eval", utils.TRAINED, utils.TEST
        )
        # Prompt provenance recorded in the CSV alongside the scores.
        query_prompt = model_utils.get_query_prompt(model_info)
        query_prompt_name = model_utils.get_query_prompt_name(model_info)
        for index_eval, dataset_info_eval in ds_info.iterrows():
            group_name_eval = dataset_info_eval['groupName']
            if group_name_eval not in EVAL_GROUPS:
                print(f"-- skip cross-dataset eval on {group_name_eval} (not in EVAL_GROUPS)", flush=True)
                continue
            print(f"---------------------------------------\n"
                  f"--- start eval model {model_name_to_eval} on dataset "
                  f"{dataset_info_eval['groupName']}\n"
                  f"---- start time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"index {index_eval} dataset_info : {dataset_info_eval}\n")
            print(f"-- loading test_queries_gold_path "
                  f"{dataset_info_eval['test_queries_gold_path']}\n"
                  f"-- loading test_candidates_path "
                  f"{dataset_info_eval['test_candidates_path']}\n", flush=True)
            test_queries, test_relevant_docs, test_corpus = \
                model_utils.get_queries_relevant_docs_and_corpus(
                    dataset_info_eval['test_queries_gold_path'],
                    model_info['setup_column_dict'],
                    dataset_info_eval['test_candidates_path']
                )

            test_name = f"{utils.TEST}_{group_name_eval}"
            torch.cuda.empty_cache()

            ir_evaluator_test = get_ir_evaluator_set(
                dataset_info_eval, utils.TEST, test_name,
                model_info['setup_column_dict'],
                preloaded=(test_queries, test_relevant_docs, test_corpus)
            )

            print(f"---- start eval with evaluator time "
                  f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                  f"eval_test_path {eval_test_path}", flush=True)
            eval_test_res = ir_evaluator_test(model, eval_test_path)
            print(f"---- auto evaluate model on {utils.TEST} set saved to "
                  f"{eval_test_path}:\nwith results:\n{eval_test_res}")

            ret_rank_res_path = auto_eval_add_model_info_and_save(
                eval_test_path, model_name_to_eval,
                model_info["base_model_path"],
                utils.TRAINED,
                output_model_path_final, utils.TEST,
                dataset_info_eval['test_queries_gold_path'],
                len(test_queries),
                dataset_info_eval['test_candidates_path'],
                len(test_corpus),
                test_name, args.group_name, group_name_eval,
                model_info["query_setup"],
                "in_batch",
                query_prompt, query_prompt_name
            )
            model_eval_path_on_dataset_lst.append(ret_rank_res_path)
            del test_corpus, test_queries, test_relevant_docs
            gc.collect()

        print(f"---------------------------------------")
        all_res_model = pd.concat([pd.read_csv(path) for path in model_eval_path_on_dataset_lst])
        print(f"all_res_model on model {model_name_to_eval} evaluated. "
              f"number of evals:\n{len(all_res_model)}", flush=True)
        # FRONT_COLS_NEW first, remaining columns sorted. Defensive: only front-order
        # columns that actually exist, and append EVERY other column (so new ones like
        # negatives_type / query_prompt are never dropped even if the FRONT_COLS_*
        # constants predate them).
        cols = list(all_res_model.columns)
        front = [c for c in utils.FRONT_COLS_NEW if c in cols]
        back = sorted(c for c in cols if c not in front)
        all_res_model = all_res_model[front + back]
        all_res_model_path = os.path.join(
            eval_test_path,
            f"all_model_evals_on_test_{datetime.now().strftime('%Y-%m-%d__%H:%M')}.csv"
        )
        all_res_model.to_csv(all_res_model_path, index=False)
        print(f"----- all_res_model saved to {all_res_model_path}. "
              f"end time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    print(f"----- end time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


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
    parser.add_argument('--model_name',
                        default='all-mpnet-base-v2',
                        help='name of the trained model to evaluate (key in the model config)')
    parser.add_argument('--query_setup',
                        default='querySentence_prompt',
                        help='setup the model was trained with')
    parser.add_argument('--format',
                        default='transductive',
                        help='transductive or inductive format')
    parser.add_argument('--group_name',
                        help='group name the model was trained on', type=str)
    args = parser.parse_args()
    main(args)
