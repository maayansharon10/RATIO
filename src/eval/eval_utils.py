"""Shared IR-evaluation helpers used by training (step 3) and test evaluation (step 4)."""

import gc
import os

import pandas as pd
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator

import src.model_utils as model_utils
import src.utils as utils


def get_ir_evaluator_set(dataset_info, dataset_split, eval_name, setup_column_dict, batch_size=16,
                         main_score_fn="cosine", query_prompt=None, preloaded=None):
    """Build an InformationRetrievalEvaluator. Called on rank 0 only when DDP is active.

    query_prompt: optional instruction prefix prepended to every query at encode time.
    preloaded: optional (queries, relevant_docs, corpus) tuple to reuse data the caller
    already loaded, instead of reading the parquet files a second time."""
    if preloaded is not None:
        queries, relevant_docs, corpus = preloaded
    else:
        queries, relevant_docs, corpus = model_utils.get_queries_relevant_docs_and_corpus(
            dataset_info[f'{dataset_split}_queries_gold_path'],
            setup_column_dict,
            dataset_info[f'{dataset_split}_candidates_path']
        )
    print(f"----- create evaluator for {eval_name} (query_prompt={'set' if query_prompt else 'none'})", flush=True)
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=False,
        name=eval_name,
        batch_size=batch_size,
        write_csv=True,
        main_score_function=main_score_fn,
        query_prompt=query_prompt,
        corpus_chunk_size=2000,
        mrr_at_k=utils.K_VALUES,
        ndcg_at_k=utils.K_VALUES,
        accuracy_at_k=utils.K_VALUES,
        precision_recall_at_k=utils.K_VALUES,
    )
    del queries, corpus, relevant_docs
    gc.collect()
    return ir_evaluator


def auto_eval_add_model_info_and_save(eval_test_path, model_name, base_model_path, model_status,
                                      output_model_path,
                                      split,
                                      test_queries_gold_path, test_queries_gold_len,
                                      test_candidates_path, test_candidates_len,
                                      test_name, queryTerm_group_train, group_name_eval, query_setup,
                                      negatives_type, query_prompt=None, query_prompt_name=None):
    """Enrich and save an IR eval CSV with model/run provenance columns.

    negatives_type: how the trained model was optimized (this pipeline: "in_batch").
    query_prompt / query_prompt_name: the instruction prefix used at eval time (or None),
    recorded so eval provenance is explicit."""
    auto_res_path = os.path.join(eval_test_path, f"Information-Retrieval_evaluation_{test_name}_results.csv")
    print(f"loading auto_res_path {auto_res_path}", flush=True)
    auto_eval_test_df = pd.read_csv(auto_res_path).drop_duplicates()
    auto_eval_test_df['split'] = split
    auto_eval_test_df['model_status'] = model_status
    auto_eval_test_df['base_model_path'] = base_model_path
    auto_eval_test_df['model_name'] = model_name
    auto_eval_test_df['output_model_path'] = output_model_path
    auto_eval_test_df['test_queries_gold_path'] = test_queries_gold_path
    auto_eval_test_df['test_queries_gold_len'] = test_queries_gold_len
    auto_eval_test_df['test_candidates_path'] = test_candidates_path
    auto_eval_test_df['test_candidates_len'] = test_candidates_len
    auto_eval_test_df['queryTerm_group_eval'] = group_name_eval
    auto_eval_test_df['queryTerm_group_train'] = queryTerm_group_train
    auto_eval_test_df['query_setup'] = query_setup
    auto_eval_test_df['negatives_type'] = negatives_type
    auto_eval_test_df['query_prompt'] = query_prompt
    auto_eval_test_df['query_prompt_name'] = query_prompt_name
    ret_rank_res_path = utils.create_path_from_dir_filename(
        eval_test_path,
        f"Information-Retrieval_evaluation_{test_name}_results_with_headers.csv"
    )
    auto_eval_test_df.to_csv(ret_rank_res_path, index=False)
    return ret_rank_res_path
