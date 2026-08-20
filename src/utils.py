"""Shared helpers for the data and training pipeline."""

import gc
import json
import os
from datetime import datetime

import pandas as pd

BASELINE = "baseline"
TRAINED = "trained"
TEST = "test"
VAL = "val"
MULTIPLE_NEGATIVE_RANKING_LOSS = 'MultipleNegativesRankingLoss'
K_VALUES = [1, 3, 5, 10, 20, 100]
FRONT_COLS_NEW = ['model_name', 'model_status', 'split', 'queryTerm_group_train', 'base_model_path',
                  'output_model_path', 'queryTerm_group_eval', 'test_queries_gold_path', 'test_queries_gold_len',
                  'test_candidates_path', 'test_candidates_len', 'query_prompt_name']


def col_as_list(df: pd.DataFrame, col: str, to_lower=True) -> list:
    if to_lower:
        return df[col].str.lower().tolist()
    return df[col].tolist()


def filter_df_by_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.dropna(subset=[col])
    df = df.reset_index(drop=True)
    return df


def create_dir(dir):
    os.makedirs(dir, exist_ok=True)


def create_path_from_dir_filename(dir: str, filename: str) -> str:
    create_dir(dir)
    return os.path.join(dir, filename)


def get_time() -> str:
    return f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"


def print_time() -> None:
    print(f"-- time {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


def concat_directories(dir1: str, dir2: str) -> str:
    dir = os.path.join(dir1, dir2)
    create_dir(dir)
    return dir


def load_json(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"{path} file won't load, e : {e}")
        raise e


def clean_cache():
    """Free CPU + GPU memory. torch is imported lazily so the data-pipeline
    stages (0-2) don't require it."""
    gc.collect()
    import torch
    torch.cuda.empty_cache()
