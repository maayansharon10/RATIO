#!/usr/bin/env python3
"""
2_split_filterGroups_construct_datasets_temporal.py — step 2 of the corpus pipeline

Takes the filtered sentence-pair dataset from step 1, deduplicates pairs, splits
it into train/val/test (temporally by publication date, or randomly grouped by
corpusid), then filters the splits into per-relation-group sub-datasets and
saves everything with leakage checks.

Inputs:
    --config-data-path   JSON config (queryTerm groups, split dirs, paths)
    --date-from/--date-to  YYYYMMDD range embedded in the step-1 output filename
                           (defaults reproduce the paper run: 20150101–20260601)

Outputs (under <dataset_original_split_dir>/<version>/[<split-mode>/]):
    <group>/<split>/s2orc_filtered__...__<group>[_<split-mode>]__<split>.parquet.gz
        train/val/test per queryTerm group ("all" plus each group in the config)
    all/.../datasets_info_table  — paths + sizes for every (format, group, split)
    <group>/queryTermCompareTable — per-queryTerm train/val/test distribution

Usage:
    python src/data/2_split_filterGroups_construct_datasets_temporal.py \
        --config-data-path config/data/configData_cs2_qt5.json \
        --split-mode temporal_cutoff_2026_valQ4 \
        --date-from 20150101 --date-to 20260601
"""

import argparse
import os
from math import isclose
from config.ProjectDataConfig import ProjectDataConfig
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src import utils


# ============================================================================
# Pair-level dedup + overlap helpers (used by date-based split paths)
# ============================================================================
# Pair identity for this project = (querySentence, goldSentence). Matches
# config.extract_columns and createDataset.mutual_columns. Independent of
# queryTerm (a single pair can be discovered via multiple query terms).
PAIR_KEY = ['querySentence', 'goldSentence']


def dedup_pairs(df: pd.DataFrame, key=PAIR_KEY) -> pd.DataFrame:
    """Drop duplicate pairs by (sentence_a, sentence_b). Logs before/after counts."""
    missing = [c for c in key if c not in df.columns]
    assert not missing, f"dedup_pairs: missing required columns {missing}. df has: {df.columns.tolist()}"

    before = len(df)
    df = df.drop_duplicates(subset=key, keep='first').reset_index(drop=True)
    after = len(df)
    print(f"--- dedup_pairs on {key}: {before:,} -> {after:,} (removed {before - after:,})", flush=True)
    return df


def check_no_pair_intersections_in_split(train_df: pd.DataFrame,
                                         val_df: pd.DataFrame,
                                         test_df: pd.DataFrame,
                                         key=PAIR_KEY) -> None:
    """Assert no pair (sentence_a, sentence_b) appears in more than one split."""
    def _pair_set(df):
        return set(map(tuple, df[key].itertuples(index=False, name=None)))

    s_train = _pair_set(train_df)
    s_val = _pair_set(val_df)
    s_test = _pair_set(test_df)

    inter_train_val = s_train & s_val
    inter_train_test = s_train & s_test
    inter_val_test = s_val & s_test

    if inter_train_val or inter_train_test or inter_val_test:
        print(f"PAIR OVERLAP DETECTED:")
        print(f"  train ∩ val  : {len(inter_train_val):,}")
        print(f"  train ∩ test : {len(inter_train_test):,}")
        print(f"  val   ∩ test : {len(inter_val_test):,}")
        assert False, "Pair-level overlap between splits detected (key=%s)" % key
    print(f"--- pair overlap check passed (key={key})", flush=True)


# ============================================================================
# Date-based split helpers (shared between temporal mid-2025 and cutoff-2026)
# ============================================================================
def _resolve_year_and_month(df: pd.DataFrame) -> tuple:
    """
    Return (year_series, month_series) for date-based splitting.

    Requires either:
      - 'year' AND 'publicationdate' (preferred — canonical S2ORC field), OR
      - 'year' AND 'month' (legacy fallback).

    Logs which path was taken and warns about unparseable publicationdate rows.
    """
    assert 'year' in df.columns, (
        f"date-based split requires 'year' column. df has: {df.columns.tolist()}"
    )

    year = pd.to_numeric(df['year'], errors='coerce')

    if 'publicationdate' in df.columns:
        print("--- using 'publicationdate' to derive month", flush=True)
        # Coerce strings/objects/datetimes uniformly to a datetime, then pull month.
        pub_dt = pd.to_datetime(df['publicationdate'], errors='coerce')
        month = pub_dt.dt.month
        n_unparseable = int(pub_dt.isna().sum())
        if n_unparseable:
            print(f"--- WARNING: {n_unparseable:,} rows have unparseable 'publicationdate'", flush=True)
    elif 'month' in df.columns:
        print("--- 'publicationdate' not found, using 'month' column", flush=True)
        month = pd.to_numeric(df['month'], errors='coerce')
    else:
        raise AssertionError(
            f"date-based split requires either 'publicationdate' or 'month' column. "
            f"df has: {df.columns.tolist()}"
        )

    return year, month


def _apply_date_masks(df: pd.DataFrame,
                      train_mask: pd.Series,
                      val_mask: pd.Series,
                      test_mask: pd.Series,
                      label_train: str,
                      label_val: str,
                      label_test: str) -> tuple:
    """Apply three mutually-exclusive date-based masks, log, and return (train, val, test)."""
    overlap = (train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)) > 1
    assert not overlap.any(), f"date masks overlap on {int(overlap.sum())} rows (bug in mask logic)"

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    dropped = len(df) - (len(train_df) + len(val_df) + len(test_df))
    if dropped > 0:
        # Most common cause: rows with null/unparseable publicationdate that
        # fall in the val/test year range (year-only check can't bucket them).
        unbucketed = df[~(train_mask | val_mask | test_mask)]
        year_counts = unbucketed['year'].value_counts(dropna=False).head(10).to_dict()
        print(f"--- WARNING: {dropped:,} rows dropped (no bucket). year distribution of dropped: {year_counts}",
              flush=True)

    total = len(train_df) + len(val_df) + len(test_df)
    print(f"--- date split result:", flush=True)
    print(f"      train ({label_train}): {len(train_df):>12,}  ({100*len(train_df)/total:5.2f}%)",
          flush=True)
    print(f"      val   ({label_val}): {len(val_df):>12,}  ({100*len(val_df)/total:5.2f}%)",
          flush=True)
    print(f"      test  ({label_test}): {len(test_df):>12,}  ({100*len(test_df)/total:5.2f}%)",
          flush=True)

    return train_df, val_df, test_df


# ============================================================================
# Temporal split (mid-2025 cutoff)
# ============================================================================
def temporal_midyear2025_split(df: pd.DataFrame) -> tuple:
    """
    Split by publication date:
      train: year <= 2024
      val  : year == 2025 AND month <= 6
      test : (year == 2025 AND month >= 7) OR year >= 2026

    Requires either:
      - 'year' AND 'publicationdate' columns (preferred, since publicationdate
         is the canonical source — matches the raw S2ORC schema), OR
      - 'year' AND 'month' columns (legacy — used if publicationdate isn't there).

    Rows that fall outside all three buckets (e.g. year=2025 with unparseable
    publicationdate / null month) are logged and dropped.

    Returns: (train_df, val_df, test_df)
    """
    print(f"--- temporal_midyear2025_split on {len(df):,} rows", flush=True)
    year, month = _resolve_year_and_month(df)

    train_mask = year <= 2024
    val_mask = (year == 2025) & (month <= 6)
    test_mask = ((year == 2025) & (month >= 7)) | (year >= 2026)

    return _apply_date_masks(df, train_mask, val_mask, test_mask,
                             label_train="year<=2024",
                             label_val="2025 H1",
                             label_test="2025 H2 + 2026")


# ============================================================================
# Cutoff 2026 split (train extends through 2025-H1)
# ============================================================================
def temporal_cutoff_2026_split(df: pd.DataFrame) -> tuple:
    """
    Split by publication date — pushes the train horizon out to mid-2025:
      train: year <= 2024  OR  (year == 2025 AND month <= 6)
      val  : year == 2025 AND month >= 7
      test : year >= 2026

    Same input requirements as temporal_midyear2025_split: 'year' plus either
    'publicationdate' (preferred) or 'month'.

    Returns: (train_df, val_df, test_df)
    """
    print(f"--- temporal_cutoff_2026_split on {len(df):,} rows", flush=True)
    year, month = _resolve_year_and_month(df)

    train_mask = (year <= 2024) | ((year == 2025) & (month <= 6))
    val_mask = (year == 2025) & (month >= 7)
    test_mask = year >= 2026

    return _apply_date_masks(df, train_mask, val_mask, test_mask,
                             label_train="year<=2024 + 2025 H1",
                             label_val="2025 H2",
                             label_test="2026+")


# ============================================================================
# Cutoff 2026 split, validation = 2025 Q4
# ============================================================================
def temporal_cutoff_2026_valQ4_split(df: pd.DataFrame) -> tuple:
    """
    Split by publication date — train horizon out to 2025 Q3, val is 2025 Q4:
      train: year <= 2024  OR  (year == 2025 AND month <= 9)
      val  : year == 2025 AND month >= 10
      test : year >= 2026

    Same input requirements as temporal_midyear2025_split: 'year' plus either
    'publicationdate' (preferred) or 'month'.

    Returns: (train_df, val_df, test_df)
    """
    print(f"--- temporal_cutoff_2026_valQ4_split on {len(df):,} rows", flush=True)
    year, month = _resolve_year_and_month(df)

    train_mask = (year <= 2024) | ((year == 2025) & (month <= 9))
    val_mask = (year == 2025) & (month >= 10)
    test_mask = year >= 2026

    return _apply_date_masks(df, train_mask, val_mask, test_mask,
                             label_train="year<=2024 + 2025 Q1-3",
                             label_val="2025 Q4",
                             label_test="2026+")


# ============================================================================
# Original helpers (unchanged)
# ============================================================================
def get_filtered_raw_data(config, date_from: str = None, date_to: str = None):
    """
    Loads the filtered S2ORC parquet for this config version.
    `date_from`/`date_to` (YYYYMMDD) are passed to the path resolver to match
    files produced by 1_filter_sentences_by_queryTerms.py, whose output
    filenames embed the publication-date range of the data they processed.
    """
    print("------------ loading data", flush=True)
    input_path = config.get_preprocess_filtered_dataset_path(date_from=date_from, date_to=date_to)
    print(f"input_path: {input_path}")
    mutual_columns = config.get_createDataset_mutual_columns()
    df = load_and_filter_columns_raw_dataset(input_path, mutual_columns)
    return df, input_path


def save_df_split_as_parquet_gz(df: pd.DataFrame, config: ProjectDataConfig, dataset_name: str, split_type: str,
                                include_index=False, subdir: str = "", filename_name: str = None) -> str:
    """
    Save one split to parquet.gz at:
        <version>/[<subdir>/]<dataset_name>/[<split_type>/]<file>

    `subdir` / `filename_name` are forwarded to config.get_dataset_path:
      - subdir: extra directory level for the temporal split name
                (e.g. "temporal_midyear2025"). "" -> original layout.
      - filename_name: token used in the FILENAME only (defaults to
                dataset_name). Lets the directory use a clean group name while
                the filename keeps the suffixed token.
    """
    split_path = config.get_dataset_path(dataset_name, split_type,
                                         subdir=subdir, filename_name=filename_name)
    print(f"save {split_type} to: {split_path}", flush=True)
    df.to_parquet(split_path, compression='gzip', index=include_index)
    return split_path


def check_no_corpusid_intersections_in_split(test_df, train_df, val_df):
    form_train = set(train_df['corpusid'].tolist())
    form_val = set(val_df['corpusid'].tolist())
    form_test = set(test_df['corpusid'].tolist())
    inter_train_test = form_train.intersection(form_test)
    inter_train_val = form_train.intersection(form_val)
    inter_val_test = form_val.intersection(form_test)
    # if there's any intersection between the sets, print the intersection
    if inter_train_test or inter_train_val or inter_val_test:
        print("There's an intersection between the sets")
        print("Intersection between train and test:", inter_train_test)
        print("Intersection between train and val:", inter_train_val)
        print("Intersection between val and test:", inter_val_test)
        assert False, "There's an intersection between the corpusids in the sets"


def create_queryTerm_distribution_comp_table(train_df, val_df, test_df, config, dataset_name,
                                             subdir: str = "", filename_name: str = None):
    # PERFORMANCE: only need raw_queryTerm + source label to build the pivot.
    # Slicing the column out FIRST (before .assign and concat) avoids copying
    # ~17 wide string columns (querySentence, goldSentence, prompts, ...) for
    # every row of every split. On Solving_problems_improvements_mitigation
    # (~15M rows), the previous version allocated ~25-30 GB of intermediate
    # data just to produce a ~50-row output table.
    train_qt = pd.DataFrame({'raw_queryTerm': train_df['raw_queryTerm'].values, 'source': 'train'})
    val_qt = pd.DataFrame({'raw_queryTerm': val_df['raw_queryTerm'].values, 'source': 'val'})
    test_qt = pd.DataFrame({'raw_queryTerm': test_df['raw_queryTerm'].values, 'source': 'test'})

    combined = pd.concat([train_qt, val_qt, test_qt], ignore_index=True)

    if len(combined) == 0:
        print(f"--- skipping queryTermCompareTable for '{dataset_name}': no rows in any split", flush=True)
        return

    freq = combined.groupby(['raw_queryTerm', 'source'], observed=True).size().reset_index(name='count')
    freq_pivot = freq.pivot(index='raw_queryTerm', columns='source', values='count').fillna(0)

    # Ensure all three source columns exist BEFORE computing totals/proportions.
    # `pivot` only creates columns for source values that actually appear in
    # the data, so an empty val_df (or train/test) leaves us missing a column.
    for col in ['train', 'val', 'test']:
        if col not in freq_pivot.columns:
            freq_pivot[col] = 0

    freq_pivot['total'] = freq_pivot[['train', 'val', 'test']].sum(axis=1)
    # Avoid divide-by-zero on rows where total == 0 (shouldn't happen by
    # construction since the row exists only if at least one split had it,
    # but be defensive).
    safe_total = freq_pivot['total'].replace(0, pd.NA)
    freq_pivot['train_prop'] = (freq_pivot['train'] / safe_total).fillna(0)
    freq_pivot['val_prop'] = (freq_pivot['val'] / safe_total).fillna(0)
    freq_pivot['test_prop'] = (freq_pivot['test'] / safe_total).fillna(0)

    final_table = freq_pivot[
        ['train', 'val', 'test', 'total', 'train_prop', 'val_prop', 'test_prop']
    ]

    save_df_split_as_parquet_gz(final_table, config, dataset_name, 'queryTermCompareTable', True,
                                subdir=subdir, filename_name=filename_name)


def sanity_check_split(df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       msg: str,
                       check_corpusid: bool = True) -> None:
    """
    Sanity check that splits sum to total and (optionally) have no corpusid overlap.

    check_corpusid=False is used under date-based splits, where a corpusid can in
    principle appear in only one bucket (single publication date per paper),
    so the per-paper check is informational; we use the pair-level check
    (check_no_pair_intersections_in_split) as the authoritative leakage guard.
    """
    print(f"sanity_check_split {msg}", flush=True)
    assert len(df) == (len(train_df) + len(val_df) + len(test_df)), \
        f"split is not absolute. {len(df)} != {len(train_df)} + {len(val_df)} + {len(test_df)}"
    if check_corpusid:
        check_no_corpusid_intersections_in_split(test_df, train_df, val_df)


def load_and_filter_columns_raw_dataset(input_path, mutual_columns):
    print(f"--- dataset path: {input_path}\n"
          f"--- load dataset", flush=True)
    df = pd.read_parquet(input_path)
    print(f"df columns: {df.columns.tolist()}")
    df = df[mutual_columns]
    return df


def regular_group_shuffle_split(df: pd.DataFrame,
                                group_col: str = 'corpusid',
                                train_size: float = 0.8,
                                val_size: float = 0.1,
                                test_size: float = 0.1,
                                random_state: int = 42) -> tuple:
    print(f"train_size = {train_size}, val_size = {val_size}, test_size = {test_size}")
    print(f"{train_size + val_size + test_size}")
    assert isclose(train_size + val_size + test_size, 1.0, rel_tol=1e-9), "The split ratios must sum to 1."

    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    first_split = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_test_idx, val_idx = next(first_split.split(df, groups=df[group_col]))

    train_test = df.iloc[train_test_idx].reset_index(drop=True)
    val = df.iloc[val_idx].reset_index(drop=True)

    second_split = GroupShuffleSplit(n_splits=1, test_size=test_size / (1 - val_size), random_state=random_state)
    train_idx, test_idx = next(second_split.split(train_test, groups=train_test[group_col]))

    train = train_test.iloc[train_idx].reset_index(drop=True)
    test = train_test.iloc[test_idx].reset_index(drop=True)

    return train, val, test


def split_and_sanity_check(df, split_mode: str = 'random'):
    """
    Dispatches to the appropriate split function and runs sanity checks.

    split_mode='random'                    : GroupShuffleSplit by corpusid, with corpusid leakage check.
    split_mode='temporal_midyear2025'      : mid-2025 cutoff (train<=2024 / val 2025-H1 / test 2025-H2+2026).
    split_mode='temporal_cutoff_2026_valH2': 2026 cutoff       (train<=2025-H1 / val 2025-H2 / test 2026+).
    split_mode='temporal_cutoff_2026_valQ4': 2026 cutoff with val as 2025 Q4 (train<=2025-Q3 / val 2025-Q4 / test 2026+).

    Date-based splits also run a pair-level (querySentence, goldSentence) leakage check.
    """
    print(f"--- splitting dataset (split_mode={split_mode})", flush=True)
    if split_mode == 'temporal_midyear2025':
        train_all, val_all, test_all = temporal_midyear2025_split(df)
        sanity_check_split(df, test_all, train_all, val_all,
                           "split raw dataset (temporal_midyear2025)", check_corpusid=True)
        check_no_pair_intersections_in_split(train_all, val_all, test_all)
    elif split_mode == 'temporal_cutoff_2026_valH2':
        train_all, val_all, test_all = temporal_cutoff_2026_split(df)
        sanity_check_split(df, test_all, train_all, val_all,
                           "split raw dataset (temporal_cutoff_2026_valH2)", check_corpusid=True)
        check_no_pair_intersections_in_split(train_all, val_all, test_all)
    elif split_mode == 'temporal_cutoff_2026_valQ4':
        train_all, val_all, test_all = temporal_cutoff_2026_valQ4_split(df)
        sanity_check_split(df, test_all, train_all, val_all,
                           "split raw dataset (temporal_cutoff_2026_valQ4)", check_corpusid=True)
        check_no_pair_intersections_in_split(train_all, val_all, test_all)
    elif split_mode == 'random':
        train_all, val_all, test_all = regular_group_shuffle_split(df)
        sanity_check_split(df, test_all, train_all, val_all,
                           "split raw dataset", check_corpusid=True)
    else:
        raise ValueError(f"unknown split_mode={split_mode!r} "
                         f"(expected 'random', 'temporal_midyear2025', "
                         f"'temporal_cutoff_2026_valH2', or 'temporal_cutoff_2026_valQ4')")
    return train_all, val_all, test_all


def save_split_datasets(train, val, test, config, group, subdir: str = "", filename_name: str = None):
    """
    Save train/val/test for one group.

    `group` is the directory-level group name (clean: "all", "Solving_..."),
    `subdir` is the temporal split-name directory level (or "" for random),
    `filename_name` is the token embedded in the filename (suffixed name, or
    None to default to `group`).
    """
    print("--- save the split datasets", flush=True)
    train_group_path = save_df_split_as_parquet_gz(train, config, group, 'train',
                                                   subdir=subdir, filename_name=filename_name)
    val_group_path = save_df_split_as_parquet_gz(val, config, group, 'val',
                                                 subdir=subdir, filename_name=filename_name)
    test_group_path = save_df_split_as_parquet_gz(test, config, group, 'test',
                                                  subdir=subdir, filename_name=filename_name)
    print(f"train_group_path: {train_group_path}\n"
          f"val_group_path {val_group_path}\n"
          f"test_group_path {test_group_path}")
    return train_group_path, val_group_path, test_group_path


def create_dataset_dict(queries, queries_path, candidates_data, candidates_path):
    return {
        'queries_gold_path': queries_path,
        'queries_len': len(queries),
        "candidates_path": candidates_path,
        "candidates_len": len(candidates_data)
    }


def update_dataset_info_dict(datasets_info_dict, group_name,
                             train_all, train_all_path, train_group, train_group_path,
                             val_all, val_all_path, val_group, val_group_path,
                             test_all, test_all_path, test_group, test_group_path):
    print("update_dataset_info_dict", flush=True)

    group_name_transductive_dict = {
        'train': create_dataset_dict(train_group, train_group_path, train_all, train_all_path),
        'val': create_dataset_dict(val_group, val_group_path, val_all, val_all_path),
        'test': create_dataset_dict(test_group, test_group_path, test_all, test_all_path)
    }
    group_name_inductive_dict = {
        'train': create_dataset_dict(train_group, train_group_path, train_group, train_group_path),
        'val': create_dataset_dict(val_group, val_group_path, val_group, val_group_path),
        'test': create_dataset_dict(test_all, test_all_path, test_all, test_all_path)
    }
    datasets_info_dict[('transductive', group_name)] = group_name_transductive_dict
    datasets_info_dict[('inductive', group_name)] = group_name_inductive_dict
    return datasets_info_dict


def save_dataset_info_dict(config, datasets_info_dict, dataset_version, info_dataset_name='all',
                           subdir: str = "", filename_name: str = None,
                           temporal_version: str = ""):
    print("save_dataset_info_dict")
    info_df = pd.DataFrame.from_dict(datasets_info_dict, orient='index')
    info_df.index.names = ['format', 'groupName']
    for col in ['train', 'val', 'test']:
        info_df = pd.concat([info_df.drop([col], axis=1), info_df[col].apply(pd.Series).add_prefix(f"{col}_")], axis=1)
    info_df = info_df.reset_index()
    info_df['dataset_version'] = dataset_version
    # Self-describing temporal split name, consumed downstream (e.g. by the
    # training script's get_model_output_path to add a directory level).
    # Empty string in random mode -> downstream treats it as "no temporal level"
    # and reproduces the original path layout (backward compatible).
    info_df['temporal_version'] = temporal_version
    save_df_split_as_parquet_gz(info_df, config, info_dataset_name, 'datasets_info_table',
                                subdir=subdir, filename_name=filename_name)


# ============================================================================
# Main
# ============================================================================
def main(config_path: str,
         split_mode: str = 'random',
         temporal_save_name: str = 'temporal_midyear2025',
         date_from: str = None,
         date_to: str = None,
         resume: bool = False) -> None:
    """
    split_mode:
      'random'                    : original GroupShuffleSplit by corpusid.
      'temporal_midyear2025'      : mid-2025 cutoff (train<=2024 / val 2025-H1 / test 2025-H2+2026).
      'temporal_cutoff_2026_valH2': 2026 cutoff       (train<=2025-H1 / val 2025-H2 / test 2026+).
      'temporal_cutoff_2026_valQ4': 2026 cutoff, val 2025 Q4 (train<=2025-Q3 / val 2025-Q4 / test 2026+).

    Path layout:
      random mode:
        <version>/<group>/<split>/...
      date-based modes: the split name is its OWN directory level, and
      the group directory is clean (no suffix). The filename keeps the
      suffixed token (<group>_<temporal_save_name>) so file names match the
      previous convention:
        <version>/<temporal_save_name>/<group>/<split>/
          s2orc_filtered__...__<group>_<temporal_save_name>__<split>.parquet.gz

    temporal_save_name is the directory level (and filename suffix) used
    whenever split_mode != 'random'.
    """
    print(f"------------ START 2_split_filterGroups_construct_datasets_temporal.py "
          f"(split_mode={split_mode}, save_name={temporal_save_name if split_mode != 'random' else 'N/A'}, "
          f"date_from={date_from}, date_to={date_to}, resume={resume}) ------------",
          flush=True)
    utils.print_time()
    print("------------ loading config", flush=True)
    config = ProjectDataConfig(config_path)

    is_date_split = split_mode in ('temporal_midyear2025', 'temporal_cutoff_2026_valH2', 'temporal_cutoff_2026_valQ4')

    # Directory level for the temporal split name. Empty in random mode, so
    # get_dataset_path falls back to the original <version>/<group>/<split>
    # layout (backward compatible).
    subdir = temporal_save_name if is_date_split else ""

    # Filename suffix glued onto the group token (filename ONLY, not dir).
    # Empty in random mode -> filename == group, identical to original.
    name_suffix = f"_{temporal_save_name}" if is_date_split else ""

    def dir_group(base: str) -> str:
        """Directory-level group name (clean, no suffix)."""
        return base

    def file_token(base: str) -> str:
        """Filename token (group + temporal suffix in date modes)."""
        return f"{base}{name_suffix}"

    # Resolve the all-split paths up front. If --resume is set and all three
    # parquets exist, skip the expensive raw-load + dedup + split and load them
    # straight from disk. Otherwise, do the full pipeline.
    train_all_path = config.get_dataset_path(dir_group("all"), 'train',
                                             subdir=subdir, filename_name=file_token("all"))
    val_all_path = config.get_dataset_path(dir_group("all"), 'val',
                                           subdir=subdir, filename_name=file_token("all"))
    test_all_path = config.get_dataset_path(dir_group("all"), 'test',
                                            subdir=subdir, filename_name=file_token("all"))
    all_paths_exist = (os.path.exists(train_all_path)
                       and os.path.exists(val_all_path)
                       and os.path.exists(test_all_path))

    if resume and all_paths_exist:
        # Skip steps 0–1: parquet load, dedup, split, save_split_datasets(all).
        # This is the expensive path (~1h on the 18M-row filtered file).
        print(f"------------ RESUME: loading existing global split from disk", flush=True)
        print(f"      train: {train_all_path}", flush=True)
        print(f"      val  : {val_all_path}", flush=True)
        print(f"      test : {test_all_path}", flush=True)
        train_all = pd.read_parquet(train_all_path)
        val_all = pd.read_parquet(val_all_path)
        test_all = pd.read_parquet(test_all_path)
        print(f"      loaded: train={len(train_all):,}  val={len(val_all):,}  test={len(test_all):,}",
              flush=True)
        # Defensive re-check: pair-level overlap on the loaded global split.
        # Cheap, and guards against hand-edited parquets between runs.
        if is_date_split:
            check_no_pair_intersections_in_split(train_all, val_all, test_all, key=PAIR_KEY)
    else:
        if resume and not all_paths_exist:
            # User asked for resume but the inputs aren't all there. Fall through
            # to the full pipeline rather than failing — but tell them why.
            print(f"------------ RESUME requested but not all all-split parquets exist:",
                  flush=True)
            print(f"      train exists: {os.path.exists(train_all_path)} ({train_all_path})", flush=True)
            print(f"      val   exists: {os.path.exists(val_all_path)} ({val_all_path})", flush=True)
            print(f"      test  exists: {os.path.exists(test_all_path)} ({test_all_path})", flush=True)
            print(f"      → falling back to full pipeline (raw load + dedup + split)", flush=True)

        df, input_path = get_filtered_raw_data(config, date_from=date_from, date_to=date_to)
        print(f"input_path : {input_path}")

        # Pre-split global dedup on (querySentence, goldSentence) — only under
        # date-based split modes, to preserve byte-identical behavior of the
        # original random-split path.
        if is_date_split:
            df = dedup_pairs(df, key=PAIR_KEY)

        # 1. Split into train_all, val_all, test_all
        train_all, val_all, test_all = split_and_sanity_check(df, split_mode=split_mode)
        train_all_path, val_all_path, test_all_path = save_split_datasets(
            train_all, val_all, test_all, config, dir_group("all"),
            subdir=subdir, filename_name=file_token("all")
        )

    datasets_info_dict = {}
    datasets_info_dict = update_dataset_info_dict(
        datasets_info_dict, dir_group("all"),
        train_all, train_all_path, train_all, train_all_path,
        val_all, val_all_path, val_all, val_all_path,
        test_all, test_all_path, test_all, test_all_path
    )

    # 2. Per-group filter + save (test stays as test_all for transductive eval)
    queryTerms_groups = config.get_queryTerms_groups()

    for group_name, group_categories_list in queryTerms_groups.items():
        if group_name == 'all':
            continue
        print(f"--- filter and split group: {group_name}", flush=True)
        utils.print_time()

        train_group = train_all[train_all['queryTerm_group'].isin(group_categories_list)]
        val_group = val_all[val_all['queryTerm_group'].isin(group_categories_list)]
        test_group = test_all[test_all['queryTerm_group'].isin(group_categories_list)]

        # Under date-based splits, also assert no pair leakage at the group level.
        # (Filtering a clean global split cannot reintroduce overlap, but this
        # catches any upstream surprise like duplicated rows in queryTerm_group
        # membership.)
        if is_date_split:
            check_no_pair_intersections_in_split(train_group, val_group, test_group)

        train_group_path, val_group_path, test_group_path = save_split_datasets(
            train_group, val_group, test_group, config, dir_group(group_name),
            subdir=subdir, filename_name=file_token(group_name)
        )

        datasets_info_dict = update_dataset_info_dict(
            datasets_info_dict, dir_group(group_name),
            train_all, train_all_path, train_group, train_group_path,
            val_all, val_all_path, val_group, val_group_path,
            test_all, test_all_path, test_group, test_group_path
        )

        create_queryTerm_distribution_comp_table(
            train_group, val_group, test_all, config, dir_group(group_name),
            subdir=subdir, filename_name=file_token(group_name)
        )

    # 3. Save info dict
    #    `subdir` is the temporal save-name in date modes, "" in random mode —
    #    exactly the value we want stored in the temporal_version column.
    save_dataset_info_dict(config, datasets_info_dict, config.get_version(),
                           info_dataset_name=dir_group("all"),
                           subdir=subdir, filename_name=file_token("all"),
                           temporal_version=subdir)

    print("------------ DONE 2_split_filterGroups_construct_datasets_temporal.py ------------", flush=True)
    utils.print_time()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-data-path', help='Path to the data JSON configuration file')

    # The split-mode name becomes the directory level for the saved datasets
    # (via --temporal-save-name).
    parser.add_argument('--split-mode',
                        choices=['random', 'temporal_midyear2025', 'temporal_cutoff_2026_valH2', 'temporal_cutoff_2026_valQ4'],
                        default='temporal_cutoff_2026_valQ4',
                        help='Split strategy (default: temporal_cutoff_2026_valQ4, the paper run). '
                             'random=GroupShuffleSplit by corpusid; '
                             'temporal_midyear2025=mid-2025 cutoff (train<=2024 / val 2025-H1 / test 2025-H2+2026); '
                             'temporal_cutoff_2026_valH2=2026 cutoff (train<=2025-H1 / val 2025-H2 / test 2026+); '
                             'temporal_cutoff_2026_valQ4=2026 cutoff, val 2025 Q4 '
                             '(train<=2025-Q3 / val 2025-Q4 / test 2026+).')

    parser.add_argument('--temporal-save-name', default=None,
                        help='Directory level (and filename suffix) used when --split-mode != random. '
                             'Defaults to the split-mode name itself: '
                             'temporal_midyear2025 -> "temporal_midyear2025"; '
                             'temporal_cutoff_2026_valH2 -> "temporal_cutoff_2026_valH2"; '
                             'temporal_cutoff_2026_valQ4 -> "temporal_cutoff_2026_valQ4". '
                             'Saved as: <version>/<temporal_save_name>/<group>/<split>/...')
    parser.add_argument('--date-from', default='20150101',
                        help='YYYYMMDD lower bound used to resolve the filtered parquet filename '
                             '(must match the suffix produced by 1_filter_sentences_by_queryTerms.py). '
                             'Default: 20150101')
    parser.add_argument('--date-to', default='20260601',
                        help='YYYYMMDD upper bound used to resolve the filtered parquet filename. '
                             'Default: 20260601 (the paper run)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from a previous run that crashed AFTER the all train/val/test '
                             'parquets were written. Loads them from disk and skips the raw-load + '
                             'dedup + split (the expensive ~1h step). Falls back to the full pipeline '
                             'if any of the three parquets is missing.')
    args = parser.parse_args()

    split_mode = args.split_mode

    # Default the save-name (directory level + filename suffix) per split mode:
    # every date-based split-mode name is its own directory name. ('random'
    # has no temporal dir; the value is unused in that mode.)
    if args.temporal_save_name is not None:
        temporal_save_name = args.temporal_save_name
    elif split_mode == 'random':
        temporal_save_name = 'temporal_midyear2025'  # unused in random mode
    else:
        temporal_save_name = split_mode

    main(args.config_data_path,
         split_mode=split_mode,
         temporal_save_name=temporal_save_name,
         date_from=args.date_from,
         date_to=args.date_to,
         resume=args.resume)