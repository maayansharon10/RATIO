#!/usr/bin/env python3
"""
1_filter_sentences_by_queryTerms.py — step 1 of the corpus pipeline

Filters the sentence corpus produced by 00_process_shard.py (step 0, or any
corpus with the same structure) down to sentences
that start with a queryTerm (a connective phrase such as "in contrast," or
"more generally," listed in the queryTerms CSV), then splits each match into
querySentence / goldSentence and builds the prompt-variant columns used for
training and evaluation. Sentences that match no known queryTerm are mined for
potential new queryTerm candidates.

Inputs:
    --config-path   JSON config (column names, queryTerms CSV path, output path)
    --shards-dir    Directory with the sentence corpus from 00_process_shard.py

Outputs:
    {config filtered-output path}_{minpub}_{maxpub}.parquet.gz
        Filtered rows with queryTerm, queryTerm_group, querySentence,
        goldSentence, and prompt-variant columns.
    {shards_dir}/potential_queryTerms_distribution.csv / .parquet
        Corpus-wide frequency of unmatched leading phrases.
    {shards_dir}/potential_queryTerms_for_llm.parquet
        Frequency-thresholded candidates + sample sentences, input for the
        downstream LLM queryTerm classifier.

Usage:
    python src/data/1_filter_sentences_by_queryTerms.py \
        --config-path config/data/configData_cs2_qt5.json \
        --shards-dir ./data_s2orc/output --num_workers 60
"""

import argparse
import glob
import json
import pandas as pd
import re
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from config.ProjectDataConfig import ProjectDataConfig
from src import utils
from src.utils import col_as_list, filter_df_by_col
import numpy as np

# pyarrow is used for row-group enumeration in single-file mode. Imported at module
# top so the ProcessPoolExecutor forks inherit it and don't have to re-import per worker.
import pyarrow.parquet as pq


def clean_citation_rows(df, column_name, verbose=False):
    """
    Remove rows where column contains citations or quotes or very short sentences.
    Returns a NEW DataFrame without modifying the original.

    Rules:
    1. Remove if contains [...] with any numbers or "et al." inside
       - [24] ✓ removed
       - [59, 60] ✓ removed
       - [A.2] ✓ removed
       - [eisenhardt and martin 2000] ✓ removed
       - [jedec committee jc-42.3 2012] ✓ removed
       - [some text] ✗ kept (no numbers)
    2. Remove if contains (et al.)
    3. Remove if contains (name, 4-digit-year) pattern
    4. Remove if the literal text "[]" appears
    5. Remove if value is NaN, empty, whitespace, or a single word
    6. Remove if shorter than 35 characters

    at the end if we have duplicate lines, remove them
    """
    initial_count = len(df)
    print(f"\n{'=' * 50}")
    print(f"Cleaning column: '{column_name}'")
    print(f"Initial row count: {initial_count}")
    print(f"{'=' * 50}")

    # Create a completely independent copy
    df_clean = df.copy(deep=True)

    # Pattern 1: Square brackets containing any numbers or "et al."
    pattern1 = r'\[(?:[^\]]*\d[^\]]*|[^\]]*et al\.[^\]]*)\]'
    mask1 = df_clean[column_name].str.contains(pattern1, regex=True, na=False)

    removed1 = mask1.sum()
    df_clean = df_clean[~mask1].copy()
    print(f"Rows removed with [citations containing numbers or et al.]: {removed1}")
    print(f"Remaining rows: {len(df_clean)}")

    # Pattern 2: Round brackets with "et al."
    pattern2 = r'\(.*et al\..*\)'
    mask2 = df_clean[column_name].str.contains(pattern2, regex=True, na=False)

    removed2 = mask2.sum()
    df_clean = df_clean[~mask2].copy()
    print(f"Rows removed with (et al.): {removed2}")
    print(f"Remaining rows: {len(df_clean)}")

    # Pattern 3: Round brackets with name and 4-digit year
    pattern3 = r'\([^)]*[a-zA-Z][^)]*[,\s]\s*\d{4}\s*\)'
    mask3 = df_clean[column_name].str.contains(pattern3, regex=True, na=False)

    removed3 = mask3.sum()
    df_clean = df_clean[~mask3].copy()
    print(f"Rows removed with (name, year): {removed3}")
    print(f"Remaining rows: {len(df_clean)}")

    pattern4 = r'\[\]'
    mask4 = df_clean[column_name].str.contains(pattern4, regex=True, na=False)

    removed4 = mask4.sum()
    df_clean = df_clean[~mask4].copy()
    print(f"Rows removed with literal []: {removed4}")
    print(f"Remaining rows: {len(df_clean)}")

    # 5th rule
    # Detect single-word strings: letters/digits/underscores only
    single_word_pattern = r'^\s*\w+\s*$'
    mask5 = (
            df_clean[column_name].isna() |
            (df_clean[column_name].str.strip() == "") |
            df_clean[column_name].str.match(single_word_pattern, na=False)
    )

    removed5 = mask5.sum()
    df_clean = df_clean[~mask5].copy()

    df_clean = df_clean[~(df_clean[column_name].str.len() < 35)]

    # Exclude list/array columns from dedup (they're unhashable)
    hashable_cols = []
    for col in df_clean.columns:
        sample = df_clean[col].dropna().iloc[0] if len(df_clean[col].dropna()) > 0 else None
        if not isinstance(sample, (list, np.ndarray)):
            hashable_cols.append(col)

    df_clean = df_clean.drop_duplicates(subset=hashable_cols, keep="first")

    #summary
    total_removed = initial_count - len(df_clean)
    print(f"\n{'=' * 50}")
    print(f"SUMMARY for '{column_name}':")
    print(f"Total rows removed: {total_removed}")
    print(f"Final row count: {len(df_clean)}")
    print(f"{'=' * 50}\n")

    return df_clean

def load_regex_list(queryTerms: list) -> list:
    return [(rf'\b{word}\b(?:\s+\w+){{0,3}},') for word in queryTerms]


def load_queryTerms_lists(queryTerms_path):
    """
    Load queryTerms from a single unified CSV. The CSV has a 'group' column that
    assigns each term to one of 7 groups: the 6 relation-type groups plus 'blacklist'.

    Returns:
        queryTerms_list:     non-blacklist terms, lowercased — used for prefix matching
        queryTerms_group:    group label per term in queryTerms_list (same order)
        special_queryTerms:  non-blacklist terms flagged special_word=True
        blacklist:           blacklist-group terms, lowercased — NOT used for prefix
                             matching (sentences starting with these stay unmatched),
                             but included in the known-set subtraction during
                             potential-queryTerm extraction so they don't surface as
                             "unknown" candidates.
    """
    df = pd.read_csv(queryTerms_path)

    # Split by group. Comparison is case-insensitive to be lenient with CSV authoring.
    is_blacklist = df['group'].astype(str).str.lower() == 'blacklist'
    df_active = df[~is_blacklist]
    df_blacklist = df[is_blacklist]

    queryTerms_list = col_as_list(df_active, "word", to_lower=True)
    queryTerms_group = col_as_list(df_active, "group")
    special_queryTerms_list = col_as_list(
        df_active[df_active['special_word'] == True], "word", to_lower=True
    )
    blacklist = col_as_list(df_blacklist, "word", to_lower=True)

    return queryTerms_list, queryTerms_group, special_queryTerms_list, blacklist


def remove_leading_numbers_and_dot(text):
    """
    Strip leading numbering like "1. ", "2) ", "12. " from a sentence start.
    Used only in the potential-queryTerm extraction stage (not in the main filter),
    so that rows like "1. more generally, X" can be recognised for their leading
    connective phrase after matching has already failed on the raw text.
    """
    if not isinstance(text, str):
        return text
    # Patterns: "1. ", "12) ", "3 - ", etc. at sentence start.
    return re.sub(r'^\s*\d+\s*[.\)\-:]\s*', '', text)


def find_prefix_fast(sentence: str, queryTerms_set: set, max_term_len: int) -> str:
    """
    Check if the sentence starts with any queryTerm using set lookups:
    ~max_term_len O(1) lookups instead of one startswith per term.
    Returns the longest matching queryTerm or None.
    """
    result = None
    limit = min(len(sentence), max_term_len)
    for length in range(1, limit + 1):
        prefix = sentence[:length]
        if prefix in queryTerms_set:
            result = prefix  # keep going to find longest match
    return result


def process_chunk(df_chunk: pd.DataFrame, text_col: str, queryTerm_col: str,
                  chunk_id: int, queryTerms_set: set, max_term_len: int) -> pd.DataFrame:
    """
    Process a DataFrame chunk: for each sentence, extract its queryTerm prefix (or None).
    """
    try:
        print(f"Processing chunk {chunk_id} with {len(df_chunk)} rows. time {utils.get_time()}", flush=True)
        df_chunk[queryTerm_col] = df_chunk[text_col].apply(
            lambda x: find_prefix_fast(x, queryTerms_set, max_term_len))
        print(f"Finished processing chunk {chunk_id}. time {utils.get_time()}", flush=True)
        return df_chunk
    except Exception as e:
        print(f"Error processing chunk {chunk_id} at time {utils.get_time()}:\n{e}")


def extract_and_filter_by_all_queryTerms_parallel(
        df: pd.DataFrame, queryTerms_list: list, text_col: str, raw_queryTerm_col: str, num_workers: int = 4
) -> tuple:
    """
    Parallelized function to filter a DataFrame based on prefix matching.

    Returns a tuple (matched_df, unmatched_df) where:
      - matched_df: rows whose sentence started with a known queryTerm (raw_queryTerm_col populated).
        This is what the downstream pipeline consumes.
      - unmatched_df: rows with no known queryTerm prefix. Returned so the caller can
        spill them to disk for potential-queryTerm extraction, then drop them from memory.
    """
    # Build fast lookup structures
    queryTerms_set = set(queryTerms_list)
    max_term_len = max(len(t) for t in queryTerms_list) if queryTerms_list else 0
    print(f"Built prefix set: {len(queryTerms_set):,} unique terms, max length {max_term_len}")

    # Split DataFrame into chunks and process sequentially. Parallelism happens at
    # the shard level (ProcessPoolExecutor in main), so no nested pools here.
    chunk_size = len(df) // num_workers + 1
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    results = [process_chunk(chunk, text_col, raw_queryTerm_col, idx,
                             queryTerms_set, max_term_len)
               for idx, chunk in enumerate(chunks)]

    # Concatenate results to maintain the original order
    filtered_df = pd.concat(results, ignore_index=True)

    # Split into matched vs. unmatched instead of dropping unmatched.
    # Matched rows feed the pipeline; unmatched rows are returned so the caller
    # can spill them to disk for later potential-queryTerm extraction.
    matched_mask = filtered_df[raw_queryTerm_col].notna()
    matched_df = filtered_df[matched_mask].reset_index(drop=True)
    unmatched_df = filtered_df[~matched_mask].reset_index(drop=True)

    return matched_df, unmatched_df


def extract_and_filter_by_special_queryTerms(df: pd.DataFrame, special_queryTerm_list: list, text_col: str,
                                             queryTerm_col: str) -> pd.DataFrame:
    # Mask to identify rows where 'queryTerm_col' is in special_queryTerm_list
    queryTerm_mask = df[queryTerm_col].str.lower().str.contains('|'.join(special_queryTerm_list), na=False)

    df_special_queryTerms = df[queryTerm_mask].copy()
    # Load regex patterns based on special queryTerms
    regex_patterns = load_regex_list(special_queryTerm_list)

    # extract the first special queryTerm that appears in the sentence
    df_special_queryTerms['matched_pattern'] = df_special_queryTerms[text_col].apply(
        lambda x: match_regex_patterns(x, regex_patterns))

    df_special_queryTerms = filter_df_by_col(df_special_queryTerms, 'matched_pattern')

    # Filter out rows where text_col does not contain any special queryTerms
    df_no_special_queryTerms = df[~queryTerm_mask].copy()
    df_no_special_queryTerms['matched_pattern'] = None

    result_df = pd.concat([df_special_queryTerms, df_no_special_queryTerms], ignore_index=True)
    result_df.reset_index(drop=True)
    return result_df


def merge_queryTerms(row):
    # Function to merge 'matched_pattern' and 'raw_queryTerm' columns into one, prioritizing 'matched_pattern'
    if pd.notna(row['matched_pattern']):
        return row['matched_pattern']
    elif pd.notna(row['raw_queryTerm']):
        return row['raw_queryTerm']
    else:
        return None


def get_queryTerms_group(row: pd.Series, queryTerm_group_dict: dict) -> str:
    # Function to merge 'matched_pattern' and 'raw_queryTerm' columns into one, prioritizing 'matched_pattern'
    return queryTerm_group_dict[row['raw_queryTerm']]


# Groups in the queryTerms CSV that mark a term as irrelevant/blacklisted.
# Compared case-insensitively, so both 'common_irrelevant' and 'BLACKLIST' spellings match.
BAD_GROUPS = {"common_irrelevant", "blacklist"}


def build_blacklist_term2group(queryTerms_path) -> dict:
    """
    Map lowercased term -> its original group label, for terms whose group in the
    queryTerms CSV is common_irrelevant/BLACKLIST. Used by
    correct_blacklisted_queryTerm_groups below.
    """
    qt = pd.read_csv(queryTerms_path)
    qt_bad = qt[qt["group"].astype(str).str.lower().isin(BAD_GROUPS)]
    return dict(zip(qt_bad["word"].astype(str).str.lower(), qt_bad["group"]))


def correct_blacklisted_queryTerm_groups(df: pd.DataFrame, term2group: dict) -> pd.DataFrame:
    """
    Double-verify blacklist/common_irrelevant marking before saving.

    WHY: 'queryTerm' is merged from matched_pattern (special-queryTerm regex) and
    raw_queryTerm, but 'queryTerm_group' is looked up by raw_queryTerm only. A
    special-term match can therefore end with a final queryTerm that the CSV lists
    as common_irrelevant/BLACKLIST while queryTerm_group still carries the group of
    the raw prefix match. Any row whose final queryTerm is in the bad list gets its
    group forced to the CSV's group, so downstream filtering on queryTerm_group
    reliably drops these rows.
    """
    if len(df) == 0 or not term2group:
        return df
    qt_lower = df["queryTerm"].astype(str).str.lower()
    already_bad = df["queryTerm_group"].astype(str).str.lower().isin(BAD_GROUPS)
    mask = qt_lower.isin(term2group) & ~already_bad
    if mask.any():
        df.loc[mask, "queryTerm_group"] = qt_lower[mask].map(term2group)
        print(f"[blacklist-verify] corrected queryTerm_group on {mask.sum():,} rows "
              f"(queryTerm listed as common_irrelevant/BLACKLIST in queryTerms CSV)",
              flush=True)
    return df


def filter_sentences_by_queryTerms(df: pd.DataFrame,
                                   config: ProjectDataConfig,
                                   num_workers: int = 4,
                                   unmatched_save_path: str = None) -> pd.DataFrame:
    """
    Filter sentences that start with a queryTerm and match special queryTerms patterns.

    If unmatched_save_path is provided, rows that did not match any known queryTerm
    are spilled to that parquet path and dropped from memory before the matched rows
    continue through the pipeline. The matched-path output is unaffected by whether
    unmatched_save_path is set.
    """

    queryTerms_list, queryTerms_group, special_queryTerms_list, _ = load_queryTerms_lists(
        config.get_queryTerms_list_path())
    raw_queryTerm = "raw_queryTerm"
    single_text = config.get_preprocessing_s2orc_raw_data_single_text_col()
    matched_df, unmatched_df = extract_and_filter_by_all_queryTerms_parallel(
        df, queryTerms_list, single_text, raw_queryTerm, num_workers=num_workers
    )

    # Spill unmatched to disk (if requested) and drop from memory before continuing.
    if unmatched_save_path is not None:
        if len(unmatched_df) > 0:
            # Keep only the columns needed for downstream extraction — single_text (the
            # sentence start we regex) and corpusid (for traceability of sample rows).
            # Full rows can carry embedding/metadata columns that would bloat disk I/O.
            cols_to_keep = [single_text]
            if 'corpusid' in unmatched_df.columns:
                cols_to_keep.append('corpusid')
            unmatched_slim = unmatched_df[cols_to_keep]
            # fastparquet can choke on list columns; reuse the existing serializer.
            unmatched_slim = _serialize_list_columns(unmatched_slim)
            unmatched_slim.to_parquet(
                unmatched_save_path, engine='fastparquet', index=False, compression='gzip'
            )
        else:
            print("No unmatched rows to spill.", flush=True)
    del unmatched_df

    df = matched_df
    result_df = extract_and_filter_by_special_queryTerms(df, special_queryTerms_list, single_text, raw_queryTerm)
    result_df['queryTerm'] = result_df.apply(merge_queryTerms, axis=1)
    # get zip of queryTerm -> queryTerm_group
    # for each queryTerm, get the group of queryTerms that it belongs to
    queryTerms_list = [word.lower() for word in queryTerms_list]
    queryTerm_group_dict = dict(zip(queryTerms_list, queryTerms_group))
    result_df['queryTerm_group'] = result_df.apply(get_queryTerms_group, axis=1, args=((queryTerm_group_dict),))

    # Double-verify blacklist/common_irrelevant marking before the result is saved:
    # special-queryTerm merging can produce a final queryTerm that the CSV blacklists
    # while queryTerm_group was assigned from the raw prefix match. Force those rows
    # to their CSV group. (See correct_blacklisted_queryTerm_groups for details.)
    bad_term2group = build_blacklist_term2group(config.get_queryTerms_list_path())
    result_df = correct_blacklisted_queryTerm_groups(result_df, bad_term2group)

    return result_df


def match_regex_patterns(sentence, regex_patterns):
    sentence_lower = sentence.lower()
    for pattern in regex_patterns:
        match = re.search(pattern, sentence_lower)
        if match:
            return match.group(0)
    return None


def extract_querySentence(row: pd.Series, multi_text: str, single_text: str) -> str:
    multi_text = row[multi_text]
    single_text = row[single_text]
    index = multi_text.find(single_text)
    querySentence = multi_text[:index]
    return " ".join(querySentence.split())


def sep_introductory_clauses(sentence):
    pattern = r'^\b(\w+(?:\s+\w+){0,3})\b,\s*'
    match = re.match(pattern, sentence, re.IGNORECASE)
    if match:
        # Remove the matched expression from the sentence
        sentence_without_expression = sentence[len(match.group(0)):]
        return match.group(1).strip(), sentence_without_expression.strip()
    return None, sentence


def remove_prefix(text):
    # Define the regex pattern for the unwanted prefixes
    pattern = r"""
        ^                       # Start of the string
        (                       # Begin capture group for unwanted prefixes
            (                   # Nested group for Roman numerals or numbers with optional punctuation
                [ivxlcdm]+       # Roman numerals (lowercase letters)
                \.               # Followed by a dot
                |                # OR
                [ivxlcdm]+       # Roman numerals (lowercase letters)
                ,                # Followed by a comma
                |                # OR
                [ivxlcdm]+       # Roman numerals (lowercase letters)
                \s               # Followed by a space
            )
            |                       # OR
            [),.]+                # Stray punctuation at the start (comma, period, etc.)
        )
        \s*                    # Optional whitespace after the prefix
    """
    # Use regex substitution to remove matched patterns
    cleaned_text = re.sub(pattern, '', text, count=1, flags=re.VERBOSE).strip()
    return cleaned_text


def extract_goldSentence(row: pd.Series, single_text: str, queryTerm: str) -> str:
    sentence = row[single_text]
    queryTerms = row[queryTerm]
    queryTerms_pos = sentence.lower().find(queryTerms.lower())
    if queryTerms_pos == -1:
        return None
    return sentence[queryTerms_pos + len(queryTerms):].strip()


def create_querySentence_variations(row: pd.Series):

    queryTerm = row['queryTerm']
    queryTerm_group = row['queryTerm_group']

    querySentence = row['querySentence']
    querySentence_prompt = "query: " + querySentence
    if querySentence.endswith("."):
        querySentence_dot_queryTerm = querySentence_prompt + queryTerm
    else:
        querySentence_dot_queryTerm = querySentence_prompt + ". " + queryTerm

    task_dict = {
        "implicit_similarity": "Retrieve a sentence that presents a related idea or parallel insight that functions as an inspiration or serves as an analogy.",
        "explicit_similarity": "Find a sentence that directly expresses the same idea as this query, using similar key terms and concepts, ensuring direct semantic matching.",
        "explicit_contrast": "Find a sentence that clearly opposes the query's main point or presenting a counter-argument, offering a view that directly contradicts it.",
        "more_specific": "Find a sentence that provides a more detailed or precise example of what this query describes, offering a narrowly focused example of the concept.",
        'more_generally': "Find a sentence that captures the broader concept or higher-level idea that this query represents,  reflecting a more general conceptual category.",
        "solving_problems_improvements_mitigation": "Find a sentence that offers a solution, improvement, or mitigation to the problem described in this query by providing actionable strategies."
    }

    # Creating the new column
    default_inst = "Instruct: Given a web search query, retrieve relevant passages that answer the query."
    querySentence_group_prompt = f"Instruct: {task_dict.get(queryTerm_group, default_inst)}\n{querySentence_prompt}"

    # New variation: fixed generic instruct prompt
    querySentence_group_s2p_query = (
            "Instruct: Given a web search query, retrieve relevant passages that answer the query.\nQuery: "
            + querySentence
    )

    return querySentence_prompt, querySentence_dot_queryTerm, querySentence_group_prompt, querySentence_group_s2p_query


def create_querySentence_variations_search_query(row: pd.Series):
    """
    search_query / search_document prefixed variants (the prompt format expected by
    nomic-style encoders).

    Produces four columns:
      - goldSentence_prompt_search_query        : "search_document: " + goldSentence
      - querySentence_prompt_search_query       : "search_query: " + querySentence
      - querySentence_dot_queryTerm_search_query: querySentence + queryTerm, "search_query: " prefixed
      - querySentence_group_search_query        : "search_query: Instruct: ...\nquery: ..."
    """
    queryTerm = row['queryTerm']
    queryTerm_group = row['queryTerm_group']
    goldSentence = row['goldSentence']
    querySentence = row['querySentence']

    # goldSentence_prompt
    goldSentence_prompt_search_query = "search_document: " + goldSentence

    # querySentence_prompt
    querySentence_prompt_search_query = "search_query: " + querySentence

    # querySentence_dot_queryTerm
    if querySentence.endswith("."):
        querySentence_dot_queryTerm_search_query = "search_query: " + querySentence + queryTerm
    else:
        querySentence_dot_queryTerm_search_query = "search_query: " + querySentence + ". " + queryTerm

    task_dict = {
        "implicit_similarity": "Retrieve a sentence that presents a related idea or parallel insight that functions as an inspiration or serves as an analogy.",
        "explicit_similarity": "Find a sentence that directly expresses the same idea as this query, using similar key terms and concepts, ensuring direct semantic matching.",
        "explicit_contrast": "Find a sentence that clearly opposes the query's main point or presenting a counter-argument, offering a view that directly contradicts it.",
        "more_specific": "Find a sentence that provides a more detailed or precise example of what this query describes, offering a narrowly focused example of the concept.",
        'more_generally': "Find a sentence that captures the broader concept or higher-level idea that this query represents,  reflecting a more general conceptual category.",
        "solving_problems_improvements_mitigation": "Find a sentence that offers a solution, improvement, or mitigation to the problem described in this query by providing actionable strategies."
    }

    # Creating the new column
    default_inst = "Instruct: Given a web search query, retrieve relevant passages that answer the query."
    querySentence_group_search_query = f"search_query: {task_dict.get(queryTerm_group, default_inst)}\nquery: {querySentence}"
    return goldSentence_prompt_search_query, querySentence_prompt_search_query, querySentence_dot_queryTerm_search_query, querySentence_group_search_query


def split_querySentence_goldSentence(df: pd.DataFrame, config: ProjectDataConfig) -> pd.DataFrame:
    # *** FIX: ensure we own the DataFrame and avoid SettingWithCopyWarning ***
    df = df.copy()

    # split multi_text to querySentence and goldSentence based on single_text and queryTerm
    # single_text = queryTerm + goldSentence
    # multi_text = querySentence + single_text + rest_of_text
    multi_text = config.get_preprocessing_s2orc_raw_data_multi_text_col()
    single_text = config.get_preprocessing_s2orc_raw_data_single_text_col()
    queryTerm = config.get_preprocessing_queryTerm_col()

    # querySentence : extract querySentence from multi_text and filter out introductory_clauses
    df['querySentence_with_introductory_clauses'] = df.apply(extract_querySentence, args=(multi_text, single_text),
                                                             axis=1)
    df[['querySentence_introductory_clauses', 'querySentence']] = df.apply(
        lambda row: sep_introductory_clauses(row['querySentence_with_introductory_clauses']), axis='columns',
        result_type='expand')
    df['querySentence'] = df['querySentence'].apply(remove_prefix)

    df[config.get_goldSentence_col()] = df.apply(extract_goldSentence, args=(single_text, queryTerm), axis=1)

    # to avoid cases where queryterms do not start with comma,
    # deleting rows where queryTerm does NOT end with a comma AND goldSentence does NOT start with a comma
    beg_len = len(df)
    df = df[~((~df['queryTerm'].str.endswith(',')) & (~df['goldSentence'].str.startswith(',')))].copy()

    df['goldSentence'] = df['goldSentence'].str.lstrip(',')
    bad_prefixes = [
        'first,', 'second,', 'third,', 'fourth,', 'fifth,', 'sixth,', 'seventh,', 'eighth,', 'ninth,',
        'firstly,', 'secondly,', 'thirdly,', 'fourthly,', 'fifthly,', 'sixthly,', 'first of all,',
        'finally,', 'lastly,'
    ]

    # Function to remove any of the bad prefixes
    def remove_bad_prefix(text, bad_prefixes):
        # Create a regex pattern to match any of the bad prefixes at the start of the string
        pattern = '^(' + '|'.join(map(re.escape, bad_prefixes)) + ')'
        return re.sub(pattern, '', text)

    # Apply the function to the DataFrame column 'text'
    df['querySentence'] = df['querySentence'].apply(remove_bad_prefix, bad_prefixes=bad_prefixes)
    df['querySentence'] = df['querySentence'].str.lstrip(',')

    # Create variations of querySentence for evaluation
    df[['querySentence_prompt',
        'querySentence_dot_queryTerm',
        'querySentence_group_prompt',
        'querySentence_group_s2p_query']] = df.apply(lambda x: create_querySentence_variations(x),
                                               axis='columns',
                                               result_type='expand')

    # Additional search_query/search_document prefixed variants
    df[['goldSentence_prompt_search_query',
        'querySentence_prompt_search_query',
        'querySentence_dot_queryTerm_search_query',
        'querySentence_group_search_query']] = df.apply(
            lambda x: create_querySentence_variations_search_query(x),
            axis='columns',
            result_type='expand')

    df = df.assign(
        goldSentence_prompt="document: " + df.goldSentence.map(str)
    )
    df['query_term_gold'] = df['querySentence'] + " " + df['queryTerm'] + " " + df['goldSentence']

    return df


def normalize_single_multi_text_cols(df, config):
    multi_text = config.get_preprocessing_s2orc_raw_data_multi_text_col()
    single_text = config.get_preprocessing_s2orc_raw_data_single_text_col()
    df[single_text] = df[single_text].str.lower().tolist()
    df[multi_text] = df[multi_text].str.lower().tolist()
    return df


def process_single_input(df, config, num_workers, unmatched_spill_path=None):
    """
    Run the full filtering + splitting + cleaning pipeline on a single DataFrame.
    Returns the processed DataFrame.

    If unmatched_spill_path is provided, rows that don't match any known queryTerm
    are spilled to that path (parquet) before downstream processing and dropped
    from memory. Callers (e.g. the shard worker) then reload the spilled file
    for potential-queryTerm extraction.
    """
    df = normalize_single_multi_text_cols(df, config)

    # ------ filter sentences by queryTerms and split to querySentence and goldSentence
    # Note: blacklist terms are NOT in the active queryTerms_list used for prefix matching
    # (see load_queryTerms_lists — blacklist is a separate group that's excluded from
    # active terms). Sentences starting with a blacklist term therefore flow into the
    # unmatched bucket and get dropped during extraction's known-set subtraction.
    print("filtering sentences by queryTerms", flush=True)
    df_filtered = filter_sentences_by_queryTerms(
        df, config, num_workers=num_workers, unmatched_save_path=unmatched_spill_path
    )
    temp_path = config.get_preprocess_filtered_dataset_path_temp()

    config.print_finished_running(f"filter_sentences_by_queryTerms DONE.\n temp saved to {temp_path}")

    df_filtered = split_querySentence_goldSentence(df_filtered, config)
    config.print_finished_running("split_querySentence_goldSentence DONE")

    if config.get_preprocessing_raw_should_extra_clean():
        df_filtered = clean_citation_rows(df_filtered, 'goldSentence', verbose=True)
        df_filtered = clean_citation_rows(df_filtered, 'querySentence', verbose=True)
        df_filtered = df_filtered[
            df_filtered['querySentence'].notna() &
            (df_filtered['querySentence'].str.strip() != '')
            ].copy()
        print(f"--- get_preprocessing_raw_should_extra_clean is True, end extra cleaning\n"
          f"df length after cleaning: {len(df_filtered)}", flush=True)

    return df_filtered


def extract_potential_queryTerms_from_unmatched(
        unmatched_parquet_path: str,
        config: ProjectDataConfig,
        counts_save_path: str,
        samples_save_path: str,
        marker_save_path: str,
        samples_per_term: int = 5,
):
    """
    Reload unmatched rows from disk and extract potential queryTerm candidates.

    Pipeline:
      1. Read unmatched parquet.
      2. Apply remove_leading_numbers_and_dot to single_text (so '1. more generally, X'
         becomes 'more generally, X' — this happens ONLY here, not in the main filter).
      3. Regex-extract the first 1-4 words followed by a comma at sentence start.
      4. Heuristic filters: drop numeric-only, non-alphanumeric start, 1-2 letter terms.
      5. Subtract known set (queryTerms ∪ blacklist) so the output contains only novel
         candidates for the LLM classifier downstream.
      6. Aggregate to (potential_queryTerm, count) via value_counts — only if non-empty.
      7. Capture up to samples_per_term example sentences per unique term — only if
         non-empty.

    Writes counts and samples only when non-empty (per plan). Always writes an empty
    marker file at marker_save_path when the function completes successfully (empty
    result or not), so the worker's resume logic can distinguish "extraction done,
    nothing found" from "extraction never ran / crashed".

    Returns: {'n_unique': int, 'n_total': int, 'wrote_counts': bool, 'wrote_samples': bool}
    """
    def _finalize_empty(reason):
        # Write the marker so resume logic knows we attempted extraction.
        Path(marker_save_path).touch()
        print(f"[extract] Empty result ({reason}); wrote marker {marker_save_path}", flush=True)
        return {'n_unique': 0, 'n_total': 0, 'wrote_counts': False, 'wrote_samples': False}

    if not Path(unmatched_parquet_path).exists():
        return _finalize_empty(f"no unmatched file at {unmatched_parquet_path}")

    df = pd.read_parquet(unmatched_parquet_path)
    n_in = len(df)
    print(f"[extract] Loaded {n_in:,} unmatched rows from {unmatched_parquet_path}", flush=True)
    if n_in == 0:
        return _finalize_empty("unmatched file had 0 rows")

    single_text = config.get_preprocessing_s2orc_raw_data_single_text_col()

    # Step 2: strip leading numbering on a COPY of single_text (keep original column intact
    # in case we emit samples — samples should show what really appears in the corpus).
    df['_normalized_single_text'] = df[single_text].apply(remove_leading_numbers_and_dot)

    # Step 3: regex extract leading phrase up to first comma (≤4 tokens, matching the
    # convention used by sep_introductory_clauses elsewhere in this file).
    # Pattern matches 1-4 space-separated word-ish tokens followed by a comma.
    first_phrase_re = re.compile(r'^\s*([^\s,]+(?:\s+[^\s,]+){0,3})\s*,')

    def _extract(text):
        if not isinstance(text, str):
            return None
        m = first_phrase_re.match(text)
        if not m:
            return None
        # Include the trailing comma to match the convention used elsewhere in the pipeline
        # (queryTerms like "consequently," are stored with the comma).
        return (m.group(1).strip() + ',').lower()

    df['potential_queryTerm'] = df['_normalized_single_text'].apply(_extract)
    df = df.dropna(subset=['potential_queryTerm'])
    after_regex = len(df)
    print(f"[extract] After regex: {after_regex:,} rows (dropped {n_in - after_regex:,})",
          flush=True)

    if after_regex == 0:
        return _finalize_empty("no rows matched the leading-phrase regex")

    # Step 4: heuristic filters.
    # 4a: drop numeric-only terms (e.g. "2019," or "100 200,")
    mask_numeric = df['potential_queryTerm'].str.match(r'^[\d\s]+,$')
    df = df[~mask_numeric]
    # 4b: must start with a letter or digit (drops punctuation-led rows)
    df = df[df['potential_queryTerm'].str.match(r'^[a-z0-9]', na=False)]
    # 4c: drop 1-2 letter terms ("it,", "a,", "aa,")
    df = df[~df['potential_queryTerm'].str.match(r'^[a-z]{1,2},$')]
    after_heuristics = len(df)
    print(f"[extract] After heuristic filters: {after_heuristics:,} rows "
          f"(dropped {after_regex - after_heuristics:,})", flush=True)

    if after_heuristics == 0:
        return _finalize_empty("all rows dropped by heuristic filters")

    # Step 5: subtract known set (active queryTerms ∪ blacklist).
    # Both come from the same unified CSV now — blacklist is just a group within it.
    queryTerms_list, _, _, blacklist = load_queryTerms_lists(config.get_queryTerms_list_path())
    known_set = set(t.lower() for t in queryTerms_list) | set(t.lower() for t in blacklist)
    df = df[~df['potential_queryTerm'].isin(known_set)]
    after_subtraction = len(df)
    print(f"[extract] After known-set subtraction: {after_subtraction:,} rows "
          f"(dropped {after_heuristics - after_subtraction:,} already-known)", flush=True)

    if after_subtraction == 0:
        return _finalize_empty("all remaining rows were already-known terms")

    # Step 6: aggregate to (potential_queryTerm, count). Only written when non-empty.
    counts = (df['potential_queryTerm']
              .value_counts()
              .rename_axis('potential_queryTerm')
              .reset_index(name='count'))
    counts.to_parquet(counts_save_path, engine='fastparquet', index=False, compression='gzip')
    wrote_counts = True
    print(f"[extract] Wrote {len(counts):,} unique potential queryTerms → {counts_save_path}",
          flush=True)

    # Step 7: sample up to samples_per_term example sentences per unique term.
    # We keep the ORIGINAL single_text (not the number-stripped version) so reviewers
    # see exactly what appears in the source.
    sample_cols = ['potential_queryTerm', single_text]
    # Keep corpusid if present — useful for tracing samples back to source papers.
    if 'corpusid' in df.columns:
        sample_cols = ['potential_queryTerm', 'corpusid', single_text]
    samples = (df[sample_cols]
               .groupby('potential_queryTerm', group_keys=False)
               .head(samples_per_term)
               .reset_index(drop=True))
    wrote_samples = False
    if len(samples) > 0:
        samples.to_parquet(samples_save_path, engine='fastparquet', index=False, compression='gzip')
        wrote_samples = True
        print(f"[extract] Wrote {len(samples):,} sample rows → {samples_save_path}", flush=True)

    # Write the completion marker so resume logic treats this shard as done.
    Path(marker_save_path).touch()

    return {
        'n_unique': len(counts),
        'n_total': int(counts['count'].sum()),
        'wrote_counts': wrote_counts,
        'wrote_samples': wrote_samples,
    }


def _inject_date_range_into_filename(original_path: str, df: pd.DataFrame,
                                     date_col: str = 'publicationdate') -> str:
    """
    Rewrite `original_path` to include a YYYYMMDD_YYYYMMDD date range derived from
    df[date_col].min() / .max(). The range is inserted just before the file's
    extension chain.

    Example:
        original_path: .../s2orc_filtered_sentences__vcs2_qt3_byQueryTermV3.parquet.gz
        min/max:       2015-01-01 / 2026-03-07
        result:        .../s2orc_filtered_sentences__vcs2_qt3_byQueryTermV3_20150101_20260307.parquet.gz

    If df[date_col] is missing or all-null, falls back to the original path
    unchanged and logs a warning. This keeps the save step working when the
    input schema changes.
    """
    if date_col not in df.columns:
        print(f"WARN: column '{date_col}' not found in df; saving without date suffix.",
              flush=True)
        return original_path

    dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
    if len(dates) == 0:
        print(f"WARN: column '{date_col}' has no valid dates; saving without date suffix.",
              flush=True)
        return original_path

    min_str = dates.min().strftime('%Y%m%d')
    max_str = dates.max().strftime('%Y%m%d')
    date_suffix = f"_{min_str}_{max_str}"

    # Split the filename on the first '.' to separate stem from extension chain
    # (handles .parquet, .parquet.gz, .parquet.zst, etc. uniformly).
    p = Path(original_path)
    name = p.name
    first_dot = name.find('.')
    if first_dot == -1:
        # No extension — just append.
        new_name = name + date_suffix
    else:
        stem = name[:first_dot]
        ext_chain = name[first_dot:]  # includes the leading dot
        new_name = stem + date_suffix + ext_chain

    new_path = str(p.with_name(new_name))
    print(f"Date range {min_str} — {max_str} injected into output filename:\n"
          f"  {original_path}\n  → {new_path}", flush=True)
    return new_path


def _serialize_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert any column whose values are lists or numpy arrays to JSON strings.
    fastparquet cannot serialize object columns containing Python lists.
    Call this before any to_parquet(..., engine='fastparquet') call.
    """
    df = df.copy()
    for col in df.columns:
        sample = df[col].dropna()
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, np.ndarray)):
            df[col] = df[col].apply(
                lambda x: json.dumps(x.tolist() if isinstance(x, np.ndarray) else x)
                if isinstance(x, (list, np.ndarray)) else x
            )
    return df


def _merge_potential_queryTerms_global(
        temp_output_dir: Path,
        output_dir: Path,
        min_count: int,
        samples_per_term: int,
) -> None:
    """
    Global merge step for per-shard potential queryTerm extractions.

    Reads all potential_queryTerms_shard_*.parquet from temp_output_dir,
    groupby-sums counts across shards, sorts by frequency descending,
    and writes:

      - {output_dir}/potential_queryTerms_distribution.csv
            Full sorted distribution (no threshold applied). CSV for easy review.
      - {output_dir}/potential_queryTerms_distribution.parquet
            Same as above, parquet for programmatic downstream use.
      - {output_dir}/potential_queryTerms_for_llm.parquet
            Threshold-filtered (count >= min_count) + up to `samples_per_term`
            example sentences per term. This is the input for the LLM
            classifier in the downstream 7-way classification step.

    Empty cases (no per-shard files, or threshold excludes everything) are
    logged and the function returns without erroring.
    """
    print(f"\n{'=' * 60}\n"
          f"Global merge of potential queryTerms\n"
          f"  temp_output_dir: {temp_output_dir}\n"
          f"  output_dir:      {output_dir}\n"
          f"  min_count:       {min_count}\n"
          f"  samples_per_term:{samples_per_term}\n"
          f"{'=' * 60}", flush=True)

    # -------- Merge per-shard counts --------
    counts_files = sorted(glob.glob(str(temp_output_dir / "potential_queryTerms_shard_*.parquet")))
    if not counts_files:
        print("[merge] No per-shard potential-queryTerm files found; nothing to merge.",
              flush=True)
        return

    print(f"[merge] Combining {len(counts_files)} per-shard counts files...", flush=True)
    per_shard_counts = [pd.read_parquet(f) for f in counts_files]
    counts_df = pd.concat(per_shard_counts, ignore_index=True)
    del per_shard_counts
    print(f"[merge] Per-shard total rows (pre-groupby): {len(counts_df):,}", flush=True)

    # Groupby-sum: same term appearing in N shards gets its counts added.
    global_counts = (counts_df
                     .groupby('potential_queryTerm', as_index=False)['count']
                     .sum()
                     .sort_values('count', ascending=False)
                     .reset_index(drop=True))
    del counts_df
    print(f"[merge] Global unique potential queryTerms: {len(global_counts):,}", flush=True)
    print(f"[merge] Total occurrences across corpus:    "
          f"{int(global_counts['count'].sum()):,}", flush=True)

    # Save the full distribution (no threshold) for review.
    dist_csv = output_dir / "potential_queryTerms_distribution.csv"
    dist_parquet = output_dir / "potential_queryTerms_distribution.parquet"
    global_counts.to_csv(dist_csv, index=False)
    global_counts.to_parquet(dist_parquet, engine='fastparquet', index=False, compression='gzip')
    print(f"[merge] Wrote full distribution → {dist_csv} ({len(global_counts):,} terms)",
          flush=True)

    # -------- Apply frequency threshold --------
    above_threshold = global_counts[global_counts['count'] >= min_count].reset_index(drop=True)
    print(f"[merge] After min_count >= {min_count}: {len(above_threshold):,} terms "
          f"(dropped {len(global_counts) - len(above_threshold):,} long-tail)", flush=True)

    if len(above_threshold) == 0:
        print(f"[merge] No terms survived the frequency threshold. "
              f"Skipping LLM-input file write.", flush=True)
        return

    # -------- Merge per-shard samples --------
    samples_files = sorted(glob.glob(str(temp_output_dir / "potential_samples_shard_*.parquet")))
    if samples_files:
        print(f"[merge] Combining {len(samples_files)} per-shard samples files...", flush=True)
        per_shard_samples = [pd.read_parquet(f) for f in samples_files]
        samples_df = pd.concat(per_shard_samples, ignore_index=True)
        del per_shard_samples
        print(f"[merge] Per-shard total sample rows (pre-trim): {len(samples_df):,}",
              flush=True)

        # Keep only samples for terms that survived the threshold.
        samples_df = samples_df[samples_df['potential_queryTerm'].isin(set(above_threshold['potential_queryTerm']))]

        # Trim to samples_per_term per unique term (across all shards).
        samples_df = (samples_df
                      .groupby('potential_queryTerm', group_keys=False)
                      .head(samples_per_term)
                      .reset_index(drop=True))
        print(f"[merge] After trim to {samples_per_term}/term: {len(samples_df):,} sample rows",
              flush=True)
    else:
        print("[merge] No per-shard samples files found. Proceeding without samples.",
              flush=True)
        samples_df = None

    # -------- Build LLM-input file: one row per surviving term, with samples as a list --------
    if samples_df is not None and len(samples_df) > 0:
        # Figure out which column holds the sample text. It's whichever column isn't
        # 'potential_queryTerm' or 'corpusid'. In practice this is the single_text column.
        text_col_candidates = [c for c in samples_df.columns
                               if c not in ('potential_queryTerm', 'corpusid')]
        if not text_col_candidates:
            print("[merge] WARN: samples_df has no text column, dropping samples.", flush=True)
            samples_agg = None
        else:
            text_col = text_col_candidates[0]
            samples_agg = (samples_df
                           .groupby('potential_queryTerm')[text_col]
                           .apply(list)
                           .rename('sample_sentences')
                           .reset_index())
    else:
        samples_agg = None

    llm_input = above_threshold.copy()
    if samples_agg is not None:
        llm_input = llm_input.merge(samples_agg, on='potential_queryTerm', how='left')
        # Any term with no samples gets an empty list (not NaN) for clean downstream handling.
        llm_input['sample_sentences'] = llm_input['sample_sentences'].apply(
            lambda x: x if isinstance(x, list) else []
        )

    # fastparquet doesn't serialize list columns — use the existing helper.
    llm_input_serialized = _serialize_list_columns(llm_input)
    llm_input_path = output_dir / "potential_queryTerms_for_llm.parquet"
    llm_input_serialized.to_parquet(
        llm_input_path, engine='fastparquet', index=False, compression='gzip'
    )
    print(f"[merge] Wrote LLM-input file → {llm_input_path} "
          f"({len(llm_input):,} terms above threshold)", flush=True)

    # Show the top of the distribution as a sanity check.
    top_n = min(20, len(global_counts))
    print(f"\n[merge] Top {top_n} potential queryTerms by corpus frequency:", flush=True)
    for _, row in global_counts.head(top_n).iterrows():
        print(f"  {row['count']:>8,}  {row['potential_queryTerm']}", flush=True)


def _discover_input_files(shards_dir: Path):
    """
    Resolve input data files in shards_dir.

    Supports two input formats:

    1. Merged corpus from 00_process_shard.py:
       s2orc_cs_sentences_<8digits>_<8digits>.parquet.gz
       Discovered directly by filename pattern. No .done marker exists for this
       format because 01 writes the merged file once at the end of its run;
       if the file is on disk at all, it's complete.

    2. Re-sharded corpus files:
       s2orc_cs_sentences_<8digits>_<8digits>.parquet.zst
       Discovered via sibling .done markers (the shard producer writes a marker
       only after the data file is fully flushed, so this excludes partial writes).

    Returns a sorted list of data-file paths as strings. If both formats are
    present in the same directory, both are returned (sorted).
    """
    data_files = []
    rejected = []

    # --- Format 1: merged .parquet.gz corpus (no .done marker) ---
    gz_stem = re.compile(r'^s2orc_cs_sentences_\d{8}_\d{8}\.parquet\.gz$')
    gz_glob = sorted(glob.glob(str(shards_dir / "s2orc_cs_sentences_*.parquet.gz")))
    for path in gz_glob:
        if gz_stem.match(Path(path).name):
            data_files.append(path)
        else:
            rejected.append(Path(path).name)

    # --- Format 2: .zst shards via .done markers ---
    done_marker_pattern = str(shards_dir / "s2orc_cs_sentences_*.parquet.zst.done")
    done_markers = sorted(glob.glob(done_marker_pattern))
    zst_stem = re.compile(r'^s2orc_cs_sentences_\d{8}_\d{8}\.parquet\.zst$')

    for marker in done_markers:
        data_path = marker[:-len(".done")]
        data_filename = Path(data_path).name
        if not zst_stem.match(data_filename):
            rejected.append(data_filename)
            continue
        if Path(data_path).exists():
            data_files.append(data_path)
        else:
            print(f"WARN: .done marker {marker} has no matching data file at {data_path}; "
                  f"skipping.", flush=True)

    if rejected:
        print(f"[discover] Skipping {len(rejected)} file(s) that match a glob but "
              f"not the strict s2orc_cs_sentences_<date>_<date> stem pattern: {rejected}",
              flush=True)

    return sorted(data_files)


def _process_shard_worker(args):
    """
    Top-level worker function for ProcessPoolExecutor.
    Loads config from path (avoids pickling config object), reads one input shard
    (or one row-group of a single large file), runs the full pipeline single-threaded
    (no inner Pool), and saves results to temp files.

    `args` is a dict with keys:
        shard_idx        : int, index used for temp filenames and logging.
        shard_path       : str, path to the parquet file to read.
        config_path      : str, path to config JSON.
        total_shards     : int, total shards/row-groups for progress logging.
        temp_output_dir  : str, where to write temp outputs.
        row_group_idx    : int or None (default None).
                           If None → read the whole file (sharded mode).
                           If int  → read only that row-group of the file
                                     (single-file row-group-streaming mode).

    Outputs per shard (filenames indexed by shard_idx):
      - filtered_shard_{idx}.parquet        : matched-row pipeline output
      - unmatched_shard_{idx}.parquet       : transient spill; deleted at end of shard
      - potential_queryTerms_shard_{idx}.parquet : (term, count) aggregation (non-empty only)
      - potential_samples_shard_{idx}.parquet    : sample sentences per term (non-empty only)
      - extracted_shard_{idx}.marker        : touch-file marking extraction stage complete

    Resume: skip shard only if BOTH filtered_shard_{idx}.parquet AND
    extracted_shard_{idx}.marker exist. Otherwise re-run the whole shard.

    Returns (shard_idx, filtered_temp_path_or_None, n_filtered_rows).
    """
    shard_idx = args['shard_idx']
    shard_path = args['shard_path']
    config_path = args['config_path']
    total_shards = args['total_shards']
    temp_output_dir = args['temp_output_dir']
    row_group_idx = args.get('row_group_idx', None)

    temp_dir = Path(temp_output_dir)
    filtered_out = temp_dir / f"filtered_shard_{shard_idx:04d}.parquet"
    unmatched_out = temp_dir / f"unmatched_shard_{shard_idx:04d}.parquet"
    potential_counts_out = temp_dir / f"potential_queryTerms_shard_{shard_idx:04d}.parquet"
    potential_samples_out = temp_dir / f"potential_samples_shard_{shard_idx:04d}.parquet"
    extraction_marker = temp_dir / f"extracted_shard_{shard_idx:04d}.marker"

    # Label used in log messages. In row-group mode, clarify what we're reading.
    label = (f"Shard {shard_idx + 1}/{total_shards}"
             if row_group_idx is None
             else f"RowGroup {shard_idx + 1}/{total_shards}")

    # --- Resume: skip only if BOTH the filtered output and the extraction marker exist. ---
    if filtered_out.exists() and extraction_marker.exists():
        try:
            n = len(pd.read_parquet(filtered_out, columns=["corpusid"]))
            print(f"[worker] {label}: SKIP — filtered + marker both exist ({n:,} rows)",
                  flush=True)
            return shard_idx, str(filtered_out), n
        except Exception:
            # Filtered parquet unreadable — wipe and re-run.
            filtered_out.unlink(missing_ok=True)
            extraction_marker.unlink(missing_ok=True)

    # If only one of the two exists, treat the shard as incomplete and re-run from scratch.
    # Clean up any stragglers so we don't mix new and old data.
    for p in (filtered_out, unmatched_out, potential_counts_out, potential_samples_out,
              extraction_marker):
        p.unlink(missing_ok=True)

    try:
        if row_group_idx is None:
            print(f"[worker] {label}: starting — {shard_path}", flush=True)
        else:
            print(f"[worker] {label}: starting — {shard_path} (row group {row_group_idx})",
                  flush=True)
        t0 = time.time()

        # Reconstruct config inside worker (avoids pickle issues)
        config = ProjectDataConfig(config_path)

        # Read input: either the whole file (sharded mode) or one row-group
        # (single-file row-group-streaming mode). Row-group reads are independent
        # in pyarrow — each worker opens its own file handle, reads just its
        # assigned row-group, and closes. No coordination needed between workers.
        if row_group_idx is None:
            df_shard = pd.read_parquet(shard_path)
        else:
            df_shard = pq.ParquetFile(shard_path).read_row_group(row_group_idx).to_pandas()
        print(f"[worker] {label}: loaded {len(df_shard):,} rows", flush=True)

        # Filter to rows with multi_text
        multi_text_col = config.get_preprocessing_s2orc_raw_data_multi_text_col()
        df_shard = df_shard[df_shard[multi_text_col].notna()].copy()
        if len(df_shard) == 0:
            print(f"[worker] {label}: no multi_text rows, skipping.", flush=True)
            # Still mark extraction as complete so resume logic doesn't re-run this shard.
            extraction_marker.touch()
            return shard_idx, None, 0

        # ------ Stage 1: matched-row filtering pipeline ------
        # Pass the unmatched spill path so filter_sentences_by_queryTerms writes
        # unmatched rows to disk and drops them from memory before downstream.
        df_result = process_single_input(
            df_shard, config, num_workers=1,
            unmatched_spill_path=str(unmatched_out),
        )
        # Free the shard's raw input now — we only need df_result downstream.
        del df_shard

        elapsed_stage1 = time.time() - t0
        n = len(df_result) if df_result is not None else 0

        if df_result is not None and n > 0:
            df_result = _serialize_list_columns(df_result)
            df_result.to_parquet(filtered_out, index=False, compression="gzip")
            print(f"[worker] {label}: stage1 done in "
                  f"{elapsed_stage1:.0f}s — {n:,} matched rows saved to {filtered_out}",
                  flush=True)
        else:
            print(f"[worker] {label}: stage1 done in "
                  f"{elapsed_stage1:.0f}s — 0 matched rows", flush=True)
        # Free the matched df before running extraction so memory never holds both halves.
        del df_result

        # ------ Stage 2: potential queryTerm extraction on unmatched rows ------
        t1 = time.time()
        extract_summary = extract_potential_queryTerms_from_unmatched(
            unmatched_parquet_path=str(unmatched_out),
            config=config,
            counts_save_path=str(potential_counts_out),
            samples_save_path=str(potential_samples_out),
            marker_save_path=str(extraction_marker),
        )
        elapsed_stage2 = time.time() - t1
        print(f"[worker] {label}: stage2 extraction done in "
              f"{elapsed_stage2:.0f}s — {extract_summary['n_unique']:,} unique terms, "
              f"{extract_summary['n_total']:,} total occurrences", flush=True)

        # ------ Cleanup: the unmatched spill file has served its purpose ------
        unmatched_out.unlink(missing_ok=True)

        total_elapsed = time.time() - t0
        print(f"[worker] {label}: ALL stages done in "
              f"{total_elapsed:.0f}s", flush=True)

        if n > 0:
            return shard_idx, str(filtered_out), n
        else:
            return shard_idx, None, 0

    except Exception as e:
        print(f"[worker] {label}: FAILED — {e}", flush=True)
        traceback.print_exc()
        # On failure, remove marker so resume re-runs this shard.
        extraction_marker.unlink(missing_ok=True)
        return shard_idx, None, 0


def main(config_path: str) -> None:
    print("1_filter_sentences_by_queryTerms.py:\n"
          "------------ preprocess_sentences_by_queryTerms ------------", flush=True)

    # ------load config parameters
    print("loading config", flush=True)
    config = ProjectDataConfig(config_path)
    config.print_start_running_preprocessing()

    if args.shards_dir:
        # =====================================================================
        # Discovery: find input files via the .done-marker convention.
        # 1 file  → single-file parallel mode (chunks across workers).
        # N files → sharded parallel mode (one worker per file).
        # =====================================================================
        shards_dir = Path(args.shards_dir)
        shard_files = _discover_input_files(shards_dir)
        if not shard_files:
            print(f"ERROR: No s2orc_cs_sentences_<date>_<date>.parquet.gz files or "
                  f".parquet.zst.done markers found in {shards_dir}", flush=True)
            return

        # Create temp directory for intermediate shard results
        temp_output_dir = shards_dir / "_filtered_temp"
        temp_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Temp shard results dir: {temp_output_dir}", flush=True)

        # --- Build worker_args: one entry per work unit (row-group OR whole file) ---
        if len(shard_files) == 1:
            # -----------------------------------------------------------------
            # SINGLE-FILE ROW-GROUP STREAMING MODE.
            # One large file, but read it as N row-groups in parallel. Each worker
            # reads just its assigned row-group from the same file — pyarrow
            # handles concurrent reads cleanly, and we never load the whole file.
            # -----------------------------------------------------------------
            input_path = shard_files[0]
            print(f"SINGLE-FILE MODE: one input file found — {input_path}", flush=True)

            # Peek at the file to enumerate row groups. This is a metadata-only
            # read (fast, doesn't load the data).
            pf = pq.ParquetFile(input_path)
            num_row_groups = pf.num_row_groups
            total_rows = pf.metadata.num_rows
            del pf
            print(f"File has {num_row_groups} row groups, "
                  f"{total_rows:,} total rows. Streaming them in parallel "
                  f"with num_workers={args.num_workers}.", flush=True)

            if num_row_groups == 0:
                print(f"ERROR: {input_path} has 0 row groups, nothing to process.",
                      flush=True)
                return

            worker_args = [
                {
                    'shard_idx': rg_idx,
                    'shard_path': input_path,
                    'config_path': config_path,
                    'total_shards': num_row_groups,
                    'temp_output_dir': str(temp_output_dir),
                    'row_group_idx': rg_idx,
                }
                for rg_idx in range(num_row_groups)
            ]
            total_units = num_row_groups
            unit_label = "row groups"
        else:
            # -----------------------------------------------------------------
            # SHARDED MODE.
            # N>=2 input files; one worker per file. Each worker reads its whole
            # file (row_group_idx=None).
            # -----------------------------------------------------------------
            print(f"SHARDED MODE: found {len(shard_files)} shard files in {shards_dir}",
                  flush=True)
            worker_args = [
                {
                    'shard_idx': idx,
                    'shard_path': path,
                    'config_path': config_path,
                    'total_shards': len(shard_files),
                    'temp_output_dir': str(temp_output_dir),
                    'row_group_idx': None,
                }
                for idx, path in enumerate(shard_files)
            ]
            total_units = len(shard_files)
            unit_label = "shards"

        # --- Shared ProcessPoolExecutor loop: works identically for both modes. ---
        print(f"Launching {args.num_workers} parallel workers over {total_units} "
              f"{unit_label}...", flush=True)
        completed_count = 0
        total_matched_rows = 0
        t_start = time.time()

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(_process_shard_worker, a): a['shard_idx']
                       for a in worker_args}

            for future in as_completed(futures):
                shard_idx = futures[future]
                try:
                    idx, temp_path, n_rows = future.result(timeout=3600)
                    total_matched_rows += n_rows

                    completed_count += 1
                    elapsed = time.time() - t_start
                    remaining = total_units - completed_count
                    eta = elapsed / completed_count * remaining
                    print(f"  Progress: {completed_count}/{total_units} {unit_label} done, "
                          f"{total_matched_rows:,} total matched rows — "
                          f"ETA: {eta / 60:.0f} min", flush=True)
                except Exception as e:
                    completed_count += 1
                    print(f"  {unit_label[:-1].capitalize()} {shard_idx}: "
                          f"Worker EXCEPTION — {e}", flush=True)
                    traceback.print_exc()

        total_time = time.time() - t_start
        print(f"\nAll {unit_label} processed in {total_time / 60:.1f} min", flush=True)

        # --- Merge all temp shard files ---
        # sorted() by filename keeps row-groups/shards in their original order
        # (filenames use zero-padded shard_idx, so numeric and lexicographic orders agree).
        temp_files = sorted(glob.glob(str(temp_output_dir / "filtered_shard_*.parquet")))
        if not temp_files:
            print(f"ERROR: No rows survived filtering across all {unit_label}.",
                  flush=True)
            return

        print(f"Combining {len(temp_files)} temp shard files from disk...", flush=True)
        dfs = [pd.read_parquet(f) for f in temp_files]
        df_filtered = pd.concat(dfs, ignore_index=True)
        del dfs
        print(f"Combined total: {len(df_filtered):,} rows", flush=True)

        # Global dedup across shards / row-groups
        before_dedup = len(df_filtered)
        hashable_cols = [col for col in df_filtered.columns
                         if not isinstance(df_filtered[col].dropna().iloc[0] if len(df_filtered[col].dropna()) > 0 else None,
                                           (list, np.ndarray))]
        df_filtered = df_filtered.drop_duplicates(subset=hashable_cols, keep="first")
        print(f"After dedup: {len(df_filtered):,} rows "
              f"(removed {before_dedup - len(df_filtered):,})", flush=True)

        # =====================================================================
        # Global merge: potential queryTerms across all shards.
        # Works for both single-file (1 set of per-shard files) and sharded (N sets).
        # =====================================================================
        _merge_potential_queryTerms_global(
            temp_output_dir=Path(temp_output_dir),
            output_dir=Path(args.shards_dir),
            min_count=args.potential_min_count,
            samples_per_term=args.potential_samples_per_term,
        )

    else:
        print("ERROR: --shards-dir is required.", flush=True)
        return

    # ------ Serialize list columns before saving ------
    df_filtered = _serialize_list_columns(df_filtered)

    # ------ save to filtered_output_dir (with date range injected into filename) ------
    filtered_output_path = config.get_preprocess_filtered_dataset_path()
    filtered_output_path = _inject_date_range_into_filename(filtered_output_path, df_filtered)
    print(f"Saving to filtered_output_dir {filtered_output_path}\n"
          f"df_filtered after all processing len {len(df_filtered)}\n", flush=True)
    df_filtered.to_parquet(filtered_output_path, engine='fastparquet', index=False, compression='gzip')

    # ------ Cleanup: remove all per-shard temp files now that the final outputs are saved.
    # Skipped if --keep-temps is set (useful for debugging or iterating on the merge step
    # without re-running workers).
    if args.shards_dir and not args.keep_temps:
        temp_output_dir = Path(args.shards_dir) / "_filtered_temp"
        if temp_output_dir.exists():
            print(f"Cleaning up temp directory {temp_output_dir}...", flush=True)
            shutil.rmtree(temp_output_dir)
            print(f"Removed {temp_output_dir}", flush=True)

    print("--- done ---", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', required=True,
                        help='Path to the JSON configuration file')
    parser.add_argument('--num_workers', help='number of cpus to use for parallel processing', type=int, default=30)
    parser.add_argument('--shards-dir', type=str, required=True,
                        help='Directory containing the sentence corpus: the merged '
                             's2orc_cs_sentences_<date>_<date>.parquet.gz from 00_process_shard.py, '
                             'and/or s2orc_cs_sentences_*.parquet.zst files with sibling .done '
                             'markers. If discovery resolves to 1 file, runs single-file '
                             'row-group-streaming mode (each row-group processed in parallel as a '
                             'pseudo-shard); with 2+ files, runs sharded mode (one worker per file).')
    parser.add_argument('--potential-min-count', type=int, default=50,
                        help='Minimum corpus-wide occurrence count for a potential queryTerm '
                             'to survive the global frequency filter. Only applied to the '
                             'potential-queryTerms output. Default: 50.')
    parser.add_argument('--potential-samples-per-term', type=int, default=5,
                        help='Max sample sentences to keep per potential queryTerm in the '
                             'final merged samples output. Default: 5.')
    parser.add_argument('--keep-temps', action='store_true',
                        help='Keep the per-shard/per-row-group temp directory after the final '
                             'outputs are written. Default: delete. Useful for debugging or '
                             'iterating on the merge step without re-running workers.')
    args = parser.parse_args()
    main(args.config_path)