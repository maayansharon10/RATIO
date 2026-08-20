import datetime
import json
import os
from typing import Optional, Dict, Any, List

import src.utils as utils


class ProjectDataConfig:
    def __init__(self, config_path: str):
        self.config_path: str = config_path
        self.version: Optional[str] = None
        # Preprocessing data attributes
        self.preprocessing_s2orc_raw_data_path: Optional[str] = None
        self.preprocessing_s2orc_raw_data_single_text_col: Optional[str] = None
        self.preprocessing_s2orc_raw_data_multi_text_col: Optional[str] = None
        self.preprocessing_queryTerm_col: Optional[str] = None
        self.preprocessing_queryTerms_version: Optional[str] = None
        self.preprocessing_manual_queryTerms_list_path: Optional[str] = None
        self.preprocessing_queryTerms_list_path: Optional[str] = None
        self.queryTerms_groups: Optional[Dict[str, List[str]]] = None
        self.preprocessing_filtered_output_dir: Optional[str] = None
        self.preprocessing_sub_datasets_dir: Optional[str] = None
        self.preprocessing_raw_data_description: Optional[str] = None
        self.preprocessing_raw_should_extra_clean: Optional[str] = None
        self.querySentence_col: Optional[str] = None
        self.goldSentence_col: Optional[str] = None
        # createDataset and subdatasets attributes
        self.createDataset_output_dir: Optional[str] = None
        self.createDataset_original_split_dir: Optional[str] = None
        self.createDataset_sub_datasets_names: Optional[Dict[str, str]] = None
        self.createDataset_dataset_description: Optional[str] = None
        self.createDataset_mutual_columns: Optional[List[str]] = None
        self.dataset_sub_combinations: Optional[Dict[str, List[str]]] = None
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r') as file:
                config_data: Dict[str, Any] = json.load(file)
                self._set_version(config_data)
                self._set_attributes_preprocessing_data(config_data)
                self._set_attributes_createDataset(config_data)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Config file at {self.config_path} is not a valid JSON file")

    def _parse_bool(self, value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v == "true":
                return True
            if v == "false":
                return False
        raise ValueError(f"Invalid boolean for {field_name}: {value!r}")

    def _get_required_field(self, data: Dict[str, Any], field_name: str, is_bool=False) -> Any:
        if field_name not in data:
            raise ValueError(f"Missing required field: {field_name}")
        if is_bool:
            return self._parse_bool(data[field_name], field_name)
        return data[field_name]

    def _set_version(self, config_data: Dict[str, Any]):
        self.version = self._get_required_field(config_data, 'version')

    def _set_attributes_preprocessing_data(self, config_data: Dict[str, Any]):
        preprocessing_data = self._get_required_field(config_data, 'preprocessing_data')
        self.preprocessing_s2orc_raw_data_path = self._get_required_field(preprocessing_data,
                                                                          's2orc_raw_data_path')
        self.preprocessing_s2orc_raw_data_single_text_col = self._get_required_field(preprocessing_data,
                                                                                     's2orc_raw_data_single_text_col')
        self.preprocessing_s2orc_raw_data_multi_text_col = self._get_required_field(preprocessing_data,
                                                                                    's2orc_raw_data_multi_text_col')
        self.preprocessing_raw_should_extra_clean = self._get_required_field(preprocessing_data,
                                                                                    'should_extra_clean', is_bool=True)

        extract_columns = self._get_required_field(preprocessing_data, 'extract_columns')
        self.querySentence_col = self._get_required_field(extract_columns, 'querySentence')
        self.goldSentence_col = self._get_required_field(extract_columns, 'goldSentence')

        query_terms = self._get_required_field(preprocessing_data, 'QueryTerms')
        self.preprocessing_queryTerm_col = self._get_required_field(query_terms,
                                                                    'queryTerm_col')
        self.preprocessing_queryTerms_version = self._get_required_field(query_terms,
                                                                         'QueryTerms_version')
        self.preprocessing_queryTerms_list_path = self._get_required_field(query_terms,
                                                                           'QueryTerms_list_Path')
        self.preprocessing_manual_queryTerms_list_path = self._get_required_field(query_terms,
                                                                                'manual_QueryTerms_list_Path')
        self.queryTerms_groups = self._get_required_field(query_terms, "queryTerms_groups")

        self.preprocessing_filtered_output_dir = self._get_required_field(preprocessing_data, 'filtered_output_dir')
        self.preprocessing_sub_datasets_dir = self._get_required_field(preprocessing_data,
                                                                       'sub_datasets_dir')
        self.preprocessing_raw_data_description = self._get_required_field(preprocessing_data, 'raw_data_description')

    def _set_attributes_createDataset(self, config_data: Dict[str, Any]):
        createDataset = self._get_required_field(config_data, 'createDataset')
        self.createDataset_original_split_dir = self._get_required_field(createDataset, 'dataset_original_split_dir')
        self.createDataset_output_dir = self._get_required_field(createDataset, 'dataset_output_dir')
        sub_datasets_names = self._get_required_field(createDataset, 'sub_datasets_names')
        self.createDataset_sub_datasets_names = sub_datasets_names
        self.createDataset_mutual_columns = self._get_required_field(createDataset, 'mutual_columns')
        self.createDataset_dataset_description = self._get_required_field(createDataset, 'dataset_description')
        self.subdataset_combinations = self._get_required_field(createDataset, 'dataset_sub_combinations')

    # Getter methods
    def get_version(self) -> Optional[str]:
        return self.version

    def get_preprocessing_s2orc_raw_data_path(self) -> Optional[str]:
        return self.preprocessing_s2orc_raw_data_path

    def get_preprocessing_s2orc_raw_data_single_text_col(self) -> Optional[str]:
        return self.preprocessing_s2orc_raw_data_single_text_col

    def get_preprocessing_s2orc_raw_data_multi_text_col(self) -> Optional[str]:
        return self.preprocessing_s2orc_raw_data_multi_text_col

    def get_querySentence_col(self) -> Optional[str]:
        return self.querySentence_col

    def get_goldSentence_col(self) -> Optional[str]:
        return self.goldSentence_col

    def get_preprocessing_queryTerm_col(self) -> Optional[str]:
        return self.preprocessing_queryTerm_col

    def get_preprocessing_queryTerms_version(self) -> Optional[str]:
        return self.preprocessing_queryTerms_version

    def get_queryTerms_list_path(self) -> Optional[str]:
        return self.preprocessing_queryTerms_list_path

    def get_manual_queryTerms_list_path(self) -> Optional[str]:
        return self.preprocessing_manual_queryTerms_list_path


    def get_queryTerms_groups(self) -> Optional[Dict[str, List[str]]]:
        return self.queryTerms_groups

    def get_preprocessing_filtered_output_dir(self) -> Optional[str]:
        return self.preprocessing_filtered_output_dir

    def get_preprocessing_sub_datasets_dir(self) -> Optional[str]:
        return self.preprocessing_sub_datasets_dir

    def get_preprocessing_raw_data_description(self) -> Optional[str]:
        return self.preprocessing_raw_data_description

    def get_preprocessing_raw_should_extra_clean(self) -> Optional[bool]:
        return self.preprocessing_raw_should_extra_clean

    def get_createDataset_output_dir(self) -> Optional[str]:
        return self.createDataset_output_dir

    def get_createDataset_original_split_dir(self) -> Optional[str]:
        return self.createDataset_original_split_dir

    def get_createDataset_sub_datasets_names(self) -> Optional[Dict[str, str]]:
        return self.createDataset_sub_datasets_names

    def get_createDataset_dataset_description(self) -> Optional[str]:
        return self.createDataset_dataset_description

    def get_createDataset_mutual_columns(self) -> Optional[List[str]]:
        return self.createDataset_mutual_columns

    def get_createDataset_sub_combinations(self) -> Optional[Dict[str, List[str]]]:
        return self.subdataset_combinations

    # print methods
    def print_start_running_preprocessing(self) -> None:
        time_now = datetime.datetime.now()
        print(f"---- Start running version {self.get_version()} \n"
              f"---- Current date and time : {time_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"---- QueryTerms_version: {self.get_preprocessing_queryTerms_version()}\n"
              f"---- raw_data_description: {self.get_preprocessing_raw_data_description()}", flush=True)

    def print_start_running_createDataset(self, msg: str = "") -> None:
        time_now = datetime.datetime.now()
        print(f"---- start running {msg}\n"
              f"---- version: {self.get_version()}\n"
              f"---- Current date and time :{time_now.strftime('%Y-%m-%d %H:%M:%S')}\n"
              f"---- QueryTerms_version: {self.get_preprocessing_queryTerms_version()}\n"
              f"---- sub_datasets_names: {self.get_createDataset_sub_datasets_names().keys()}\n"
              f"---- dataset_description: {self.get_createDataset_dataset_description()}", flush=True)

    def print_finished_running(self, msg: str = "") -> None:
        time_now = datetime.datetime.now()
        print(f"---- Finished running {msg}\n"
              f"---- version: {self.get_version()}\n"
              f"---- Current date and time : {time_now.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    def get_preprocess_filtered_dataset_path_temp(self):
        dir_version = utils.create_path_from_dir_filename(self.get_preprocessing_filtered_output_dir(),     # data/filtered
                                                          self.get_version())                               # /cs1_qt1
        filename = f's2orc_filtered_sentences__v{self.get_version()}_byQueryTermV{self.get_preprocessing_queryTerms_version()}_temp.parquet.gz'
        return utils.create_path_from_dir_filename(dir_version,
                                                   filename)


    def get_preprocess_filtered_dataset_path(self,
                                             date_from: Optional[str] = None,
                                             date_to: Optional[str] = None) -> str:
        """
        Build the path for the final filtered dataset.

        If both `date_from` and `date_to` are provided (expected format: 'YYYYMMDD'),
        the filename includes a `_{date_from}_{date_to}` suffix before the extension —
        e.g. 's2orc_filtered_sentences__vcs2_qt3_byQueryTermV3_20150101_20260307.parquet.gz'.
        Otherwise, the filename has no date suffix (backward-compatible with callers
        that don't pass dates).

        Callers producing data from a known date range should compute the range from
        their dataframe's publicationdate column and pass it in; the resulting
        filename then always reflects the actual data range.
        """
        dir_version = utils.create_path_from_dir_filename(self.get_preprocessing_filtered_output_dir(),     # data/filtered
                                                          self.get_version())                               # /cs1_qt1
        date_suffix = ""
        if date_from and date_to:
            date_suffix = f"_{date_from}_{date_to}"
        filename = (f's2orc_filtered_sentences__v{self.get_version()}'
                    f'_byQueryTermV{self.get_preprocessing_queryTerms_version()}'
                    f'{date_suffix}.parquet.gz')
        return utils.create_path_from_dir_filename(dir_version,
                                                   filename)

    def get_createData_subDataset_path(self, sub_dataset_name: str) -> str:
        dir_version = utils.create_path_from_dir_filename(self.get_preprocessing_sub_datasets_dir(),
                                                          self.get_version())
        filename = f's2orc_filtered_{sub_dataset_name}__v{self.get_version()}_byQueryTermV{self.get_preprocessing_queryTerms_version()}.parquet.gz'
        return utils.create_path_from_dir_filename(dir_version, filename)


    def get_dataset_path_setup2(self, dataset_name: str, split_part: str) -> str:
        dir_version = utils.create_path_from_dir_filename(self.get_createDataset_original_split_dir(),
                                                          self.get_version())
        if split_part:
            filename = f's2orc_filtered__v{self.get_version()}_byQueryTermV{self.get_preprocessing_queryTerms_version()}__{dataset_name}__{split_part}.parquet.gz'
        else:
            filename = f's2orc_filtered__v{self.get_version()}_byQueryTermV{self.get_preprocessing_queryTerms_version()}__{dataset_name}.parquet.gz'

        return utils.create_path_from_dir_filename(dir_version, filename)


    def get_dataset_path(self, dataset_name: str, split_part: str,
                         subdir: str = "", filename_name: str = None) -> str:
        """
        Path: <original_split_dir>/<version>/[<subdir>/]<dataset_name>/[<split_part>/]<file>
        filename_name defaults to dataset_name; subdir="" reproduces the non-temporal layout.
        """
        dir_full = utils.create_path_from_dir_filename(self.get_createDataset_original_split_dir(),
                                                       self.get_version())
        if subdir:
            dir_full = utils.create_path_from_dir_filename(dir_full, subdir)
        dir_full = utils.create_path_from_dir_filename(dir_full, dataset_name)
        if split_part:
            dir_full = utils.create_path_from_dir_filename(dir_full, split_part)

        name_token = filename_name if filename_name else dataset_name
        base = f's2orc_filtered__v{self.get_version()}_byQueryTermV{self.get_preprocessing_queryTerms_version()}__{name_token}'
        filename = f'{base}__{split_part}.parquet.gz' if split_part else f'{base}.parquet.gz'

        os.makedirs(dir_full, exist_ok=True)
        return utils.create_path_from_dir_filename(dir_full, filename)
