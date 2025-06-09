"""General utility functions for Waymo Open Dataset challenges."""

from functools import reduce
import operator
import os
from typing import Dict, List, Optional, Union, Tuple
import warnings

import matplotlib.pyplot as plt
import pyarrow
import pyarrow.compute as pc
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import tensorflow
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as wod

## ------------------------------------------
## Utility functions for Parquet data format
## ------------------------------------------
def get_parquet_col_names(
    data_dir: str,
    file_id: str,
    data_subdir: Optional[str] = "",
) -> List[str]:
    """Get column names of parquet file without loading into memory.

    Args
        data_dir: data directory name
        file_id: file name (excluding .parquet suffix)
        data_subdir: optional data subdirectory name
    """
    if not isinstance(data_subdir, str):
        raise ValueError(
            f"data_subdir must be empty or non-empty str, but got type {type(data_subdir)}"
        )
    file_path = os.path.join(data_dir, data_subdir, f"{file_id}.parquet")
    return pds.dataset(file_path).schema.names

def load_parquet_data(
    data_dir: str,
    file_id: str,
    data_subdir: Optional[str] = "",
    subset_cols: Optional[List[str]] = None,
    filter_rows: Optional[Dict[str, List[Union[str, int, float]]]] = None,
    scanner_batch_size: int = 128,
) -> pyarrow.Table:
    """Load parquet dataset.

    Args:
        data_dir: data directory name
        file_id: file name (excluding .parquet suffix)
        data_subdir: optional data subdirectory name
        subset_cols: optional list of column names to retain
        filter_rows: optional dict of {column_name_1:[filter_val_1, ...], ...}
        scanner_batch_size: batch size to use for pyarrow.dataset scanner (default=128)

    Returns:
        pyarrow Table with filtered rows and columns (if applicable)
    """
    if not isinstance(data_subdir, str):
        raise ValueError(
            f"data_subdir must be empty or non-empty str, but got type {type(data_subdir)}"
        )
    file_path = os.path.join(data_dir, data_subdir, f"{file_id}.parquet")
    data = pds.dataset(file_path, format="parquet")
    # Filter rows
    row_filter = None
    if filter_rows is not None:
        filter_exprs = []
        for col_name, list_filter_vals in filter_rows.items():
            if not isinstance(list_filter_vals, list):
                raise ValueError(
                    f"All values in filter_rows dict must be lists; got {type(list_filter_vals)}"
                )
            if col_name in data.schema.names:
                if len(list_filter_vals) == 1:
                    exprs = (pds.field(col_name) == list_filter_vals[0])
                else:
                    exprs = pds.field(col_name).isin(list_filter_vals)
                filter_exprs.append(exprs)
            else:
                warnings.warn(
                    f"Column '{col_name}' from filter_rows dict not found in schema "+
                    f"(filepath = {file_path})"
                )
        # Take intersection of all filter expressions
        if len(filter_exprs) > 1:
            row_filter = reduce(operator.and_, filter_exprs)
        elif filter_exprs:
            row_filter = filter_exprs[0]
        else:
            warnings.warn(
                f"User provided filter_rows arg, but no filter expressions could "+
                "be generated. Check that column names in filter_rows match data schema."
            )
    # Filter columns
    if subset_cols is None:
        subset_cols = data.schema.names
    else:
        for col_name in subset_cols:
            if col_name not in data.schema.names:
                warnings.warn(
                    f"Column '{col_name}' from subset_cols list not found in schema "+
                    f"(filepath = {file_path})"
                )
        subset_cols = [col for col in subset_cols if col in data.schema.names]
    # Scan data in batches to avoid OOM error
    scanner = data.scanner(
        filter=row_filter,
        columns=subset_cols,
        batch_size=scanner_batch_size,
    )
    matching_batches = []
    for batch in scanner.to_batches():
        if len(batch) > 0:
            matching_batches.append(batch)
    if matching_batches:
        return pyarrow.Table.from_batches(matching_batches)
    else:
        return pyarrow.Table.from_batches([], schema=data.schema)


def load_parquet_data2(
    data_dir: str,
    file_id: str,
    data_subdir: Optional[str] = "",
    subset_cols: Optional[List[str]] = None,
    filter_rows: Optional[Dict[str, List[Union[str, int, float]]]] = None,
) -> pyarrow.Table:
    """Load parquet dataset.

    Args:
        data_dir: data directory name
        file_id: file name (excluding .parquet suffix)
        data_subdir: optional data subdirectory name
        subset_cols: optional list of column names to retain
        filter_rows: optional dict of {column_name_1:[filter_val_1, ...], ...}

    Returns:
        pyarrow Table with filtered rows and columns (if applicable)
    """
    if not isinstance(data_subdir, str):
        raise ValueError(
            f"data_subdir must be empty or non-empty str, but got type {type(data_subdir)}"
        )
    file_path = os.path.join(data_dir, data_subdir, f"{file_id}.parquet")
    data = pds.dataset(file_path, format="parquet")
    # Filter rows
    row_filter = None
    if filter_rows is not None:
        filter_exprs = []
        for col_name, filter_vals in filter_rows.items():
            exprs = (pds.field(col_name) == filter_vals)
            filter_exprs.append(exprs)
        # Take intersection of all filter expressions
        row_filter = reduce(operator.and_, filter_exprs)
    # Filter columns
    if subset_cols is None:
        subset_cols = data.schema.names
    else:
        for col_name in subset_cols:
            if col_name not in data.schema.names:
                warnings.warn(
                    f"Column '{col_name}' from subset_cols list not found in schema "+
                    f"(filepath = {file_path})"
                )
        subset_cols = [col for col in subset_cols if col in data.schema.names]
    print("Loading table")
    
    return data.to_table(filter=row_filter, columns=subset_cols)


def filter_rows_equal(
    table: pyarrow.Table,
    filter_dict: Dict[str, List[Union[str, int, float]]],
) -> pyarrow.Table:
    """Filter rows of pyarrow table using equality matching.

    Args:
        table: pyarrow table to filter
        filter_dict: dict of {column_name_1:[filter_val_1, ...], ...}

    Returns:
        filtered pyarrow table
    """
    mask = pyarrow.array([True] * len(table))
    for col_name, list_filter_vals in filter_dict.items():
        condition_match = pc.is_in(table[col_name], list_filter_vals)
        mask = pc.and_kleene(mask, condition_match)
    return table.filter(mask)

## ------------------------------------------
## Utility functions for TFRecord data format
## ------------------------------------------
def load_TFRecord(
    datadir: str,
    filename: str,
) -> tf.data.Dataset:
    """Load TFRecord dataset."""
    return tf.data.TFRecordDataset(
        os.path.join(datadir, filename),
        compression_type="",
    )

def extract_frames_from_TFRecord(
    dataset: tf.data.Dataset,
    max_n_frames: Optional[int] = None,
) -> List[wod.Frame]:
    """Extract frames (sequences) from TFRecord dataset."""
    # Validate max_n_frames arg
    if max_n_frames is not None:
        if (max_n_frames <= 0) or (not isinstance(max_n_frames, int)):
            raise ValueError(
                f"max_n_frames argument ({max_n_frames}) must be positive integer"
            )
    # Extract frames
    frames = []
    for data in dataset:
        frame = wod.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frames.append(frame)
        if (max_n_frames is not None) and (len(frames) >= max_n_frames):
            break
    return frames
