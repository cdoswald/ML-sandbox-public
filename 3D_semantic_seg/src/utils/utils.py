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
    filter_rows: Optional[Dict[str, Union[str, int, float]]] = None,
) -> pyarrow.Table:
    """Load parquet dataset.

    Args
        data_dir: data directory name
        file_id: file name (excluding .parquet suffix)
        data_subdir: optional data subdirectory name
        subset_cols: optional list of column names to retain
        filter_rows: optional dict of {col_name:filter_val} to filter rows

    Returns
        pyarrow Table with filtered rows and columns (if applicable)
    """
    if not isinstance(data_subdir, str):
        raise ValueError(
            f"data_subdir must be empty or non-empty str, but got type {type(data_subdir)}"
        )
    file_path = os.path.join(data_dir, data_subdir, f"{file_id}.parquet")
    data = pds.dataset(file_path, format="parquet")
    # Filter rows and columns
    row_filter = None
    if filter_rows is not None:
        filter_exprs = [(pds.field(col) == val) for col, val in filter_rows.items()]
        row_filter = reduce(operator.and_, filter_exprs)
    if subset_cols is None:
        subset_cols = data.schema.names
    return data.to_table(filter=row_filter, columns=subset_cols)

def filter_rows_equal(
    table: pyarrow.lib.Table,
    filter_dict: Dict[str, Union[str, int, float]],
) -> pyarrow.lib.Table:
    """Filter rows of pyarrow table using equality matching.

    Args
        table: pyarrow table to filter
        filter_dict: mapping of {col_name:filter_val}

    Returns
        filtered pyarrow table
    """
    mask = pyarrow.array([True] * len(table))
    for col_name, filter_val in filter_dict.items():
        condition_match = pc.equal(table[col_name], filter_val)
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
