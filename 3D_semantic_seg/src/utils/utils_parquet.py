"""Utility functions for Parquet data format"""

from functools import reduce
import operator
from typing import Dict, List, Optional, Union
import warnings

import os
import pyarrow
import pyarrow.compute as pc
import pyarrow.dataset as pds


def _join_path(base: str, *parts: str) -> str:
    """Join path parts, handling both local and GCS (gs://) paths."""
    if base.startswith("gs://"):
        return "/".join([base.rstrip("/")] + [p.strip("/") for p in parts if p])
    return os.path.join(base, *parts)

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
    file_path = _join_path(data_dir, data_subdir, f"{file_id}.parquet")
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
    file_path = _join_path(data_dir, data_subdir, f"{file_id}.parquet")
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
                "User provided filter_rows arg, but no filter expressions could "+
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
        condition_match = pc.is_in(
            table[col_name], pyarrow.array(list_filter_vals)
        )
        mask = pc.and_kleene(mask, condition_match)
    return table.filter(mask)