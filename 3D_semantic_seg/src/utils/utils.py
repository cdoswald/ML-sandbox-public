"""General utility functions for Waymo Open Dataset challenges."""

import os
from typing import Dict, List, Optional, Union, Tuple
import warnings

import matplotlib.pyplot as plt
import pyarrow
import pyarrow.compute as pc
import tensorflow
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as wod


def filter_table(
    table: pyarrow.lib.Table,
    filter_col: str,
    filter_val: Union[str, int],
) -> pyarrow.lib.Table:
    """Filter pyarrow table where 'filter_col' matches 'filter_val'."""
    mask = pc.equal(table[filter_col], filter_val)
    return table.filter(mask)

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

def convert_range_image_to_tensor(
    range_image: wod.MatrixFloat,
) -> tf.Tensor:
    """Convert range image from protocol buffer MatrixFloat object
    to Tensorflow tensor object.

    Based on https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb.
    """
    return tf.reshape(
        tf.convert_to_tensor(range_image.data),
        range_image.shape.dims
    )
