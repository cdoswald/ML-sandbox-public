"""Waymo-provided utility functions for Waymo Open Dataset challenges.

Note that all functions in this module are originally provided by Waymo
and used without modification. Functions that are based on Waymo utility 
functions but that have been modified will be stored in utils.py 
(with source info).
"""

import os
from typing import Dict, List, Optional, Tuple
import warnings
import zlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as wod

## Source: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_3d_semseg.ipynb
def convert_range_image_to_point_cloud_labels(
    frame, range_images, segmentation_labels, ri_index=0
):
    """Convert segmentation labels from range images to point clouds.

    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      segmentation_labels: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      ri_index: 0 for the first return, 1 for the second return.

    Returns:
      point_labels: {[N, 2]} list of 3d lidar points's segmentation labels. 0 for
        points that are not labeled.
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    point_labels = []
    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims
        )
        range_image_mask = range_image_tensor[..., 0] > 0

        if c.name in segmentation_labels:
            sl = segmentation_labels[c.name][ri_index]
            sl_tensor = tf.reshape(tf.convert_to_tensor(sl.data), sl.shape.dims)
            sl_points_tensor = tf.gather_nd(sl_tensor, tf.where(range_image_mask))
        else:
            num_valid_point = tf.math.reduce_sum(tf.cast(range_image_mask, tf.int32))
            sl_points_tensor = tf.zeros([num_valid_point, 2], dtype=tf.int32)

        point_labels.append(sl_points_tensor.numpy())
    return point_labels


## Source: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_3d_semseg.ipynb
def get_range_image_point_indexing(range_images, ri_index=0):
  """Get the indices of the valid points (of the TOP lidar) in the range image.

  The order of the points match those from convert_range_image_to_point_cloud
  and convert_range_image_to_point_cloud_labels.

  Args:
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    ri_index: 0 for the first return, 1 for the second return.

  Returns:
    points_indexing_top: (N, 2) col and row indices of the points in the
      TOP lidar.
  """
  points_indexing_top = None
  xgrid, ygrid = np.meshgrid(range(TOP_LIDAR_COL_NUM), range(TOP_LIDAR_ROW_NUM))
  col_row_inds_top = np.stack([xgrid, ygrid], axis=-1)
  range_image = range_images[open_dataset.LaserName.TOP][ri_index]
  range_image_tensor = tf.reshape(
      tf.convert_to_tensor(range_image.data), range_image.shape.dims)
  range_image_mask = range_image_tensor[..., 0] > 0
  points_indexing_top = col_row_inds_top[np.where(range_image_mask)]
  return points_indexing_top


## Source: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_3d_semseg.ipynb
def compress_array(array: np.ndarray, is_int32: bool = False):
  """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

  Args:
    array: A numpy array.
    is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

  Returns:
    The compressed bytes.
  """
  if is_int32:
    m = open_dataset.MatrixInt32()
  else:
    m = open_dataset.MatrixFloat()
  m.shape.dims.extend(list(array.shape))
  m.data.extend(array.reshape([-1]).tolist())
  return zlib.compress(m.SerializeToString())


## Source: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_3d_semseg.ipynb
def decompress_array(array_compressed: bytes, is_int32: bool = False):
  """Decompress bytes (of serialized MatrixFloat/Int32) to a numpy array.

  Args:
    array_compressed: bytes.
    is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

  Returns:
    The decompressed numpy array.
  """
  decompressed = zlib.decompress(array_compressed)
  if is_int32:
    m = open_dataset.MatrixInt32()
    dtype = np.int32
  else:
    m = open_dataset.MatrixFloat()
    dtype = np.float32
  m.ParseFromString(decompressed)
  return np.array(m.data, dtype=dtype).reshape(m.shape.dims)