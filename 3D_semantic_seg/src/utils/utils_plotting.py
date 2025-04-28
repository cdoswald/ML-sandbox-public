"""Plotting utility functions for Waymo Open Dataset challenges."""

import io
import os
from PIL import Image, ImageDraw
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import pyarrow
import tensorflow
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as wod

from utils import utils as utl
from utils import utils_constants as utl_c
from utils import utils_waymo as utl_w


def show_camera_image_with_bboxes(
    camera_image_table: pyarrow.lib.Table,
    camera_box_table: pyarrow.lib.Table,
    obs_id: str,
    camera_id: int,
) -> None:
    """Display camera image with object bounding boxes.
    
    Args
        camera_image_table: camera_image pyarrow table
        camera_box_table: camera_box pyarrow table
        obs_id: unique observation string ID
        camera_id: camera name string ID
    """
    # Camera image
    obs_camera_image = utl.filter_table(camera_image_table, "index", obs_id)
    obs_camera_image = utl.filter_table(obs_camera_image, "key.camera_name", camera_id)
    assert(len(obs_camera_image) == 1)
    obs_camera_image_bytes = obs_camera_image["[CameraImageComponent].image"][0].as_py()
    obs_camera_image = Image.open(io.BytesIO(obs_camera_image_bytes))
    draw = ImageDraw.Draw(obs_camera_image)
    # Object bounding boxes
    obs_camera_boxes = utl.filter_table(camera_box_table, "index", obs_id)
    obs_camera_boxes = utl.filter_table(obs_camera_boxes, "key.camera_name", camera_id)
    for i in range(len(obs_camera_boxes)):
        center_x = obs_camera_boxes["[CameraBoxComponent].box.center.x"][i].as_py()
        center_y = obs_camera_boxes["[CameraBoxComponent].box.center.y"][i].as_py()
        size_x = obs_camera_boxes["[CameraBoxComponent].box.size.x"][i].as_py()
        size_y = obs_camera_boxes["[CameraBoxComponent].box.size.y"][i].as_py()
        bbox_coords = [
            center_x - size_x/2,
            center_y - size_y/2,
            center_x + size_x/2,
            center_y + size_y/2,
        ]
        draw.rectangle(bbox_coords, outline="red", width=4)
    obs_camera_image.show()


def plot_range_image_tensor(
    range_image: tf.Tensor,
    dim_map: Dict[int, str],
    invert_colormap: bool = False,
    style_params: Optional[Dict] = None,
) -> None:
    """Plot tensor-formatted range image.

    Args
        range_image: range image formatted as tf.Tensor
        dim_map: dict mapping last dimension index of tensor to corresponding name
        invert_colormap: invert pixel intensities (light becomes dark and vice versa)
        style_params: dict mapping style param name to values

    Returns
        None
    """
    # Specify default style params
    config = {
        "figsize": (12, 8),
        "gridspec_kw": {"hspace": 0.3},
        "fontsize": 20,
        "pad_amt": 10,
        "subtitle_loc": "left",
        "cmap": "gray",
    }

    # Update style params
    if style_params is not None:
        for key, value in style_params.items():
            if key not in config:
                warnings.warn(f'Style param "{key}" is not currently supported')
            else:
                config[key] = style_params[key]

    # Invert pixel intensities
    if invert_colormap:
        range_image = tf.where(
            tf.greater_equal(range_image, 0),
            range_image,
            tf.ones_like(range_image) * 1e10,
        )

    # Plot distance, intensity, and elongation
    fig, axes = plt.subplots(
        nrows=len(dim_map),
        figsize=config["figsize"],
        gridspec_kw=config["gridspec_kw"],
    )
    for idx, axes_name in dim_map.items():
        axes[idx].imshow(range_image[..., idx], cmap=config["cmap"], aspect="auto")
        axes[idx].set_title(
            axes_name,
            fontsize=config["fontsize"],
            pad=config["pad_amt"],
            loc=config["subtitle_loc"],
        )
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])
