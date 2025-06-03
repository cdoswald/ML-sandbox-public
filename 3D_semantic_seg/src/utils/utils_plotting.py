"""Image plotting utility functions for Waymo Open Dataset challenges."""

import io
import os
from typing import Dict, List, Optional, Tuple
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyarrow
import tensorflow
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()

from waymo_open_dataset import dataset_pb2 as wod

from utils import utils as utl
from utils import utils_constants as utl_c
from utils import utils_waymo as utl_w


def display_camera_images(
    camera_image_table: pyarrow.lib.Table,
    camera_box_table: Optional[pyarrow.lib.Table] = None,
    camera_id: int = 1,
    return_frames: bool = True,
) -> Optional[List]:
    """Display camera image(s) with optional object bounding boxes.
    
    Args
        camera_image_table: camera_image pyarrow table
        camera_box_table: camera_box pyarrow table (optional; default = None)
        camera_id: camera name string ID (default = 1 (front camera))
        return_frames: 
    """
    camera_image_table = utl.filter_rows_equal(camera_image_table, {"key.camera_name":camera_id})
    if len(camera_image_table) < 1:
        raise ValueError(
            f"Camera image table does not contain any observations with camera_id={camera_id}"
        )
    # Sort by timestamp
    camera_image_table = camera_image_table.sort_by([("index", "ascending")])
    frames = [] if return_frames else None
    for i in range(len(camera_image_table)):
        obs_id = camera_image_table["index"][i].as_py()
        # Convert camera image from bytes to numpy array
        obs_camera_image_bytes = camera_image_table["[CameraImageComponent].image"][i].as_py()
        obs_camera_image = cv2.imdecode(
            np.frombuffer(obs_camera_image_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        obs_camera_image = cv2.cvtColor(obs_camera_image, cv2.COLOR_BGR2RGB)
        # Draw object bounding boxes (if applicable)
        if camera_box_table is not None:
            obs_camera_boxes = utl.filter_rows_equal(
                camera_box_table, {"index":obs_id, "key.camera_name":camera_id}
            )
            for j in range(len(obs_camera_boxes)):
                center_x = obs_camera_boxes["[CameraBoxComponent].box.center.x"][j].as_py()
                center_y = obs_camera_boxes["[CameraBoxComponent].box.center.y"][j].as_py()
                size_x = obs_camera_boxes["[CameraBoxComponent].box.size.x"][j].as_py()
                size_y = obs_camera_boxes["[CameraBoxComponent].box.size.y"][j].as_py()
                x1 = int(center_x - size_x/2)
                y1 = int(center_y - size_y/2)
                x2 = int(center_x + size_x/2)
                y2 = int(center_y + size_y/2)
                cv2.rectangle(
                    obs_camera_image, (x1, y1), (x2, y2),
                    color=(0, 0, 255), thickness=2
                )
        # Append to frames list
        if return_frames:
            frames.append(obs_camera_image)
        # Display image
        plt.imshow(obs_camera_image)
        plt.axis("off")
        plt.show()
    # Return all frames (if applicable)
    return frames


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
