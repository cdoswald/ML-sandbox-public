"""Camera utility functions for Waymo Open Dataset challenges."""

import io
import os
from typing import Dict, List, Optional, Tuple
import warnings

import cv2
import numpy as np
import pyarrow

from utils import utils as utl


def extract_camera_images(
    camera_image_table: pyarrow.lib.Table,
    camera_box_table: Optional[pyarrow.lib.Table] = None,
    camera_id: int = 1,
) -> List[np.ndarray]:
    """Extract camera image(s) from pyarrow table and convert from bytes to
    numpy array(s). If camera_box_table provided, will also include object
    bounding boxes.

    Args
        camera_image_table: camera_image pyarrow table
        camera_box_table: camera_box pyarrow table (optional; default = None)
        camera_id: camera name string ID (default = 1 (front camera))

    Returns
        frames: list of image frames stored as numpy ndarrays
    """
    camera_image_table = utl.filter_rows_equal(camera_image_table, {"key.camera_name":camera_id})
    if len(camera_image_table) < 1:
        raise ValueError(
            f"Camera image table does not contain any observations with camera_id={camera_id}"
        )
    # Sort by timestamp
    camera_image_table = camera_image_table.sort_by([("index", "ascending")])
    frames = []
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
        frames.append(obs_camera_image)
    return frames
