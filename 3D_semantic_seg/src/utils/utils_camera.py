"""Camera utility functions for Waymo Open Dataset challenges."""

from itertools import chain
import io
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyarrow

from utils import utils as util
from utils import utils_constants as util_cons


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
    camera_image_table = util.filter_rows_equal(camera_image_table, {"key.camera_name":[camera_id]})
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
            obs_camera_boxes = util.filter_rows_equal(
                camera_box_table, {"index":obs_id, "key.camera_name":[camera_id]}
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


def display_frames(
    frames: Union[Sequence[np.ndarray], Dict[int, Sequence[np.ndarray]]],
    orient_veh_view: bool = True,
) -> None:
    """Display camera image frame(s). For single-camera display, 'frames'
    should be a sequence of numpy ndarrays. For multicamera display,
    'frames' should be a dictionary mapping camera ID to sequence of 
    numpy ndarrays.

    If multicamera display, then images will be automatically reordered
    to match the viewpoint of the vehicle (e.g., left side camera will be
    the furthest left image, front camera will be the middle image, etc.).
    To display the images according to the original order of camera IDs 
    in the 'frames' dictionary, set 'orient_veh_view' arg to False.

    Args:
        frames: for single camera, sequence of images stored as numpy 
            ndarray(s); for multicamera, dict mapping camera ID to sequence
            of images stored as numpy ndarray(s)
        orient_veh_view: bool indicator for whether to automatically display
            images from vehicle perspective (default = True); only applies
            if frames is dict for multicamera display

    Returns:
        None
    """
    # Multicamera display
    if isinstance(frames, dict):
        camera_ids = list(frames.keys())
        if orient_veh_view:
            camera_ids = orient_camera_ids(camera_ids)
        n_cameras = len(camera_ids)
        n_frames = max([len(x) for x in frames.values()])
        for i in range(n_frames):
            fig, axes = plt.subplots(1, n_cameras, figsize=(8,6))
            fig.subplots_adjust(wspace=0.03, hspace=0)
            for j, camera_id in enumerate(camera_ids):
                try:
                    frame = frames[camera_id][i]
                    axes[j].imshow(frame)
                finally:
                    axes[j].axis("off")
            plt.show()
            plt.close()
    # Single-camera display
    else:
        for frame in frames:
            plt.imshow(frame)
            plt.axis("off")
            plt.show()


def orient_camera_ids(camera_ids: List[int]) -> List[int]:
    """Reorder list of camera IDs to reflect vehicle viewpoint.

    Args:
        camera_ids: list of camera IDs containing at least one camera frame

    Returns:
        ordered_camera_ids: reordered list of camera IDs
    """
    ordered_camera_ids = []
    camera_name_idx_map = {v:k for k,v in util_cons.get_camera_idx_map().items()}
    camera_name_order = util_cons.get_veh_view_camera_name_order()
    for camera_name in camera_name_order:
        camera_idx = camera_name_idx_map.get(camera_name)
        if camera_idx is not None and camera_idx in camera_ids:
            ordered_camera_ids.append(camera_idx)
    return ordered_camera_ids


def write_frames_to_video_file(
    frames: Union[List[np.ndarray], Dict[int, List[np.ndarray]]],
    dir_name: str,
    file_name: str,
    fps: float = 10.0,
) -> None:
    """Write image frame(s) stored as numpy ndarray(s) to mp4 video file."""
    frames = [frames] if not isinstance(frames, list) else frames
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(dir_name, f"{file_name}.mp4")
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    try:
        for idx, frame in enumerate(frames):
            height_i, width_i, _ = frame.shape
            if (height_i != height or width_i != width):
                raise ValueError(
                    f"Height and width of frame {idx} does not match frame 0"
                )
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame)
    finally:
        video.release()
