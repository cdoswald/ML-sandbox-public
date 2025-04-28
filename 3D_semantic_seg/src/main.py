"""Main execution"""

import io
import logging
import os
import numpy as np
from PIL import Image, ImageDraw
from typing import Optional, Union

import pyarrow
import pyarrow.parquet as pq
import pyarrow.compute as pc
import polars as pl
import torch

from waymo_open_dataset import dataset_pb2 as wod
from waymo_open_dataset.utils import frame_utils

from models import PlaceholderModel

from utils import utils as utl
from utils import utils_constants as utl_c
from utils import utils_plotting as utl_p
from utils import utils_waymo as utl_w


if __name__ == "__main__":

    # # Set up logging
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(filename="main.log", encoding="utf-8", level=logging.DEBUG)
    # logger.debug("Logger set-up successful")

    # Define constants
    LASER_NAME_MAP = dict(wod.LaserName.Name.items())
    CAMERA_NAME_MAP = dict(wod.CameraName.Name.items())
    LIDAR_RETURN_MAP = dict()
    RANGE_IMAGE_DIM_MAP = utl_c.get_range_image_final_dim_dict()
    SEG_IMAGE_DIM_MAP = utl_c.get_seg_image_final_dim_dict()

    DATA_DIR = "/workspace/hostfiles/3DSemanticSeg/data"
    
    # Get list of file IDs that are present in all data subfolders
    file_ids = []
    camera_image_files = os.listdir(os.path.join(DATA_DIR, "camera_image"))
    for camera_image_file in camera_image_files:
        file_ids.append(camera_image_file.split(".")[0])
    for data_subdir in os.listdir(DATA_DIR):
        for file_id in file_ids:
            file_id_missing = True
            for file in os.listdir(os.path.join(DATA_DIR, data_subdir)):
                if file_id == file.split(".")[0]:
                    file_id_missing = False
                    break
            if file_id_missing:
                file_ids.remove(file_id)
                print(f"Removed file ID {file_id} (missing in {data_subdir})")
    print(f"Total # of file IDs: {len(file_ids)}")

    # Load data for file ID
    for file_id in file_ids:
        file_id = file_ids[0] #TODO: remove

        camera_image_table = pq.read_table(
            os.path.join(DATA_DIR, "camera_image", f"{file_id}.parquet"),
        )
        camera_box_table = pq.read_table(
            os.path.join(DATA_DIR, "camera_box", f"{file_id}.parquet")
        )
        camera_calib_table = pq.read_table(
            os.path.join(DATA_DIR, "camera_calibration", f"{file_id}.parquet")
        )
        camera_segment_table = pq.read_table(
            os.path.join(DATA_DIR, "camera_segmentation", f"{file_id}.parquet")
        )

        # Get list of observation IDs in camera image file
        obs_ids = [str(x) for x in np.unique(camera_image_table["index"].to_pylist())]

        # Filter data for observation ID
        for obs_id in obs_ids:
            obs_id = obs_ids[0] #TODO: remove
            break #TODO: remove
    
            # Display camera images with object boxes
            camera_ids = utl.filter_table(camera_image_table, "index", obs_id).column(
                "key.camera_name"
            )
            for camera_id in camera_ids:
                utl_p.show_camera_image_with_bboxes(
                    camera_image_table,
                    camera_box_table,
                    obs_id,
                    camera_id=camera_id,
                )
        
            DEL = utl.filter_table(camera_segment_table, "index", obs_id)
            DEL = utl.filter_table(DEL, "key.camera_name", 1)
        





# Camera images
['index',
 'key.segment_context_name',
 'key.frame_timestamp_micros',
 'key.camera_name',
 '[CameraImageComponent].image',
 '[CameraImageComponent].pose.transform', # Transform from camera ref to vehicle ref system
 '[CameraImageComponent].velocity.linear_velocity.x',
 '[CameraImageComponent].velocity.linear_velocity.y',
 '[CameraImageComponent].velocity.linear_velocity.z',
 '[CameraImageComponent].velocity.angular_velocity.x',
 '[CameraImageComponent].velocity.angular_velocity.y',
 '[CameraImageComponent].velocity.angular_velocity.z',
 '[CameraImageComponent].pose_timestamp',
 '[CameraImageComponent].rolling_shutter_params.shutter',
 '[CameraImageComponent].rolling_shutter_params.camera_trigger_time',
 '[CameraImageComponent].rolling_shutter_params.camera_readout_done_time']

# Camera boxes
['index',
 'key.segment_context_name',
 'key.frame_timestamp_micros',
 'key.camera_name',
 'key.camera_object_id',
 '[CameraBoxComponent].box.center.x',
 '[CameraBoxComponent].box.center.y',
 '[CameraBoxComponent].box.size.x',
 '[CameraBoxComponent].box.size.y',
 '[CameraBoxComponent].type',
 '[CameraBoxComponent].difficulty_level.detection',
 '[CameraBoxComponent].difficulty_level.tracking']

# Camera calibration
['key.segment_context_name',
 'key.camera_name',
 '[CameraCalibrationComponent].intrinsic.f_u',
 '[CameraCalibrationComponent].intrinsic.f_v',
 '[CameraCalibrationComponent].intrinsic.c_u',
 '[CameraCalibrationComponent].intrinsic.c_v',
 '[CameraCalibrationComponent].intrinsic.k1',
 '[CameraCalibrationComponent].intrinsic.k2',
 '[CameraCalibrationComponent].intrinsic.p1',
 '[CameraCalibrationComponent].intrinsic.p2',
 '[CameraCalibrationComponent].intrinsic.k3',
 '[CameraCalibrationComponent].extrinsic.transform',
 '[CameraCalibrationComponent].width',
 '[CameraCalibrationComponent].height',
 '[CameraCalibrationComponent].rolling_shutter_direction']
    
    # Extract frames
    frames = utl.extract_frames_from_datafile(dataset)
    frame = frames[24]  # TODO: generalize

    # Parse range images
    range_images, camera_projections, seg_labels, range_image_top_pose = (
        frame_utils.parse_range_image_and_camera_projection(frame)
    )

    # Convert range and segmentation images to tensors
    ## range_images, camera_projections, and seg_labels are
    ## dictionaries formatted: {laser_index: [return1, return2]}
    range_image_tensor = utl.convert_range_image_to_tensor(
        range_images[LASER_NAME_MAP["TOP"]][0]
    )
    cp_tensor = utl.convert_range_image_to_tensor(
        camera_projections[LASER_NAME_MAP["TOP"]][0]
    )
    seg_image_tensor = utl.convert_range_image_to_tensor(
        seg_labels[LASER_NAME_MAP["TOP"]][0]
    )

    # Plot example range and segmentation image
    utl_p.plot_range_image_tensor(
        range_image_tensor,
        RANGE_IMAGE_DIM_MAP,
        invert_colormap=True,
    )
    utl_p.plot_range_image_tensor(
        seg_image_tensor, SEG_IMAGE_DIM_MAP, style_params={"cmap": "tab20"}
    )

    # Plot example point cloud
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose,
        ri_index=0,  # First return
    )
    point_labels = utl_w.convert_range_image_to_point_cloud_labels(
        frame,
        range_images,
        seg_labels,
        ri_index=0,  # First return
    )
    points_all = np.concatenate(points, axis=0)
    point_labels_all = np.concatenate(point_labels, axis=0)
    cp_points_all = np.concatenate(cp_points, axis=0)

    # TODO: plot point cloud
    
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(points_all[:, 0], points_all[:, 1], points_all[:, 2])

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(cp_points_all[:, 0], cp_points_all[:, 1], cp_points_all[:, 2])
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(cp_points_all[:, 3], cp_points_all[:, 4], cp_points_all[:, 5])
    
    # Placeholder model (getting pipeline set up before iterating on model development)
    model = PlaceholderModel()
    range_image_tensor = torch.from_numpy(range_image_tensor.numpy())
    output = model(range_image_tensor)

    # Generate protobuff submission file for validation set


    range_image_tensor.shape
    TOP_LIDAR_ROW_NUM = 64
    TOP_LIDAR_COL_NUM = 2650
    
 