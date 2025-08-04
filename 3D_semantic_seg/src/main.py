"""Main execution"""

interactive_mode = True
if interactive_mode:
    import os
    os.chdir("/workspace/hostfiles/src")
    print(os.getcwd())

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

# from waymo_open_dataset import dataset_pb2 as wod
# from waymo_open_dataset.utils import (
#     frame_utils,
#     range_image_utils,
# )

from models import PlaceholderModel

from utils import utils as utl
from utils import utils_camera as utl_cam
from utils import utils_constants as utl_cons
from utils import utils_lidar as utl_li
from utils import utils_open3d as utl_o3d

# from utils import utils_plotting as utl_p
# from utils import utils_waymo as utl_w


if __name__ == "__main__":

    # # Set up logging
    # logger = logging.getLogger(__name__)
    # logging.basicConfig(filename="main.log", encoding="utf-8", level=logging.DEBUG)
    # logger.debug("Logger set-up successful")

    # Define constants
    # LASER_NAME_MAP = dict(wod.LaserName.Name.items())
    # CAMERA_NAME_MAP = dict(wod.CameraName.Name.items())
    # LIDAR_RETURN_MAP = dict()
    # RANGE_IMAGE_DIM_MAP = utl_cons.get_range_image_final_dim_dict()
    # SEG_IMAGE_DIM_MAP = utl_cons.get_seg_image_final_dim_dict()

    DATA_DIR = "/workspace/hostfiles/data"
    VIDEOS_DIR = "/workspace/hostfiles/videos"
    POINTCLOUD_DIR = "/workspace/hostfiles/pointclouds"

    os.makedirs(VIDEOS_DIR, exist_ok=True)
    os.makedirs(POINTCLOUD_DIR, exist_ok=True)
    
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

        # Get column names of tables
        camera_image_cols = utl.get_parquet_col_names(DATA_DIR, file_id, "camera_image")
        camera_box_cols = utl.get_parquet_col_names(DATA_DIR, file_id, "camera_box")
        camera_calib_cols = utl.get_parquet_col_names(DATA_DIR, file_id, "camera_calibration")
        camera_segment_cols = utl.get_parquet_col_names(DATA_DIR, file_id, "camera_segmentation")

        lidar_image_cols = utl.get_parquet_col_names(DATA_DIR, file_id, "lidar")
        lidar_calib_cols = utl.get_parquet_col_names(DATA_DIR, file_id, "lidar_calibration")
        lidar_segment_cols = utl.get_parquet_col_names(DATA_DIR, file_id, "lidar_segmentation")

        # Get list of observation ids that have 3D semantic segmentation labels
        lidar_segment_table_all = utl.load_parquet_data(DATA_DIR, file_id, "lidar_segmentation")
        labeled_obs_ids = lidar_segment_table_all["index"].combine_chunks().to_pylist()
        print(f"# of labeled obs ids: {len(labeled_obs_ids)}")

        # Extract labeled LiDAR data and corresponding camera images
        filter_lidar_rows = {"index":labeled_obs_ids, "key.laser_name":[1]}
        filter_camera_rows = {"index":labeled_obs_ids, "key.camera_name":list(np.arange(1, 6))}

        camera_image_table = utl.load_parquet_data(
            DATA_DIR, file_id, "camera_image", filter_rows=filter_camera_rows
        )
        camera_box_table = utl.load_parquet_data(
            DATA_DIR, file_id, "camera_box", filter_rows=filter_camera_rows
        )
        camera_calib_table = utl.load_parquet_data(
            DATA_DIR, file_id, "camera_calibration", filter_rows=filter_camera_rows
        )
        camera_segment_table = utl.load_parquet_data(
            DATA_DIR, file_id, "camera_segmentation", filter_rows=filter_camera_rows
        )

        lidar_calib_table = utl.load_parquet_data(
            DATA_DIR, file_id, "lidar_calibration", filter_rows=filter_lidar_rows
        )

        # Display camera images with object boxes
        camera_image_table_obs = utl.filter_rows_equal(camera_image_table, filter_camera_rows)
    
        camera_frames = {}
        for camera_id in range(10):
            print(f"Trying camera id {camera_id}")
            try:
                camera_frames[camera_id] = utl_cam.extract_camera_images(
                    camera_image_table_obs,
                    camera_box_table,
                    camera_id=camera_id,
                )
            except ValueError as e:
                print(f"Could not extract frames for camera_id={camera_id}; got error: '{e}'")
        
        utl_cam.write_frames_to_video_file(
            camera_frames,
            VIDEOS_DIR,
            f"camera_{file_id}",
            fps=5.0,
        )

        # Get lidar range values for one observation at a time due to memory constraints
        all_points_dict = {}
        for labeled_obs_id in sorted(labeled_obs_ids):
            filter_lidar_rows["index"] = [labeled_obs_id]
            lidar_image_table = utl.load_parquet_data(
                DATA_DIR, file_id, "lidar", filter_rows=filter_lidar_rows
            )
            lidar_segment_table = utl.load_parquet_data(
                DATA_DIR, file_id, "lidar_segmentation", filter_rows=filter_lidar_rows
            )
            points = utl_li.convert_lidar_range_image_to_xyz_coords(
                lidar_image_table, lidar_calib_table, lidar_segment_table, convert_to_world_ref=False
            )
            all_points_dict[labeled_obs_id] = points

        np.savez(
            os.path.join(POINTCLOUD_DIR, f"pointcloud_{file_id}.npz"),
            **{k:v for k,v in all_points_dict.items()}
        )

        utl_o3d.visualize_pointcloud_headless(
            pointcloud_file=f"pointcloud_{file_id}.npz",
            pointcloud_dir=POINTCLOUD_DIR,
            videos_dir=VIDEOS_DIR,
            fps=5
        )

        break


        #TODO: resolve pylint errors (switch to Ruff?)
        #TODO: display pointcloud and camera videos side-by-side
        #TODO: modify to handle batch size > 1 obs
        #TODO: start developing ML model
    


# "index" = key.segment_context_name + key.frame_timestamp_micros






    


        








#     # Extract frames
#     frames = utl.extract_frames_from_datafile(dataset)
#     frame = frames[24]  # TODO: generalize

#     # Parse range images
#     range_images, camera_projections, seg_labels, range_image_top_pose = (
#         frame_utils.parse_range_image_and_camera_projection(frame)
#     )

#     # Convert range and segmentation images to tensors
#     ## range_images, camera_projections, and seg_labels are
#     ## dictionaries formatted: {laser_index: [return1, return2]}
#     range_image_tensor = utl.convert_range_image_to_tensor(
#         range_images[LASER_NAME_MAP["TOP"]][0]
#     )
#     cp_tensor = utl.convert_range_image_to_tensor(
#         camera_projections[LASER_NAME_MAP["TOP"]][0]
#     )
#     seg_image_tensor = utl.convert_range_image_to_tensor(
#         seg_labels[LASER_NAME_MAP["TOP"]][0]
#     )

#     # Plot example range and segmentation image
#     utl_p.plot_range_image_tensor(
#         range_image_tensor,
#         RANGE_IMAGE_DIM_MAP,
#         invert_colormap=True,
#     )
#     utl_p.plot_range_image_tensor(
#         seg_image_tensor, SEG_IMAGE_DIM_MAP, style_params={"cmap": "tab20"}
#     )

#     # Plot example point cloud
#     points, cp_points = frame_utils.convert_range_image_to_point_cloud(
#         frame,
#         range_images,
#         camera_projections,
#         range_image_top_pose,
#         ri_index=0,  # First return
#     )
#     point_labels = utl_w.convert_range_image_to_point_cloud_labels(
#         frame,
#         range_images,
#         seg_labels,
#         ri_index=0,  # First return
#     )
#     points_all = np.concatenate(points, axis=0)
#     point_labels_all = np.concatenate(point_labels, axis=0)
#     cp_points_all = np.concatenate(cp_points, axis=0)

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
    
    # # Placeholder model (getting pipeline set up before iterating on model development)
    # model = PlaceholderModel()
    # range_image_tensor = torch.from_numpy(range_image_tensor.numpy())
    # output = model(range_image_tensor)

    # # Generate protobuff submission file for validation set


    # range_image_tensor.shape
    # TOP_LIDAR_ROW_NUM = 64
    # TOP_LIDAR_COL_NUM = 2650
    
 