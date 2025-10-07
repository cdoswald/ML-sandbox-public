"""Main execution"""

interactive_mode = True
if interactive_mode:
    import os
    os.chdir("/workspace/hostfiles/src")
    print(os.getcwd())

import os                                   #noqa: E402
import numpy as np                          #noqa: E402

from utils import utils as utl              #noqa: E402
from utils import utils_camera as utl_cam   #noqa: E402
from utils import utils_lidar as utl_li     #noqa: E402
from utils import utils_open3d as utl_o3d   #noqa: E402

# from utils import utils_plotting as utl_p
# from utils import utils_waymo as utl_w

# from waymo_open_dataset import dataset_pb2 as wod
# from waymo_open_dataset.utils import (
#     frame_utils,
#     range_image_utils,
# )

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

# "index" = key.segment_context_name + key.frame_timestamp_micros
