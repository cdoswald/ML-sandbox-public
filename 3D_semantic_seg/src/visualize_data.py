"""Generate video and pointcloud visualizations for labeled data"""

interactive_mode = True
if interactive_mode:
    import os
    os.chdir("/workspace/hostfiles/src")
    print(os.getcwd())

import os                                   #noqa: E402
import h5py                                 #noqa: E402
import numpy as np                          #noqa: E402

from config import Config
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

    # Get config args
    args = Config()
    
    # Get list of file IDs that are present in all data subfolders
    file_ids = utl.get_ids_of_complete_data_files(args.data_dir)
    print(f"Total # of file IDs: {len(file_ids)}")

    # Load data for file ID
    for i, file_id in enumerate(file_ids):

        # Get list of observation ids that have 3D semantic segmentation labels (not all are labeled)
        lidar_segment_table_all = utl.load_parquet_data(args.data_dir, file_id, "lidar_segmentation")
        labeled_obs_ids = lidar_segment_table_all["index"].combine_chunks().to_pylist()
        print(f"# of labeled obs ids: {len(labeled_obs_ids)}")
        del lidar_segment_table_all

        # Extract labeled LiDAR data and corresponding camera images
        # ("index" = key.segment_context_name + key.frame_timestamp_micros)
        filter_lidar_rows = {"index":labeled_obs_ids, "key.laser_name":[1]}
        filter_camera_rows = {"index":labeled_obs_ids, "key.camera_name":list(np.arange(1, 6))}

        # Load tables that are small enough to load all observations in memory
        camera_box_table = utl.load_parquet_data(
            args.data_dir, file_id, "camera_box", filter_rows=filter_camera_rows
        )
        lidar_calib_table = utl.load_parquet_data(
            args.data_dir, file_id, "lidar_calibration", filter_rows=filter_lidar_rows
        )

        # Display camera images with object boxes
        camera_image_table_obs = utl.load_parquet_data(
            args.data_dir, file_id, "camera_image", filter_rows=filter_camera_rows
        )

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
            args.videos_dir,
            f"camera_{file_id}",
            fps=5.0,
        )
        
        del camera_image_table_obs, camera_frames

        # Get lidar range values for one observation at a time due to memory constraints
        # Use HDF5 for incremental writing to avoid memory exhaustion
        output_pointcloud_file = os.path.join(args.pointcloud_dir, f"pointcloud_{file_id}.h5")
        
        with h5py.File(output_pointcloud_file, 'w') as f:
            for labeled_obs_id in sorted(labeled_obs_ids):
                filter_lidar_rows["index"] = [labeled_obs_id]
                lidar_image_table = utl.load_parquet_data(
                    args.data_dir, file_id, "lidar", filter_rows=filter_lidar_rows
                )
                lidar_segment_table = utl.load_parquet_data(
                    args.data_dir, file_id, "lidar_segmentation", filter_rows=filter_lidar_rows
                )
                points = utl_li.convert_lidar_range_image_to_xyz_coords(
                    lidar_image_table, lidar_calib_table, lidar_segment_table, convert_to_world_ref=False
                )
                # Write immediately without accumulating in memory
                f.create_dataset(str(labeled_obs_id), data=points, compression='gzip', compression_opts=4)
        
        del lidar_image_table, lidar_segment_table, points
                
        # Visualize pointcloud with Open3D
        utl_o3d.visualize_pointcloud_headless(
            pointcloud_file=f"pointcloud_{file_id}.h5",
            pointcloud_dir=args.pointcloud_dir,
            videos_dir=args.videos_dir,
            fps=5
        )
        
        if i > 2:
            break
