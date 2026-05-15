"""Generate video and pointcloud visualizations for labeled data"""

import multiprocessing as mp
import os                                           #noqa: E402
import h5py                                         #noqa: E402
import numpy as np                                  #noqa: E402

from config import Config
from utils import utils as utl                      #noqa: E402
from utils import utils_camera as utl_cam           #noqa: E402
from utils import utils_gcp as utl_gcp              #noqa: E402
from utils import utils_lidar as utl_li             #noqa: E402
from utils import utils_open3d as utl_o3d           #noqa: E402
from utils import utils_parquet as utl_prq          #noqa: E402


def process_scene(file_id, args_dict):

    # Convert args_dict back to dataclass
    class DummyArgs:
        pass
    args = DummyArgs()
    args.__dict__.update(args_dict)

    # Get list of observation ids that have 3D semantic segmentation labels (not all are labeled)
    lidar_segment_table_all = utl_prq.load_parquet_data(args.data_dir, file_id, "lidar_segmentation")
    labeled_obs_ids = lidar_segment_table_all["index"].combine_chunks().to_pylist()
    print(f"[PID {os.getpid()}] {file_id}: # of labeled obs ids: {len(labeled_obs_ids)}")
    del lidar_segment_table_all

    # Extract labeled LiDAR data and corresponding camera images
    filter_lidar_rows = {"index":labeled_obs_ids, "key.laser_name":[1]}
    filter_camera_rows = {"index":labeled_obs_ids, "key.camera_name":list(np.arange(1, 6))}

    camera_box_table = utl_prq.load_parquet_data(
        args.data_dir, file_id, "camera_box", filter_rows=filter_camera_rows
    )
    lidar_calib_table = utl_prq.load_parquet_data(
        args.data_dir, file_id, "lidar_calibration", filter_rows=filter_lidar_rows
    )

    camera_image_table_obs = utl_prq.load_parquet_data(
        args.data_dir, file_id, "camera_image", filter_rows=filter_camera_rows
    )

    camera_frames = {}
    for camera_id in range(10):
        print(f"[PID {os.getpid()}] {file_id}: Trying camera id {camera_id}")
        try:
            camera_frames[camera_id] = utl_cam.extract_camera_images(
                camera_image_table_obs,
                camera_box_table,
                camera_id=camera_id,
            )
        except ValueError as e:
            print(f"[PID {os.getpid()}] {file_id}: Could not extract frames for camera_id={camera_id}; got error: '{e}'")

    # Create camera video file
    utl_cam.write_frames_to_video_file(
        camera_frames,
        args.videos_dir,
        f"camera_{file_id}",
        fps=args.scene_vis_fps,
    )

    del camera_image_table_obs, camera_frames

    # Get lidar range values for one observation at a time due to memory constraints
    output_pointcloud_file = os.path.join(args.pointcloud_dir, f"pointcloud_{file_id}.h5")
    with h5py.File(output_pointcloud_file, 'w') as f:
        for labeled_obs_id in sorted(labeled_obs_ids):
            filter_lidar_rows["index"] = [labeled_obs_id]
            lidar_image_table = utl_prq.load_parquet_data(
                args.data_dir, file_id, "lidar", filter_rows=filter_lidar_rows
            )
            lidar_segment_table = utl_prq.load_parquet_data(
                args.data_dir, file_id, "lidar_segmentation", filter_rows=filter_lidar_rows
            )
            points = utl_li.convert_lidar_range_image_to_xyz_coords(
                lidar_image_table,
                lidar_calib_table,
                lidar_segment_table,
                convert_to_world_ref=False
            )
            f.create_dataset(
                str(labeled_obs_id),
                data=points,
                compression='gzip',
                compression_opts=4
            )

    del lidar_image_table, lidar_segment_table, points

    # Create pointcloud video file
    utl_o3d.visualize_pointcloud_headless(
        pointcloud_file=f"pointcloud_{file_id}.h5",
        pointcloud_dir=args.pointcloud_dir,
        videos_dir=args.videos_dir,
        fps=args.scene_vis_fps
    )
    
    # Stitch camera and pointcloud videos together (if applicable)
    if args.scene_vis_stitch_videos:
        video_file = os.path.join(args.videos_dir, f"camera_{file_id}.mp4")
        pointcloud_video_file = os.path.join(args.videos_dir, f"pointcloud_{file_id}.mp4")
        if os.path.exists(video_file) and os.path.exists(pointcloud_video_file):
            output_video_file = os.path.join(args.videos_dir, f"combined_{file_id}.mp4")
            # TODO
            # utl_cam.stitch_videos(
            #     video_file,
            #     pointcloud_video_file,
            #     output_video_file,
            #     orientation="vertical",
            # )
        else:
            print(
                f"[PID {os.getpid()}] {file_id}: Could not stitch videos together "+
                f"since one or both video files were not found: '{video_file}', '{pointcloud_video_file}'"
            )


if __name__ == "__main__":

    args = Config()

    if args.use_gcp:
        gcs_client = utl_gcp.connect_to_gcp_storage(args.gcp_project_name)
        file_ids = utl.get_ids_of_complete_data_files(args.data_dir, gcs_client=gcs_client)
    else:
        file_ids = utl.get_ids_of_complete_data_files(args.data_dir)
    print(f"Total # of file IDs: {len(file_ids)}")

    args_dict = args.__dict__.copy() # ensures serialization
    with mp.Pool(processes=args.scene_vis_n_workers) as pool:
        pool.starmap(process_scene, [(file_id, args_dict) for file_id in file_ids])
