"""Open3D utility functions"""

import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from utils import utils_constants as utl_c


def visualize_pointcloud(
    pointcloud_file: str,
    pointcloud_dir: str,
    output_resolution: Tuple[int, int] = (1280, 720)
    fps: float = 10.0,
    camera_zoom: Optional[float] = 0.2,
    save_video: bool = True,
    videos_dir: Optional[str] = None,
) -> None:
    """ """ #TODO: add docstring
    video_writer = None

    # Create OpenCV video writer (if applicable)
    if save_video:
        if videos_dir is None:
            raise ValueError("Argument 'videos_dir' required but not provided.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_file = os.path.join(videos_dir, pointcloud_file.replace(".npz", ".mp4"))
        video_writer = cv2.VideoWriter(video_file, fourcc, fps, output_resolution)

    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=output_resolution[0], height=output_resolution[1])
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())

    # Create geometry object
    pcd = o3d.geometry.PointCloud()

    # Load and extract data
    data = np.load(os.path.join(pointcloud_dir, pointcloud_file))

    # Display data stream
    first_pass = True
    for obs_id, points in data.items():

        # Add (x,y,z) coords
        points_xyz = points[..., :3].reshape((-1, 3))
        pcd.points = o3d.utility.Vector3dVector(points_xyz)

        # Add semseg labels (if applicable)
        if points.shape[-1] > 3:
            points_semseg = points[..., 3:]
            points_instances = points_semseg[..., 0].flatten()
            points_classes = points_semseg[..., 1].flatten()

            # Map semseg classes to colormap
            color_dict = utl_c.get_semseg_rgb_map()
            colors = np.array([color_dict[label] for label in points_classes])
            pcd.colors = o3d.utility.Vector3dVector(colors)

        # Update geometry
        if first_pass:
            vis.add_geometry(pcd)
            first_pass = False
            # Add render warm-up
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
        else:
            vis.update_geometry(pcd)

        # Adjust default camera position
        cam_lookat = np.array([0, 0, 0])
        cam_up = np.array([0, 0, 1])
        cam_front = np.array([-1, 0, 1])
        cam_front = cam_front / np.linalg.norm(cam_front)

        ctr = vis.get_view_control()
        ctr.set_lookat(cam_lookat)
        ctr.set_front(cam_front)
        ctr.set_up(cam_up)
        ctr.set_zoom(camera_zoom)

        # Render image
        vis.poll_events()
        vis.update_renderer()

        if save_video:

            # Capture frame and convert to BGR for OpenCV output
            img = vis.capture_screen_float_buffer(do_render=False)
            img = (np.asarray(img) * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # Write frame to video
            video_writer.write(img_bgr)

        time.sleep(1/fps)

    # Clean up
    if video_writer is not None:
        video_writer.release()
    vis.destroy_window()
