"""Open3D utility functions"""

import os
import time
from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from utils import utils_constants as utl_c


def visualize_pointcloud_headless(
    pointcloud_file: str,
    pointcloud_dir: str,
    videos_dir: str,
    output_resolution: tuple[int, int] = (1280, 720),
    fps: float = 10.0,
    camera_zoom: Optional[float] = 0.2,
) -> None:
    """Render pointcloud sequence from saved npz file in headless environment
    (e.g., Docker container). Rendered image will only be saved to .mp4 file and
    will not be displayed in a graphical user interface.

    Args:
        pointcloud_file: pointcloud npz filename
        pointcloud_dir: pointcloud npz directory name
        videos_dir: directory name to save rendered pointcloud sequence file
        output_resolution: (width, height) of rendered frames
        fps: frames per second
        camera_zoom: camera zoom amount

    Returns:
        None
    """
    # Create OpenCV video writer
    if videos_dir is None:
        raise ValueError("Argument 'videos_dir' required but not provided.")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_file = os.path.join(videos_dir, pointcloud_file.replace(".npz", ".mp4"))
    video_writer = cv2.VideoWriter(video_file, fourcc, fps, output_resolution)

    # Set up Open3D materials
    materials = o3d.visualization.rendering.MaterialRecord()
    materials.shader = "defaultLit"

    # Set up Open3D renderer
    renderer = o3d.visualization.rendering.OffscreenRenderer(
        width=output_resolution[0], height=output_resolution[1]
    )
    renderer.scene.add_geometry(
        "axes",
        o3d.geometry.TriangleMesh.create_coordinate_frame(),
        materials,
    )

    # Load and extract data
    data = np.load(os.path.join(pointcloud_dir, pointcloud_file))

    # Display data stream
    first_pass = True
    for obs_id, points in data.items():

        # Add (x,y,z) coords
        pcd = o3d.geometry.PointCloud()
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
        if not first_pass:
            renderer.scene.remove_geometry("pcd")
        renderer.scene.add_geometry("pcd", pcd, materials)
        first_pass = False

        # Adjust default camera position
        cam_lookat = np.array([0, 0, 0])
        cam_up = np.array([0, 0, 1])
        cam_front = np.array([-1, 0, 0.5])
        cam_front = cam_front / np.linalg.norm(cam_front)
        cam_distance = 3.0 * (1.0 / camera_zoom)
        cam_pos = cam_lookat + cam_front * cam_distance

        renderer.setup_camera(60, cam_lookat, cam_pos, cam_up)

        # Render image
        img = renderer.render_to_image()
        img = np.asarray(img).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write frame to video
        video_writer.write(img_bgr)

    # Clean up
    video_writer.release()
    del renderer


def visualize_pointcloud(
    pointcloud_file: str,
    pointcloud_dir: str,
    output_resolution: tuple[int, int] = (1280, 720),
    fps: float = 10.0,
    camera_zoom: Optional[float] = 0.2,
    save_video: bool = False,
    videos_dir: Optional[str] = None,
) -> None:
    """Render and display pointcloud sequence from saved npz file.

    Args:
        pointcloud_file: pointcloud npz filename
        pointcloud_dir: pointcloud npz directory name
        output_resolution: (width, height) of rendered frames
        fps: frames per second
        camera_zoom: camera zoom amount
        save_video: if True, will save rendered frames to .mp4 file
        videos_dir: if save_video is True, then .mp4 file will be saved to this dir

    Returns:
        None
    """
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
