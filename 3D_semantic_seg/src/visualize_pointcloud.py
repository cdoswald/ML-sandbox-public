import os
import time

import cv2
import numpy as np
import open3d as o3d

from utils import utils_constants as utl_c

POINTCLOUD_DIR = "pointclouds"
VIDEOS_DIR = "videos"

CAM_ZOOM = 0.2
FRAME_DELAY_SEC = 0.2
FRAME_SIZE = (1280, 720)
FPS = int(1 / FRAME_DELAY_SEC)

# Get list of pointcloud files
pointcloud_files = [
    x for x in os.listdir(POINTCLOUD_DIR) if x.endswith(".npz")
]

# Create LiDAR pointcloud video for each pointcloud file
for pointcloud_file in pointcloud_files:

    # Create OpenCV video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_file = os.path.join(VIDEOS_DIR, pointcloud_file.replace(".npz", ".mp4"))
    video_writer = cv2.VideoWriter(video_file, fourcc, FPS, FRAME_SIZE)

    # Create Open3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=FRAME_SIZE[0], height=FRAME_SIZE[1])
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())

    # Create geometry object
    pcd = o3d.geometry.PointCloud()

    # Load and extract data
    data = np.load(os.path.join(POINTCLOUD_DIR, pointcloud_file))

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
        ctr.set_zoom(CAM_ZOOM)

        # Render image
        vis.poll_events()
        vis.update_renderer()

        # Capture frame and convert to BGR for OpenCV output
        img = vis.capture_screen_float_buffer(do_render=False)
        img = (np.asarray(img) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write frame to video
        video_writer.write(img_bgr)

        time.sleep(FRAME_DELAY_SEC)

    # Clean up
    video_writer.release()
    vis.destroy_window()
