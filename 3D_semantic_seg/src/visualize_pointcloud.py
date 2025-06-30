import os
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from utils import utils_constants as utl_c

POINTCLOUD_DIR = "pointclouds"
CAM_ZOOM = 0.2
FRAME_DELAY_SEC = 0.2

# Get list of pointcloud files
pointcloud_files = [
    x for x in os.listdir(POINTCLOUD_DIR) if x.endswith(".npz")
]

# Create Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())

# Create geometry object
pcd = o3d.geometry.PointCloud()

# Display data stream
first_pass = True
for pointcloud_file in pointcloud_files:

    # Load and extract data
    data = np.load(os.path.join(POINTCLOUD_DIR, pointcloud_file))

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

        vis.poll_events()
        vis.update_renderer()
        time.sleep(FRAME_DELAY_SEC)

# Clean up
vis.destroy_window()

#TODO: left off here 6/23 @ 11pm; output as video using OpenCV