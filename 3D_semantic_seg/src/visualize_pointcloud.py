import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from utils import utils_constants as utl_c


POINTCLOUD_PATH = "../pointcloud.npz"
CAM_ZOOM = 0.2

# Load and extract data
data = np.load(POINTCLOUD_PATH)
points = data["points"]

# Create Open3D visualization
pcd = o3d.geometry.PointCloud()

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

# Adjust default camera position
cam_lookat = np.array([0, 0, 0])
cam_up = np.array([0, 0, 1])
cam_front = np.array([-1, 0, 1])
cam_front = cam_front / np.linalg.norm(cam_front)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.add_geometry(
    o3d.geometry.TriangleMesh.create_coordinate_frame()
)

ctr = vis.get_view_control()
ctr.set_lookat(cam_lookat)
ctr.set_front(cam_front)
ctr.set_up(cam_up)
ctr.set_zoom(CAM_ZOOM)

vis.run()

vis.destroy_window()
