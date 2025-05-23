import numpy as np
import open3d as o3d

POINTCLOUD_PATH = "../pointcloud.npz"
CAM_ZOOM = 0.2

data = np.load(POINTCLOUD_PATH)

points = data["points"]
if len(points.shape) > 2:
    points = points.reshape((-1, 3))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

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
