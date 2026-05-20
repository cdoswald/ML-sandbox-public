from dataclasses import dataclass
import os

@dataclass
class Config:
    """Configuration parameters"""

    use_gcp: bool = True
    gcp_data_dir: str = "gs://waymo_open_data_copy/data"
    gcp_project_name: str = "waymo-3d-semseg"

    local_data_dir: str = "/workspace/hostfiles/data"
    videos_dir: str = "/workspace/hostfiles/videos"
    pointcloud_dir: str = "/workspace/hostfiles/pointclouds"

    # Scene visualization
    scene_vis_n_workers: int = 4
    scene_vis_fps: float = 5.0
    scene_vis_vstack_videos: bool = True

    max_epochs: int = 2
    batch_size: int = 4
    lr: float = 1e-3

    validation_share: float = 0.2
    test_share: float = 0.2


    def __post_init__(self):
        self.data_dir = self.gcp_data_dir if self.use_gcp else self.local_data_dir

        # Create local directories only when running locally
        if not self.use_gcp:
            os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.pointcloud_dir, exist_ok=True)
