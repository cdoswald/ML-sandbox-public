from dataclasses import dataclass, field
import logging
import os


@dataclass
class Config:
    """Configuration parameters"""

    # Logging
    log_dir: str = "/workspace/hostfiles/logs"
    log_level: int = logging.INFO

    # Input data directories
    use_gcp: bool = True
    gcp_data_dir: str = "gs://waymo_open_data_copy/data"
    gcp_project_name: str = "waymo-3d-semseg"

    local_data_dir: str = "/workspace/hostfiles/data"

    data_dir: str = field(init=False)

    # Output data directories
    videos_dir: str = "/workspace/hostfiles/videos"
    pointcloud_dir: str = "/workspace/hostfiles/pointclouds"
    models_dir: str = "/workspace/hostfiles/models"

    # Scene visualization
    scene_vis_n_workers: int = 4
    scene_vis_fps: float = 5.0
    scene_vis_vstack_videos: bool = True

    # Computation
    train_num_cpus: int = 4

    # Model training hyperparams
    max_epochs: int = 2
    batch_size: int = 4
    lr: float = 1e-3
    early_stopping_epochs: int = 5

    validation_share: float = 0.2
    test_share: float = 0.2

    def __post_init__(self) -> None:

        # Set data directory
        self.data_dir = self.gcp_data_dir if self.use_gcp else self.local_data_dir

        # Create directories
        create_dirs = [
            self.log_dir,
            self.videos_dir,
            self.pointcloud_dir,
            self.models_dir,
        ]
        if not self.use_gcp:
            create_dirs.append(self.data_dir)
        for dir_path in create_dirs:
            os.makedirs(dir_path, exist_ok=True)

    # Define constants
    # LASER_NAME_MAP = dict(wod.LaserName.Name.items())
    # CAMERA_NAME_MAP = dict(wod.CameraName.Name.items())
    # LIDAR_RETURN_MAP = dict()
    # RANGE_IMAGE_DIM_MAP = utl_cons.get_range_image_final_dim_dict()
    # SEG_IMAGE_DIM_MAP = utl_cons.get_seg_image_final_dim_dict()
