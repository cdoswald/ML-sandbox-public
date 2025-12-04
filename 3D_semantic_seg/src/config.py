from dataclasses import dataclass
import os

@dataclass
class Config:
    """Configuration parameters"""

    data_dir = "/workspace/hostfiles/data"
    videos_dir = "/workspace/hostfiles/videos"
    pointcloud_dir = "/workspace/hostfiles/pointclouds"

    max_epochs: int = 2
    batch_size: int = 4
    lr: float = 1e-3
    
    validation_share: float = 0.2
    test_share: float = 0.2

    def __post_init__(self):

        # Create directories
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.pointcloud_dir, exist_ok=True)
