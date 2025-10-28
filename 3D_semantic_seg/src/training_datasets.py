"""Custom PyTorch dataset classes."""

import torch
from torch.utils.data import Dataset

from utils import utils as utl

class RangeImageDataset(Dataset):
    """Custom PyTorch dataset class for LiDAR range image data."""

    def __init__(
        self,
        labeled_file_obs: list[tuple[str, str]],
        data_dir: str,
    ) -> None:
        """
        Args:
            labeled_file_obs: list of tuples containing (file_id, obs_id)
            data_dir: str path to data directory

        Returns:
            None
        """
        self.labeled_file_obs = labeled_file_obs
        self.data_dir = data_dir


    def __len__(self) -> int:
        return len(self.labeled_file_obs)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get range image and corresponding segmentation labels.
        
        Args:
            idx: observation index

        Returns:
            tuple of PyTorch tensors containing (range_image, range_image_labels)
        """
        
        file_id, obs_id = self.labeled_file_obs[idx]

        # Load pyarrow tables for lidar image and lidar segmentation labels
        filter_lidar_rows = {"index":[obs_id], "key.laser_name":[1]} # single obs only; top laser only
        lidar_image_table = utl.load_parquet_data(
            self.data_dir,
            file_id,
            data_subdir="lidar",
            subset_cols=[
                "[LiDARComponent].range_image_return1.values",
                "[LiDARComponent].range_image_return1.shape",
            ],
            filter_rows=filter_lidar_rows,
        )
        lidar_segment_table = utl.load_parquet_data(
            self.data_dir,
            file_id,
            data_subdir="lidar_segmentation",
            subset_cols=[
                "[LiDARSegmentationLabelComponent].range_image_return1.values",
                "[LiDARSegmentationLabelComponent].range_image_return1.shape",
            ],
            filter_rows=filter_lidar_rows,
        )
        
        # Reformat as tensors
        range_image = (
            lidar_image_table["[LiDARComponent].range_image_return1.values"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            [0] # extract first and only array (since only one obs_id was passed to load_parquet_data)
        )
        range_image_shape = (
            lidar_image_table["[LiDARComponent].range_image_return1.shape"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            [0]
        )
        range_image = torch.tensor(range_image.reshape(range_image_shape))
        
        range_image_labels = (
            lidar_segment_table["[LiDARSegmentationLabelComponent].range_image_return1.values"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            [0]
        )
        range_image_labels_shape = (
            lidar_segment_table["[LiDARSegmentationLabelComponent].range_image_return1.shape"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            [0]
        )
        range_image_labels = torch.tensor(range_image_labels.reshape(range_image_labels_shape))

        return range_image, range_image_labels
