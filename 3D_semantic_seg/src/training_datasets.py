"""Custom PyTorch and Ray dataset classes."""

import os

import pyarrow.parquet as pq
import pyarrow.compute as pc
import ray
import torch
from torch.utils.data import Dataset

from utils import utils_parquet as utl_prq
from utils import utils as utl


class RangeImageDatasetRay:
    """Custom Ray dataset class for LiDAR range image data."""

    def __init__(self, labeled_file_obs: dict[str, list[str]], data_dir: str):
        self.labeled_file_obs = labeled_file_obs
        self.data_dir = data_dir

    def get_dataset(self) -> ray.data.Dataset:
        file_ids = list(self.labeled_file_obs.keys())
        base_ds = ray.data.from_items([{"file_id": fid} for fid in file_ids])

        def load_file(ds_row: dict) -> list[dict]:
            file_id = ds_row["file_id"]
            valid_obs_ids = set(self.labeled_file_obs[file_id])

            lidar_data = utl_prq.load_parquet_data(
                data_dir=self.data_dir,
                file_id=file_id,
                data_subdir="lidar",
                subset_cols=[
                    "index",
                    "[LiDARComponent].range_image_return1.values",
                    "[LiDARComponent].range_image_return1.shape",
                ],
                filter_rows={"index": list(valid_obs_ids), "key.laser_name": [1]},
            )

            seg_labels_data = utl_prq.load_parquet_data(
                data_dir=self.data_dir,
                file_id=file_id,
                data_subdir="lidar_segmentation",
                subset_cols=[
                    "index",
                    "[LiDARSegmentationLabelComponent].range_image_return1.values",
                    "[LiDARSegmentationLabelComponent].range_image_return1.shape",
                ],
                filter_rows={"index": list(valid_obs_ids), "key.laser_name": [1]},
            )

            # TODO: left off here at 10:15pm on 6/4;
            # need to join lidar_data and seg_labels_data for Ray dataset?
            # also use yield instead of return?

        final_ds = base_ds.flat_map(load_file)
        return final_ds


class RangeImageDatasetTorch(Dataset):
    """Custom PyTorch dataset class for LiDAR range image data."""

    def __init__(
        self,
        labeled_file_obs: dict[str, list[str]],
        data_dir: str,
        return_channels_first: bool = True,
    ) -> None:
        """
        Args:
            labeled_file_obs: dict mapping file_id to list of file observation ids
            data_dir: str path to data directory
            return_channels_first: if True, return [C, H, W]; otherwise, return [H, W, C]

        Returns:
            None
        """
        self.labeled_file_obs = labeled_file_obs
        self.data_dir = data_dir
        self.return_channels_first = return_channels_first

        # Format labeled_file_obs for __getitem__ indexing
        self.file_obs_tuples: list[tuple[str, str]] = []
        for file_id, obs_ids in labeled_file_obs.items():
            self.file_obs_tuples.extend([(file_id, obs_id) for obs_id in obs_ids])

    def __len__(self) -> int:
        return len(self.file_obs_tuples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get range image and corresponding segmentation labels.

        Args:
            idx: observation index

        Returns:
            tuple of PyTorch tensors containing (range_image, range_image_labels)
        """

        file_id, obs_id = self.file_obs_tuples[idx]

        # Load pyarrow tables for lidar image and lidar segmentation labels
        filter_lidar_rows = {
            "index": [obs_id],
            "key.laser_name": [1],
        }  # single obs only; top laser only
        lidar_image_table = utl_prq.load_parquet_data(
            self.data_dir,
            file_id,
            data_subdir="lidar",
            subset_cols=[
                "[LiDARComponent].range_image_return1.values",
                "[LiDARComponent].range_image_return1.shape",
            ],
            filter_rows=filter_lidar_rows,
        )
        lidar_segment_table = utl_prq.load_parquet_data(
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
            .to_numpy(zero_copy_only=False)[
                0
            ]  # extract first and only array (since only one obs_id was passed to load_parquet_data)
        )
        range_image_shape = (
            lidar_image_table["[LiDARComponent].range_image_return1.shape"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)[0]
        )
        range_image = torch.tensor(range_image.reshape(range_image_shape))

        range_image_labels = (
            lidar_segment_table[
                "[LiDARSegmentationLabelComponent].range_image_return1.values"
            ]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)[0]
        )
        range_image_labels_shape = (
            lidar_segment_table[
                "[LiDARSegmentationLabelComponent].range_image_return1.shape"
            ]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)[0]
        )
        range_image_labels = torch.tensor(
            range_image_labels.reshape(range_image_labels_shape)
        )

        # Permute to return channels first (if applicable)
        if self.return_channels_first:
            range_image = torch.permute(range_image, (2, 0, 1))
            range_image_labels = torch.permute(range_image_labels, (2, 0, 1))

        return range_image, range_image_labels
