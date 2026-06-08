"""Custom PyTorch and Ray dataset classes."""

import os
from typing import Iterable, Optional

import pyarrow.parquet as pq
import pyarrow.compute as pc
import ray
import torch
from torch.utils.data import Dataset

from utils import utils_parquet as utl_prq
from utils import utils as utl


class RangeImageDatasetRay:
    """Custom Ray dataset class for LiDAR range image data."""

    def __init__(
        self,
        labeled_file_obs: dict[str, list[str]],
        data_dir: str,
        return_channels_first: bool = True,
    ) -> None:
        """Initialize the Ray dataset wrapper.

        Args:
            labeled_file_obs: Mapping of file ids to observation ids that have labels.
            data_dir: Root directory containing Waymo parquet subdirectories.
            return_channels_first: If True, return images with channels first (C, H, W);
                otherwise, return images with channels last (H, W, C).
        """
        self.labeled_file_obs = labeled_file_obs
        self.data_dir = data_dir
        self.return_channels_first = return_channels_first

    def _load_file(self, row: dict) -> Iterable[dict]:
        """Load and join LiDAR and segmentation label rows for one file.

        Args:
            row: A Ray row containing a ``file_id`` key.

        Yields:
            Joined LiDAR + segmentation records, one per observation id.
        """
        file_id = row["file_id"]
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

        joined_data = lidar_data.join(seg_labels_data, keys="index", how="inner")

        for record in joined_data.to_pylist():
            yield record

    def _transform_geometry_batch(self, batch: dict) -> dict:
        """Reshape range-image arrays and optionally move channels before height and width dims.

        Args:
            batch: Numpy-format batch emitted by Ray Data containing flattened
                value arrays and corresponding shape arrays.

        Returns:
            Dictionary with observation ids, reshaped range images, and
            reshaped segmentation labels.
        """

        # Extract the whole column arrays from the batch
        images = batch["[LiDARComponent].range_image_return1.values"]
        images_shapes = batch["[LiDARComponent].range_image_return1.shape"]
        labels = batch["[LiDARSegmentationLabelComponent].range_image_return1.values"]
        labels_shapes = batch[
            "[LiDARSegmentationLabelComponent].range_image_return1.shape"
        ]

        for i in range(len(images)):
            # Reshape lidar image and segmentation labels in place
            img = images[i].reshape(images_shapes[i])
            lbl = labels[i].reshape(labels_shapes[i])

            # Swap channels to return [C, H, W] if necessary
            if self.return_channels_first:
                img = img.transpose(2, 0, 1)
                if lbl.ndim == 3 and lbl.shape[2] == 1:
                    lbl = lbl.transpose(2, 0, 1)

            # Overwrite element in existing array reference to avoid temporary copy
            images[i] = img
            labels[i] = lbl

        return {
            "obs_id": batch["index"],
            "range_image": images,
            "segmentation_labels": labels,
        }

    def get_dataset(self) -> ray.data.Dataset:
        """Build the full Ray dataset pipeline.

        Returns:
            Ray dataset where each row contains ``obs_id``, ``range_image``,
            and ``segmentation_labels``.
        """
        file_ids = list(self.labeled_file_obs.keys())
        base_ds = ray.data.from_items([{"file_id": fid} for fid in file_ids])

        return base_ds.flat_map(self._load_file).map_batches(
            self._transform_geometry_batch, batch_format="numpy"
        )


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
        """Return the number of labeled observations available."""
        return len(self.file_obs_tuples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a range image tensor and matching segmentation labels.

        Args:
            idx: Observation index in the flattened ``(file_id, obs_id)`` list.

        Returns:
            Tuple of tensors ``(range_image, range_image_labels)``.
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
