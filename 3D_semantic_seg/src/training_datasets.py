"""Custom PyTorch and Ray dataset classes."""

from typing import Any, Iterator, cast

import polars as pl
import ray
import torch
from torch.utils.data import Dataset

from utils import utils_parquet as utl_prq


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

    def _load_file(self, row: dict[str, Any]) -> Iterator[dict[str, Any]]:
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

        # Replace pyarrow join with polars join due to following pyarrow error:
        # "pyarrow.lib.ArrowInvalid: Data type list<item: float> is not supported
        # in join non-key field [LiDARComponent].range_image_return1.values"
        joined_data = cast(pl.DataFrame, pl.from_arrow(lidar_data)).join(
            cast(pl.DataFrame, pl.from_arrow(seg_labels_data)), on="index", how="inner"
        )

        for record in joined_data.iter_rows(named=True):
            yield record

    def _transform_geometry_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
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
                lbl = lbl.transpose(2, 0, 1)

            # Overwrite element in existing array reference to avoid temporary copy
            images[i] = torch.tensor(img, dtype=torch.float32)
            labels[i] = torch.tensor(lbl, dtype=torch.int64)

        return {
            "range_image": images,
            "segmentation_labels": labels,
        }

    def get_dataset(self, transform_batch_size: int = 16) -> ray.data.Dataset:
        """Build the full Ray dataset pipeline.

        Returns:
            Ray dataset where each row contains ``range_image`` and ``segmentation_labels``.
        """
        file_ids = list(self.labeled_file_obs.keys())
        base_ds = ray.data.from_items([{"file_id": fid} for fid in file_ids])
        loaded_ds = base_ds.flat_map(self._load_file)
        return loaded_ds.map_batches(
            self._transform_geometry_batch,
            batch_format="numpy",
            batch_size=transform_batch_size,
        )


class RangeImageDatasetTorch(Dataset[tuple[torch.Tensor, torch.Tensor]]):
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
            range_image_labels.reshape(range_image_labels_shape), dtype=torch.int64
        )

        # Permute to return channels first (if applicable)
        if self.return_channels_first:
            range_image = torch.permute(range_image, (2, 0, 1))
            range_image_labels = torch.permute(range_image_labels, (2, 0, 1))

        return range_image, range_image_labels
