"""LiDAR utility functions"""

from typing import Optional

import numpy as np
import pyarrow


def convert_lidar_range_image_to_xyz_coords(
    lidar_image_table: pyarrow.lib.Table,
    lidar_calib_table: pyarrow.lib.Table,
    lidar_segment_table: Optional[pyarrow.lib.Table] = None,
    lidar_return_count: int = 1,
    lidar_return_type: int = 0,
    convert_to_world_ref: bool = False,
) -> np.ndarray:
    """Project 2D LiDAR range image values to 3D coordinates.

    Args
        lidar_image_table: pyarrow table containing LiDAR image
        lidar_calib_table: pyarrow table containing LiDAR calibration
        lidar_segment_table: pyarrow table containing LiDAR semantic segmentation labels 
            (optional; default = None)
        lidar_return_count: index of LiDAR return number (options = {1, 2}; default = 1)
        lidar_return_type: index of LiDAR return type (default = 0 (distance))
        convert_to_world_ref: bool indicating whether to convert LiDAR coordinate
            system to world reference system (default = False)

    Returns
        numpy ndarray of (LiDAR rows, LiDAR cols, (x, y, z))
    """
    if lidar_return_count not in [1, 2]:
        raise ValueError(
            f"Lidar return ID must be 1 or 2; got {lidar_return_count}"
        )

    # Extract lidar values
    col_prefix = f"[LiDARComponent].range_image_return{lidar_return_count}"
    lidar_shape = lidar_image_table.column(f"{col_prefix}.shape").combine_chunks().to_pylist()[0]
    lidar_vals = np.array(
        lidar_image_table.column(f"{col_prefix}.values").combine_chunks().to_pylist()[0]
    ).reshape(lidar_shape)[..., lidar_return_type]

    # import matplotlib.pyplot as plt
    # plt.imshow(lidar_vals)
    # plt.colorbar()
    # plt.show()

    # Create grid of azimuth angles x beam inclination angles
    # (from horizontal plane of sensor, negative lidar beam inclination
    # points toward ground and positive points toward sky)
    beam_col = "[LiDARCalibrationComponent].beam_inclination.values"
    lidar_beam_incl_vals = np.array(
        lidar_calib_table.column(beam_col).combine_chunks().to_pylist()[0]
    )
    # (reverse to align beam inclination orientation with range image)
    lidar_beam_incl_vals = lidar_beam_incl_vals[::-1]
    # (similarly, azimuth should go from pi to -pi to align with range image; assume evenly spaced)
    lidar_azimuth_vals = np.linspace(np.pi, -np.pi, num=lidar_shape[1], endpoint=False) 
    azimuth_grid, beam_incl_grid = np.meshgrid(lidar_azimuth_vals, lidar_beam_incl_vals)

    # Convert to (x,y,z) coords (theta = beam inclination; phi = azimuth; r = distance)
    x = lidar_vals * np.cos(beam_incl_grid) * np.cos(azimuth_grid)
    y = lidar_vals * np.cos(beam_incl_grid) * np.sin(azimuth_grid)
    z = lidar_vals * np.sin(beam_incl_grid)
    points = np.stack([x, y, z], axis=-1)

    # Convert to world reference coords (if applicable)
    if convert_to_world_ref:

        # Extract lidar extrinsic matrix
        extrin_col = "[LiDARCalibrationComponent].extrinsic.transform"
        lidar_extrin_matrix = np.array(
            lidar_calib_table.column(extrin_col).combine_chunks().to_pylist()[0]
        ).reshape((4,4))

        # Convert (x,y,z) to homogenous coords (x,y,z,1)
        n_rows = points.shape[0]
        n_cols = points.shape[1]
        ones_tensor = np.ones((n_rows, n_cols, 1))
        points = np.concatenate([points, ones_tensor], axis=-1)

        # Multiply homogenous coords by lidar extrinsic matrix to convert to world ref
        points = np.einsum("ij,klj->kli", lidar_extrin_matrix, points)

        # Convert back to (x,y,z) coords
        points = points[..., :3]

    # Concatenate semantic segmentation labels (if applicable)
    if lidar_segment_table is not None:

        semseg_col_prefix = f"[LiDARSegmentationLabelComponent].range_image_return{lidar_return_count}"
        semseg_shape = lidar_segment_table.column(f"{semseg_col_prefix}.shape").combine_chunks().to_pylist()[0]
        semseg_vals = np.array(
            lidar_segment_table.column(f"{semseg_col_prefix}.values").combine_chunks().to_pylist()[0]
        ).reshape(semseg_shape) # Return all final dims (0 is instance ID, 1 is class ID)

        points = np.concatenate([points, semseg_vals], axis=-1)

    return points
