"""LiDAR utility functions for Waymo Open Dataset challenges."""

def convert_lidar_range_image_to_xyz_coords(
    lidar_image_table: pyarrow.lib.Table,
    lidar_calib_table: pyarrow.lib.Table,
    lidar_return_count: int = 1,
    lidar_return_type: int = 0,
    convert_to_world_ref: bool = True, 
):
    """Project 2D LiDAR range image values to 3D coordinates.

    Args

    Returns

    """
    if lidar_return not in [1, 2]:
        raise ValueError(
            f"Lidar return ID must be 1 or 2; got {lidar_return}"
        )

    # Extract lidar values
    col_prefix = f"[LiDARComponent].range_image_return{lidar_return}"
    lidar_shape = lidar_image_table.column(f"{col_prefix}.shape").combine_chunks().to_pylist()[0]
    lidar_vals = np.array(
        lidar_image_table.column(f"{col_prefix}.values").combine_chunks().to_pylist()[0]
    ).reshape(lidar_shape)[..., lidar_return_type]

    # Create grid of azimuth angles x beam inclination angles
    beam_col = "[LiDARCalibrationComponent].beam_inclination.values"
    lidar_beam_incl_vals = np.array(
        lidar_calib_table.column(beam_col).combine_chunks().to_pylist()[0]
    )
    lidar_azimuth_vals = np.linspace(-np.pi, np.pi, num=lidar_shape[1], endpoint=False) # Assume evenly spaced
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
        flattened_points = points.reshape((-1, 3)).T
        ones_array = np.ones((1, flattened_points.shape[-1]))
        homogenous_coords = np.concatenate([flattened_points, ones_array], axis=0)

        # Multiply by extrinsics (on left), drop homogenous coord, and reshape
        points = (lidar_extrin_matrix @ homogenous_coords) # 4 x n matrix
        points = points[:3, :].T.reshape((n_rows, n_cols))

    return points
