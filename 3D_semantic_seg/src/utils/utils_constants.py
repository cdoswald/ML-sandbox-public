"""Constant utility functions"""

from typing import Dict, List, Tuple

import matplotlib.colors as mcolors


def get_laser_idx_map() -> Dict[int, str]:
    """Get dictionary mapping LiDAR laser index to laser name."""
    # Note that you can programatically access the values with:
    # ----------------------------------------------------------
    #   from waymo_open_dataset import dataset_pb2 as wod
    #   {v:k for k,v in wod.LaserName.Name.items()}
    # ----------------------------------------------------------
    # but hardcoding so this module does not require wod dependency
    return {
        0: "UNKNOWN",
        1: "TOP",
        2: "FRONT",
        3: "SIDE_LEFT",
        4: "SIDE_RIGHT",
        5: "REAR",
    }


def get_camera_idx_map() -> Dict[int, str]:
    """Get dictionary mapping camera index to camera name."""
    # Note that you can programatically access the values with:
    # ----------------------------------------------------------
    #   from waymo_open_dataset import dataset_pb2 as wod
    #   {v:k for k,v in wod.CameraName.Name.items()}
    # ----------------------------------------------------------
    # but hardcoding so this module does not require wod dependency
    return {
        0: "UNKNOWN",
        1: "FRONT",
        2: "FRONT_LEFT",
        3: "FRONT_RIGHT",
        4: "SIDE_LEFT",
        5: "SIDE_RIGHT",
        6: "REAR_LEFT",
        7: "REAR",
        8: "REAR_RIGHT",
    }


def get_veh_view_camera_name_order() -> List[str]:
    """Get list of camera names ordered to reflect vehicle viewpoint."""
    return [
        "REAR_LEFT",
        "SIDE_LEFT",
        "FRONT_LEFT",
        "FRONT",
        "FRONT_RIGHT",
        "SIDE_RIGHT",
        "REAR_RIGHT",
        "REAR",
    ]


def get_range_image_last_dim_map() -> Dict[int, str]:
    """Get dictionary mapping last dim of range image to signal type
    (e.g., distance, intensity, elongation)."""
    return {
        0: "DISTANCE",
        1: "INTENSITY",
        2: "ELONGATION",
    }


def get_semseg_image_last_dim_map() -> Dict[int, str]:
    """Get dictionary mapping last dim of segmentation image to type
    (e.g., instance, class)."""
    return {
        0: "INSTANCE_ID",
        1: "CLASS_ID",
    }


def get_semseg_idx_map() -> Dict[int, Dict[str, str]]:
    """Get dictionary mapping 3D semantic segmentation index to label
    name and color."""
    return {
        0: {"name": "Undefined", "color": "darkgray"},
        1: {"name": "Car", "color": "teal"},
        2: {"name": "Truck", "color": "darkturquoise"},
        3: {"name": "Bus", "color": "cadetblue"},
        4: {"name": "Other Vehicle", "color": "mediumpurple"},
        5: {"name": "Motorcyclist", "color": "red"},  # Need confirmation
        6: {"name": "Bicyclist", "color": "lime"},
        7: {"name": "Pedestrian", "color": "magenta"},
        8: {"name": "Sign", "color": "lemonchiffon"},
        9: {"name": "Traffic Light", "color": "yellow"},
        10: {"name": "Pole", "color": "darkslategrey"},
        11: {"name": "Construction Cone", "color": "orange"},
        12: {"name": "Bicycle", "color": "palegreen"},
        13: {"name": "Motorcycle", "color": "red"},  # Need confirmation
        14: {"name": "Building", "color": "burlywood"},
        15: {"name": "Vegetation", "color": "forestgreen"},
        16: {"name": "Tree Trunk", "color": "sienna"},
        17: {"name": "Curb", "color": "darkgoldenrod"},
        18: {"name": "Road", "color": "black"},
        19: {"name": "Lane marker", "color": "snow"},
        20: {"name": "Walkable", "color": "moccasin"},
        21: {"name": "Other ground", "color": "darkolivegreen"},
        22: {"name": "Sidewalk", "color": "gainsboro"},
    }


def get_semseg_rgb_map() -> Dict[int, Tuple[float, float, float]]:
    """Get dictionary mapping 3D semantic segmentation index to RGB value."""
    semseg_map = get_semseg_idx_map()
    return {k: mcolors.to_rgb(v["color"]) for k, v in semseg_map.items()}
