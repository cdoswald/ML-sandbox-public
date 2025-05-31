"""Constant utility functions for Waymo Open Dataset challenges."""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


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
        5: "REAR"
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
        8: "REAR_RIGHT"
    }

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
        0: {"name": "Car", "color": "darkorange"},
        1: {"name": "Truck", "color": "orangered"},
        2: {"name": "Bus", "color": "tomato"},
        3: {"name": "Motorcyclist", "color": "gold"},
        4: {"name": "Bicyclist", "color": "khaki"},
        5: {"name": "Pedestrian", "color": "yellow"},
        6: {"name": "Sign", "color": "deepskyblue"},
        7: {"name": "Traffic Light", "color": "dodgerblue"},
        8: {"name": "Pole", "color": "lightskyblue"},
        9: {"name": "Construction Cone", "color": "sandybrown"},
        10: {"name": "Bicycle", "color": "lightgoldenrodyellow"},
        11: {"name": "Motorcycle", "color": "palegoldenrod"},
        12: {"name": "Building", "color": "slategray"},
        13: {"name": "Vegetation", "color": "forestgreen"},
        14: {"name": "Tree Trunk", "color": "saddlebrown"},
        15: {"name": "Curb", "color": "darkslategray"},
        16: {"name": "Road", "color": "dimgray"},
        17: {"name": "Lane Marker", "color": "gray"},
        18: {"name": "Walkable", "color": "lightgray"},
        19: {"name": "Sidewalk", "color": "gainsboro"},
        20: {"name": "Other Ground", "color": "lightsteelblue"},
        21: {"name": "Other Vehicle", "color": "lightsalmon"},
        22: {"name": "Undefined", "color": "black"}
    }

def get_semseg_rgb_map() -> Dict[int, Tuple[float, float, float]]:
    """Get dictionary mapping 3D semantic segmentation index to RGB value."""
    semseg_map = get_semseg_idx_map()
    return {
        k:mcolors.to_rgb(v["color"]) for k,v in semseg_map.items()
    }
