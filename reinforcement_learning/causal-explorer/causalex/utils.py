import h5py
from math import comb
from typing import Dict, List

import cv2
import numpy as np


def calculate_n_interactions(n_action_dims: int):
    """Calculate number of unique interactions for action space."""
    total_combinations = 0
    for i in range(1, n_action_dims+1): # Note that C(n_action_dims, 0) is not applicable
        total_combinations += comb(n_action_dims, i)
    return total_combinations


def save_video(frames, save_path, fps=30):
    height, width, layers = frames[0].shape
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(* "mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, size)
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def save_dict_to_hdf5(filepath: str, data_dict: Dict[str, List[np.float32]]):
    with h5py.File(filepath, "a") as file:
        for key, data in data_dict.items():
            if key not in file:
                file.create_dataset(key, data=data, maxshape=(None,), chunks=True)
            else:
                file[key].resize((file[key].shape[0] + len(data),))
                file[key][-len(data):] = data
