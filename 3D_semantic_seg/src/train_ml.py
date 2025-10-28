"""ML model training"""

interactive_mode = True
if interactive_mode:
    import os
    os.chdir("/workspace/hostfiles/src")
    print(os.getcwd())

import os                                   #noqa: E402
import numpy as np                          #noqa: E402
from torch.utils.data import DataLoader

from config import Config
from training_datasets import RangeImageDataset
from utils import utils as utl              #noqa: E402
from utils import utils_camera as utl_cam   #noqa: E402
from utils import utils_lidar as utl_li     #noqa: E402
from utils import utils_open3d as utl_o3d   #noqa: E402


if __name__ == "__main__":

    args = Config()
    file_ids = utl.get_ids_of_complete_data_files(args.data_dir)

    # Get list of observation ids that have 3D semantic segmentation labels in each file
    # (note that each file in the lidar range image data contains multiple timesteps/observations,
    # and each observation has 5 lidar range image records--one for each laser. 
    # Only the lidar range images for the top-mounted laser have lidar segmentation labels, and
    # only about 30-50 of the approx. 200 timestep observations for the top-mounted laser in each 
    # file  have labels)
    labeled_file_obs: list[tuple[str, str]] = []
    for file_id in file_ids:
        lidar_segment_table_all = utl.load_parquet_data(args.data_dir, file_id, "lidar_segmentation")
        labeled_obs_ids = lidar_segment_table_all["index"].combine_chunks().to_pylist()
        labeled_file_obs.extend([(file_id, obs_id) for obs_id in labeled_obs_ids])
 
    # Split train, validation, and test at the file_id level 
    # (all timesteps in the same driving segment will be grouped together)
    train_share = (1 - args.validation_share - args.test_share)
    if train_share < 0.1:
        raise ValueError(f"Training data share must be at least 0.1, but got {train_share}")
    n_file_ids = len(file_ids)
    train_file_ids = np.random.choice(
        file_ids,
        size=int(train_share * n_file_ids),
        replace=False
    )
    validation_file_ids = np.random.choice(
        list(set(file_ids).difference(set(train_file_ids))),
        size=int(args.validation_share * n_file_ids),
        replace=False
    )
    test_file_ids = np.array(list(
        set(file_ids).difference(train_file_ids).difference(validation_file_ids)
    ))
    print(
        f"# obs in Training set: {len(train_file_ids)} / {n_file_ids}"+
        f"\n# obs in Validation set: {len(validation_file_ids)} / {n_file_ids}" +
        f"\n# obs in Test set: {len(test_file_ids)} / {n_file_ids}"
    )

    # Create PyTorch dataset and dataloader
    # train_data = RangeImageDataset(labeled_file_obs, args.data_dir)
    # validation_data = RangeImageDataset(labeled_file_obs, args.data_dir)
    # test_data = RangeImageDataset(labeled_file_obs, args.data_dir)




    # # plot range image
    # plt.imshow(temp[..., 0])
    