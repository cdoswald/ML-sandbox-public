"""ML model training"""

interactive_mode = True
if interactive_mode:
    import os
    os.chdir("/workspace/hostfiles/src")
    print(os.getcwd())

import os                                       #noqa: E402
import numpy as np                              #noqa: E402
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import Config
from models import BaselineCNN
from training_datasets import RangeImageDataset
from utils import utils as utl                  #noqa: E402
from utils import utils_camera as utl_cam       #noqa: E402
from utils import utils_constants as utl_cons   #noqa: E402
from utils import utils_lidar as utl_li         #noqa: E402
from utils import utils_open3d as utl_o3d       #noqa: E402


if __name__ == "__main__":

    args = Config()
    file_ids = utl.get_ids_of_complete_data_files(args.data_dir)

    # Get list of observation ids that have 3D semantic segmentation labels in each file
    # (note that each file in the lidar range image data contains multiple timesteps/observations,
    # and each observation has 5 lidar range image records--one for each laser. 
    # Only the lidar range images for the top-mounted laser have lidar segmentation labels, and
    # only about 30-50 of the approx. 200 timestep observations for the top-mounted laser in each 
    # file  have labels)
    labeled_file_obs: dict[str, list[str]] = {}
    for file_id in file_ids:
        lidar_segment_table_all = utl.load_parquet_data(args.data_dir, file_id, "lidar_segmentation")
        labeled_obs_ids = lidar_segment_table_all["index"].combine_chunks().to_pylist()
        labeled_file_obs[file_id] = labeled_obs_ids
 
    # Split train, validation, and test at the file_id level 
    # (all observations in the same driving segment will be grouped in the same split)
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
        f"\n# file ids in Training set: {len(train_file_ids)} / {n_file_ids}"+
        f"\n# file ids in Validation set: {len(validation_file_ids)} / {n_file_ids}" +
        f"\n# file ids in Test set: {len(test_file_ids)} / {n_file_ids}"
    )

    # Create PyTorch dataset and dataloaders
    train_file_obs = {k:v for k,v in labeled_file_obs.items() if k in train_file_ids}
    validation_file_obs = {k:v for k,v in labeled_file_obs.items() if k in validation_file_ids}
    test_file_obs = {k:v for k,v in labeled_file_obs.items() if k in test_file_ids}

    train_data = RangeImageDataset(train_file_obs, args.data_dir)
    validation_data = RangeImageDataset(validation_file_obs, args.data_dir)
    test_data = RangeImageDataset(test_file_obs, args.data_dir)

    n_total_obs = sum([len(x) for x in labeled_file_obs.values()])
    print(
        f"\n# obs in Training set: {len(train_data)} / {n_total_obs}"+
        f"\n# obs in Validation set: {len(validation_data)} / {n_total_obs}" +
        f"\n# obs in Test set: {len(test_data)} / {n_total_obs}"
    )

    train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_dl = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # Create model, optimizer, and loss function
    example_input, example_target = next(iter(train_dl))
    model = BaselineCNN(
        example_input,
        n_classes=len(utl_cons.get_semseg_idx_map()),
        hidden_channels=[128, 256, 512, 1024],
        avgpool_layers=[(1,5), (1,5), (1,2)],
        verbose=True
    )
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    # model(example_input)

    # Train model
    writer = SummaryWriter("runs/run_001")
    for epoch_i in range(args.max_epochs):
        for batch_j, (range_image, range_image_labels) in enumerate(train_dl):
            break

            # writer.add_scalar()
    

    