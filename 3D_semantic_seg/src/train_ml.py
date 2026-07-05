"""ML model training"""

interactive_mode = True
if interactive_mode:
    import os

    os.chdir("/workspace/hostfiles/src")
    print(os.getcwd())

from datetime import datetime  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import numpy as np  # noqa: E402
import ray  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import Adam  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from config import Config  # noqa: E402
from models import BaselineCNN  # noqa: E402
from training_datasets import RangeImageDatasetRay  # noqa: E402
from utils import utils as utl  # noqa: E402
from utils import utils_constants as utl_cons  # noqa: E402
from utils import utils_gcp as utl_gcp  # noqa: E402
from utils import utils_parquet as utl_prq  # noqa: E402

# from torch.utils.data import DataLoader # noqa: E402
# from training_datasets import RangeImageDatasetTorch # noqa: E402


if __name__ == "__main__":
    # Import config params
    args = Config()

    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=args.train_num_cpus)

    # Set up logging
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )

    # Get file IDs of all complete data files in the data directory (either GCP or local)
    if args.use_gcp:
        gcs_client = utl_gcp.connect_to_gcp_storage(args.gcp_project_name)
        file_ids = utl.get_ids_of_complete_data_files(
            args.data_dir, gcs_client=gcs_client
        )
    else:
        file_ids = utl.get_ids_of_complete_data_files(args.data_dir)
    logging.info(f"Total # of file IDs: {len(file_ids)}")

    # Get list of observation ids that have 3D semantic segmentation labels in each file
    # (note that each file in the lidar range image data contains multiple timesteps/observations,
    # and each observation has 5 lidar range image records--one for each laser.
    # Only the lidar range images for the top-mounted laser have lidar segmentation labels, and
    # only about 30-50 of the approx. 200 timestep observations for the top-mounted laser in each
    # file  have labels)
    labeled_file_obs: dict[str, list[str]] = {}
    for file_id in file_ids:
        lidar_segment_table_all = utl_prq.load_parquet_data(
            args.data_dir, file_id, "lidar_segmentation"
        )
        labeled_obs_ids = lidar_segment_table_all["index"].combine_chunks().to_pylist()
        labeled_file_obs[file_id] = labeled_obs_ids

    # Split train, validation, and test at the file_id level
    # (all observations in the same driving segment will be grouped in the same split)
    train_share = 1 - args.validation_share - args.test_share
    if train_share < 0.1:
        raise ValueError(
            f"Training data share must be at least 0.1, but got {train_share}"
        )
    n_file_ids = len(file_ids)
    train_file_ids = np.random.choice(
        file_ids, size=int(train_share * n_file_ids), replace=False
    )
    validation_file_ids = np.random.choice(
        list(set(file_ids).difference(set(train_file_ids))),
        size=int(args.validation_share * n_file_ids),
        replace=False,
    )
    test_file_ids = np.array(
        list(set(file_ids).difference(train_file_ids).difference(validation_file_ids))
    )
    logging.info(
        f"\n# file ids in Training set: {len(train_file_ids)} / {n_file_ids}"
        + f"\n# file ids in Validation set: {len(validation_file_ids)} / {n_file_ids}"
        + f"\n# file ids in Test set: {len(test_file_ids)} / {n_file_ids}"
    )

    train_file_obs = {k: v for k, v in labeled_file_obs.items() if k in train_file_ids}
    validation_file_obs = {
        k: v for k, v in labeled_file_obs.items() if k in validation_file_ids
    }
    test_file_obs = {k: v for k, v in labeled_file_obs.items() if k in test_file_ids}

    train_data = RangeImageDatasetRay(train_file_obs, args.data_dir).get_dataset(
        transform_batch_size=4
    )
    validation_data = RangeImageDatasetRay(
        validation_file_obs, args.data_dir
    ).get_dataset(transform_batch_size=4)
    test_data = RangeImageDatasetRay(test_file_obs, args.data_dir).get_dataset(
        transform_batch_size=4
    )

    # Create data iterators
    train_data_iterator = train_data.iter_torch_batches(batch_size=args.batch_size)
    validation_data_iterator = validation_data.iter_torch_batches(
        batch_size=args.batch_size
    )

    # Get example data for setting model input dims
    example_data = next(iter(train_data_iterator))
    example_input, example_target = (
        example_data["range_image"],
        example_data["segmentation_labels"],
    )

    # Create model, optimizer, and loss function
    model = BaselineCNN(
        example_input,
        n_classes=len(utl_cons.get_semseg_idx_map()),
        hidden_channels=[128, 256, 512, 1024],
        avgpool_layers=[(1, 5), (1, 5), (1, 2)],
        verbose=True,
    )
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    # Test forward pass
    with torch.no_grad():
        ex_result = model(example_input)

    # Train model
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/run_{current_time}")
    semseg_class_id_idx = {
        v: k for k, v in utl_cons.get_semseg_image_last_dim_map().items()
    }.get("CLASS_ID")

    lowest_validation_epoch_loss = float("inf")
    epochs_without_improvement = 0

    for epoch_i in range(args.max_epochs):
        # Training loop
        training_batch_losses = []
        for batch_j, data in enumerate(train_data_iterator):
            inputs = data["range_image"]
            targets = data["segmentation_labels"][:, semseg_class_id_idx, :, :]

            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record batch loss
            training_batch_losses.append(loss.item())

        # Record epoch loss (average of batch losses)
        train_epoch_loss = np.mean(training_batch_losses)
        writer.add_scalar(
            tag="Loss/train_epoch",
            scalar_value=train_epoch_loss,
            global_step=epoch_i,
        )

        # Validation loop
        with torch.no_grad():
            validation_batch_losses = []
            for batch_j, data in enumerate(validation_data_iterator):
                inputs = data["range_image"]
                targets = data["segmentation_labels"][:, semseg_class_id_idx, :, :]

                # Forward pass
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                validation_batch_losses.append(loss.item())

            # Record validation loss (average of batch losses)
            validation_epoch_loss = np.mean(validation_batch_losses)
            writer.add_scalar(
                tag="Loss/validation_epoch",
                scalar_value=validation_epoch_loss,
                global_step=epoch_i,
            )

            # Check for early stopping
            if validation_epoch_loss < lowest_validation_epoch_loss:
                lowest_validation_epoch_loss = validation_epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stopping_epochs:
                    logging.info(
                        f"Early stopping triggered after {epochs_without_improvement} epochs without improvement."
                        + f"\nLowest validation loss: {lowest_validation_epoch_loss:.4f} at epoch {epoch_i - epochs_without_improvement}"
                    )
                    break

## ARCHIVE
## --------------------------------------------------------------------
# # Create PyTorch dataset and dataloaders
# train_data = RangeImageDataset(train_file_obs, args.data_dir)
# validation_data = RangeImageDataset(validation_file_obs, args.data_dir)
# test_data = RangeImageDataset(test_file_obs, args.data_dir)

# train_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
# validation_dl = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True)
# test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# n_total_obs = sum([len(x) for x in labeled_file_obs.values()])
# logging.info(
#     f"\n# obs in Training set: {len(train_data)} / {n_total_obs}"+
#     f"\n# obs in Validation set: {len(validation_data)} / {n_total_obs}" +
#     f"\n# obs in Test set: {len(test_data)} / {n_total_obs}"
# )
