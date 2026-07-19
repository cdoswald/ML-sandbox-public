"""ML model training"""

interactive_mode = True
if interactive_mode:
    import os

    os.chdir("/workspace/hostfiles/src")
    print(os.getcwd())

from datetime import datetime  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
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

    # Seed RNG for reproducibility
    run_seed = 9999
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

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

    ## Model Training
    # Record run metadata
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(f"runs/run_{current_time}")
    model_subdir = os.path.join(args.models_dir, f"run_{current_time}")
    os.makedirs(model_subdir, exist_ok=True)
    semseg_idx_map = utl_cons.get_semseg_idx_map()
    n_classes = len(semseg_idx_map)
    semseg_name_map = {
        idx: cls_info["name"] for idx, cls_info in semseg_idx_map.items()
    }
    semseg_class_id_idx = {
        v: k for k, v in utl_cons.get_semseg_image_last_dim_map().items()
    }.get("CLASS_ID")

    run_metadata = {
        "run_id": f"run_{current_time}",
        "seed": run_seed,
        "hyperparameters": {
            "max_epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "early_stopping_epochs": args.early_stopping_epochs,
            "train_num_cpus": args.train_num_cpus,
        },
        "data_split": {
            "total_file_ids": int(n_file_ids),
            "train_file_ids": int(len(train_file_ids)),
            "validation_file_ids": int(len(validation_file_ids)),
            "test_file_ids": int(len(test_file_ids)),
            "validation_share": float(args.validation_share),
            "test_share": float(args.test_share),
        },
        "model": {
            "name": model.__class__.__name__,
            "n_classes": int(n_classes),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        },
    }
    writer.add_text("Run/metadata", json.dumps(run_metadata, indent=2), global_step=0)
    with open(
        os.path.join(model_subdir, "run_metadata.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(run_metadata, f, indent=2)

    lowest_validation_epoch_loss = float("inf")
    epochs_without_improvement = 0
    best_validation_miou = float("-inf")
    best_epoch = -1
    best_checkpoint_path = None

    for epoch_i in range(args.max_epochs):
        epoch_start_time = time.perf_counter()

        # Training loop
        training_batch_losses = []
        training_batch_grad_norms = []
        train_batch_count = 0
        train_sample_count = 0
        train_loop_start_time = time.perf_counter()
        for batch_j, data in enumerate(train_data_iterator):
            inputs = data["range_image"]
            targets = data["segmentation_labels"][:, semseg_class_id_idx, :, :]

            # Forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Record gradient norm to monitor optimization stability
            grad_norm_sq_sum = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.detach().norm(2).item()
                    grad_norm_sq_sum += param_grad_norm**2
            training_batch_grad_norms.append(grad_norm_sq_sum**0.5)

            optimizer.step()

            # Record batch loss
            training_batch_losses.append(loss.item())
            train_batch_count += 1
            train_sample_count += int(inputs.shape[0])

        train_loop_duration_sec = time.perf_counter() - train_loop_start_time

        # Record epoch loss (average of batch losses)
        train_epoch_loss = np.mean(training_batch_losses)
        train_epoch_grad_norm = np.mean(training_batch_grad_norms)
        current_lr = optimizer.param_groups[0]["lr"]
        train_batches_per_second = train_batch_count / max(
            train_loop_duration_sec, 1e-8
        )
        train_samples_per_second = train_sample_count / max(
            train_loop_duration_sec, 1e-8
        )

        writer.add_scalar(
            tag="Loss/train_epoch",
            scalar_value=train_epoch_loss,
            global_step=epoch_i,
        )
        writer.add_scalar(
            tag="Optimization/learning_rate",
            scalar_value=current_lr,
            global_step=epoch_i,
        )
        writer.add_scalar(
            tag="Optimization/gradient_norm_l2_train_epoch",
            scalar_value=train_epoch_grad_norm,
            global_step=epoch_i,
        )
        writer.add_scalar(
            tag="Performance/train_batches_per_sec",
            scalar_value=train_batches_per_second,
            global_step=epoch_i,
        )
        writer.add_scalar(
            tag="Performance/train_samples_per_sec",
            scalar_value=train_samples_per_second,
            global_step=epoch_i,
        )

        # Validation loop
        with torch.no_grad():
            validation_batch_losses = []
            val_loop_start_time = time.perf_counter()
            confusion_matrix = torch.zeros((n_classes, n_classes), dtype=torch.long)
            n_valid_pixels = 0
            for batch_j, data in enumerate(validation_data_iterator):
                inputs = data["range_image"]
                targets = data["segmentation_labels"][:, semseg_class_id_idx, :, :]

                # Forward pass
                outputs = model(inputs)
                loss = loss_func(outputs, targets)
                validation_batch_losses.append(loss.item())

                # Build confusion matrix for IoU calculations
                predictions = torch.argmax(outputs, dim=1)
                target_flat = targets.reshape(-1).to(torch.long)
                prediction_flat = predictions.reshape(-1).to(torch.long)
                valid_mask = (target_flat >= 0) & (target_flat < n_classes)
                if torch.any(valid_mask):
                    target_valid = target_flat[valid_mask]
                    prediction_valid = prediction_flat[valid_mask]
                    bincount = torch.bincount(
                        n_classes * target_valid + prediction_valid,
                        minlength=n_classes * n_classes,
                    )
                    confusion_matrix += bincount.reshape(n_classes, n_classes).cpu()
                    n_valid_pixels += int(valid_mask.sum().item())

            validation_loop_duration_sec = time.perf_counter() - val_loop_start_time

            # Record validation loss (average of batch losses)
            validation_epoch_loss = np.mean(validation_batch_losses)
            writer.add_scalar(
                tag="Loss/validation_epoch",
                scalar_value=validation_epoch_loss,
                global_step=epoch_i,
            )

            # Segmentation quality telemetry
            true_positives = torch.diag(confusion_matrix).to(torch.float32)
            false_positives = (
                confusion_matrix.sum(dim=0).to(torch.float32) - true_positives
            )
            false_negatives = (
                confusion_matrix.sum(dim=1).to(torch.float32) - true_positives
            )
            per_class_iou = true_positives / (
                true_positives + false_positives + false_negatives + 1e-8
            )
            mean_iou = float(per_class_iou.mean().item())
            pixel_accuracy = float(
                true_positives.sum().item() / max(float(n_valid_pixels), 1.0)
            )

            writer.add_scalar("Metrics/validation_mIoU", mean_iou, epoch_i)
            writer.add_scalar(
                "Metrics/validation_pixel_accuracy", pixel_accuracy, epoch_i
            )
            for class_idx in range(n_classes):
                class_name = semseg_name_map.get(class_idx, f"class_{class_idx}")
                class_tag = class_name.replace(" ", "_").lower()
                writer.add_scalar(
                    f"Metrics/validation_iou_per_class/{class_idx}_{class_tag}",
                    float(per_class_iou[class_idx].item()),
                    epoch_i,
                )

            # Runtime and memory telemetry
            epoch_duration_sec = time.perf_counter() - epoch_start_time
            writer.add_scalar(
                "Performance/epoch_duration_sec", epoch_duration_sec, epoch_i
            )
            writer.add_scalar(
                "Performance/validation_duration_sec",
                validation_loop_duration_sec,
                epoch_i,
            )
            if torch.cuda.is_available():
                writer.add_scalar(
                    "System/gpu_memory_allocated_mb",
                    torch.cuda.memory_allocated() / (1024**2),
                    epoch_i,
                )
                writer.add_scalar(
                    "System/gpu_memory_reserved_mb",
                    torch.cuda.memory_reserved() / (1024**2),
                    epoch_i,
                )

            # Check for early stopping
            if mean_iou > best_validation_miou:
                best_validation_miou = mean_iou
                best_epoch = epoch_i
                epochs_without_improvement = 0

                # Checkpoint model
                checkpoint_path = os.path.join(
                    model_subdir, f"model_checkpoint_epoch_{epoch_i}.pt"
                )
                torch.save(model.state_dict(), checkpoint_path)
                best_checkpoint_path = checkpoint_path
                logging.info(
                    f"Checkpoint saved at epoch {epoch_i} with validation loss {validation_epoch_loss:.4f}"
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stopping_epochs:
                    logging.info(
                        f"Early stopping triggered after {epochs_without_improvement} epochs without improvement."
                        + f"\nLowest validation loss: {lowest_validation_epoch_loss:.4f} at epoch {epoch_i - epochs_without_improvement}"
                    )
                    break

    run_summary = {
        "run_id": f"run_{current_time}",
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(lowest_validation_epoch_loss),
        "best_validation_mIoU": float(best_validation_miou),
        "best_checkpoint_path": best_checkpoint_path,
    }
    writer.add_text("Run/summary", json.dumps(run_summary, indent=2), global_step=0)
    with open(
        os.path.join(model_subdir, "run_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(run_summary, f, indent=2)
    logging.info(f"Run summary: {json.dumps(run_summary)}")

    writer.flush()
    writer.close()

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
