"""ML model training"""

interactive_mode = False
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


def _compute_confusion_metrics(
    confusion_matrix: torch.Tensor,
    n_valid_pixels: int,
    epsilon: float = 1e-8,
) -> dict[str, float | torch.Tensor]:
    """Compute per-class and aggregate segmentation metrics from a confusion matrix."""
    conf = confusion_matrix.to(torch.float32)
    true_positives = torch.diag(conf)
    false_positives = conf.sum(dim=0) - true_positives
    false_negatives = conf.sum(dim=1) - true_positives
    true_negatives = conf.sum() - (
        true_positives + false_positives + false_negatives
    )

    per_class_iou = true_positives / (
        true_positives + false_positives + false_negatives + epsilon
    )
    per_class_precision = true_positives / (true_positives + false_positives + epsilon)
    per_class_recall = true_positives / (true_positives + false_negatives + epsilon)

    mean_iou = float(per_class_iou.mean().item())
    mean_precision = float(per_class_precision.mean().item())
    mean_recall = float(per_class_recall.mean().item())
    pixel_accuracy = float(
        true_positives.sum().item() / max(float(n_valid_pixels), 1.0)
    )

    return {
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "tn": true_negatives,
        "per_class_iou": per_class_iou,
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
    }


def _update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    n_classes: int,
) -> int:
    """Accumulate confusion counts for one batch and return number of valid pixels."""
    target_flat = targets.reshape(-1).to(torch.long)
    prediction_flat = predictions.reshape(-1).to(torch.long)
    valid_mask = (target_flat >= 0) & (target_flat < n_classes)
    if not torch.any(valid_mask):
        return 0

    target_valid = target_flat[valid_mask]
    prediction_valid = prediction_flat[valid_mask]
    bincount = torch.bincount(
        n_classes * target_valid + prediction_valid,
        minlength=n_classes * n_classes,
    )
    confusion_matrix += bincount.reshape(n_classes, n_classes)
    return int(valid_mask.sum().item())


def _compute_gradient_norm_l2(model: nn.Module) -> float:
    """Compute global L2 norm of parameter gradients."""
    grad_norm_sq_sum = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_grad_norm = param.grad.detach().norm(2).item()
            grad_norm_sq_sum += param_grad_norm**2
    return grad_norm_sq_sum**0.5


def _run_epoch(
    model: nn.Module,
    data_iterator: object,
    loss_func: nn.Module,
    device: torch.device,
    n_classes: int,
    semseg_class_id_idx: int,
    is_training: bool,
    optimizer: Adam | None = None,
    track_confusion: bool = True,
    loop_name: str = "epoch",
) -> dict[str, float | int | dict[str, float | torch.Tensor] | None]:
    """Run one training or evaluation epoch over a Ray torch-batch iterator."""
    if is_training and optimizer is None:
        raise ValueError("Optimizer is required when running a training epoch.")

    model.train(mode=is_training)

    batch_losses: list[float] = []
    batch_count = 0
    sample_count = 0
    batch_grad_norms: list[float] = []
    confusion_matrix = torch.zeros((n_classes, n_classes), dtype=torch.long, device=device)
    n_valid_pixels = 0

    loop_start_time = time.perf_counter()
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for data in data_iterator:
            inputs = data["range_image"].to(device)
            targets = data["segmentation_labels"][:, semseg_class_id_idx, :, :].to(
                device
            )

            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                batch_grad_norms.append(_compute_gradient_norm_l2(model))
                optimizer.step()

            if track_confusion:
                predictions = torch.argmax(outputs, dim=1)
                n_valid_pixels += _update_confusion_matrix(
                    confusion_matrix=confusion_matrix,
                    targets=targets,
                    predictions=predictions,
                    n_classes=n_classes,
                )

            batch_losses.append(loss.item())
            batch_count += 1
            sample_count += int(inputs.shape[0])

    loop_duration_sec = time.perf_counter() - loop_start_time
    if not batch_losses:
        raise RuntimeError(
            f"{loop_name.capitalize()} iterator yielded zero batches. "
            "Check dataset construction and split sizes."
        )

    epoch_loss = float(np.mean(batch_losses))
    mean_grad_norm = (
        float(np.mean(batch_grad_norms)) if is_training and batch_grad_norms else None
    )
    metrics = (
        _compute_confusion_metrics(
            confusion_matrix=confusion_matrix,
            n_valid_pixels=n_valid_pixels,
        )
        if track_confusion
        else None
    )

    return {
        "loss": epoch_loss,
        "batch_count": batch_count,
        "sample_count": sample_count,
        "duration_sec": float(loop_duration_sec),
        "mean_grad_norm": mean_grad_norm,
        "metrics": metrics,
    }


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

    # Get example data for setting model input dims
    example_data = next(iter(train_data.iter_torch_batches(batch_size=args.batch_size)))
    ## TODO: check if this is causing Ray to load all data
    example_input, example_target = (
        example_data["range_image"],
        example_data["segmentation_labels"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model, optimizer, and loss function
    model = BaselineCNN(
        example_input,
        n_classes=len(utl_cons.get_semseg_idx_map()),
        hidden_channels=[128, 256, 512, 1024],
        avgpool_layers=[(1, 5), (1, 5), (1, 2)],
        verbose=True,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss().to(device)

    # Test forward pass
    with torch.no_grad():
        ex_result = model(example_input.to(device))

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
    if semseg_class_id_idx is None:
        raise ValueError(
            "Could not find CLASS_ID channel index in semantic segmentation labels."
        )

    run_metadata = {
        "run_id": f"run_{current_time}",
        "seed": run_seed,
        "compute_train_confusion_metrics": bool(
            getattr(args, "compute_train_confusion_metrics", True)
        ),
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
    best_validation_loss_epoch = -1
    epochs_without_improvement = 0
    best_validation_miou = float("-inf")
    best_epoch = -1
    best_checkpoint_path = None
    test_epoch_loss = float("nan")
    test_miou = float("nan")
    test_pixel_accuracy = float("nan")
    compute_train_confusion_metrics = bool(
        getattr(args, "compute_train_confusion_metrics", True)
    )

    for epoch_i in range(args.max_epochs):
        epoch_start_time = time.perf_counter()
        train_data_iterator = train_data.iter_torch_batches(batch_size=args.batch_size)
        train_result = _run_epoch(
            model=model,
            data_iterator=train_data_iterator,
            loss_func=loss_func,
            device=device,
            n_classes=n_classes,
            semseg_class_id_idx=semseg_class_id_idx,
            is_training=True,
            optimizer=optimizer,
            track_confusion=compute_train_confusion_metrics,
            loop_name="training",
        )

        # Record train epoch telemetry
        train_epoch_loss = float(train_result["loss"])
        train_epoch_grad_norm = float(train_result["mean_grad_norm"])
        train_batch_count = int(train_result["batch_count"])
        train_sample_count = int(train_result["sample_count"])
        train_loop_duration_sec = float(train_result["duration_sec"])
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
        if compute_train_confusion_metrics and train_result["metrics"] is not None:
            train_metrics = train_result["metrics"]
            writer.add_scalar(
                "Metrics/train_mIoU",
                float(train_metrics["mean_iou"]),
                epoch_i,
            )
            writer.add_scalar(
                "Metrics/train_pixel_accuracy",
                float(train_metrics["pixel_accuracy"]),
                epoch_i,
            )
            writer.add_scalar(
                "Metrics/train_mean_precision",
                float(train_metrics["mean_precision"]),
                epoch_i,
            )
            writer.add_scalar(
                "Metrics/train_mean_recall",
                float(train_metrics["mean_recall"]),
                epoch_i,
            )

        # Validation loop
        validation_data_iterator = validation_data.iter_torch_batches(
            batch_size=args.batch_size
        )
        validation_result = _run_epoch(
            model=model,
            data_iterator=validation_data_iterator,
            loss_func=loss_func,
            device=device,
            n_classes=n_classes,
            semseg_class_id_idx=semseg_class_id_idx,
            is_training=False,
            optimizer=None,
            track_confusion=True,
            loop_name="validation",
        )

        validation_epoch_loss = float(validation_result["loss"])
        validation_loop_duration_sec = float(validation_result["duration_sec"])
        validation_metrics = validation_result["metrics"]
        if validation_metrics is None:
            raise RuntimeError("Validation metrics were not computed.")

        if validation_epoch_loss < lowest_validation_epoch_loss:
            lowest_validation_epoch_loss = float(validation_epoch_loss)
            best_validation_loss_epoch = epoch_i
        writer.add_scalar(
            tag="Loss/validation_epoch",
            scalar_value=validation_epoch_loss,
            global_step=epoch_i,
        )

        per_class_iou = validation_metrics["per_class_iou"]
        mean_iou = float(validation_metrics["mean_iou"])
        pixel_accuracy = float(validation_metrics["pixel_accuracy"])

        writer.add_scalar("Metrics/validation_mIoU", mean_iou, epoch_i)
        writer.add_scalar(
            "Metrics/validation_pixel_accuracy", pixel_accuracy, epoch_i
        )
        writer.add_scalar(
            "Metrics/validation_mean_precision",
            float(validation_metrics["mean_precision"]),
            epoch_i,
        )
        writer.add_scalar(
            "Metrics/validation_mean_recall",
            float(validation_metrics["mean_recall"]),
            epoch_i,
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
        writer.add_scalar("Performance/epoch_duration_sec", epoch_duration_sec, epoch_i)
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
                    + f"\nLowest validation loss: {lowest_validation_epoch_loss:.4f} at epoch {best_validation_loss_epoch}"
                )
                break

    # Evaluate best checkpoint on held-out test set
    if best_checkpoint_path is not None:
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
        test_data_iterator = test_data.iter_torch_batches(batch_size=args.batch_size)
        test_result = _run_epoch(
            model=model,
            data_iterator=test_data_iterator,
            loss_func=loss_func,
            device=device,
            n_classes=n_classes,
            semseg_class_id_idx=semseg_class_id_idx,
            is_training=False,
            optimizer=None,
            track_confusion=True,
            loop_name="test",
        )
        test_metrics = test_result["metrics"]
        if test_metrics is None:
            raise RuntimeError("Test metrics were not computed.")

        test_epoch_loss = float(test_result["loss"])
        test_miou = float(test_metrics["mean_iou"])
        test_pixel_accuracy = float(test_metrics["pixel_accuracy"])

        writer.add_scalar("Loss/test_epoch", test_epoch_loss, global_step=0)
        writer.add_scalar("Metrics/test_mIoU", test_miou, global_step=0)
        writer.add_scalar(
            "Metrics/test_pixel_accuracy", test_pixel_accuracy, global_step=0
        )
        writer.add_scalar(
            "Metrics/test_mean_precision",
            float(test_metrics["mean_precision"]),
            global_step=0,
        )
        writer.add_scalar(
            "Metrics/test_mean_recall",
            float(test_metrics["mean_recall"]),
            global_step=0,
        )

        logging.info(
            "Test metrics for best checkpoint: "
            + f"loss={test_epoch_loss:.4f}, "
            + f"mIoU={test_miou:.4f}, "
            + f"pixel_accuracy={test_pixel_accuracy:.4f}"
        )

    run_summary = {
        "run_id": f"run_{current_time}",
        "best_epoch": int(best_epoch),
        "best_validation_loss_epoch": int(best_validation_loss_epoch),
        "best_validation_loss": float(lowest_validation_epoch_loss),
        "best_validation_mIoU": float(best_validation_miou),
        "best_checkpoint_path": best_checkpoint_path,
        "test_loss": float(test_epoch_loss),
        "test_mIoU": float(test_miou),
        "test_pixel_accuracy": float(test_pixel_accuracy),
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
