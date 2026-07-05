"""General utility functions"""

import os
from collections.abc import Callable

from google.cloud import storage as gcs

# import tensorflow
# import tensorflow.compat.v1 as tf
# tf.enable_eager_execution()

# from waymo_open_dataset import dataset_pb2 as wod


def _gcs_listdir(gcs_path: str, client: gcs.Client | None = None) -> list[str]:
    """List immediate subdirectory/file names under a GCS path."""
    # Parse bucket and prefix from gs://bucket/prefix
    path = gcs_path[len("gs://") :]
    bucket_name, _, prefix = path.partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    if client is None:
        client = gcs.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
    # Consume blobs to populate prefixes
    names = [os.path.basename(b.name) for b in blobs if not b.name.endswith("/")]
    prefixes = [p.rstrip("/").split("/")[-1] for p in (blobs.prefixes or [])]
    return names + prefixes


def get_ids_of_complete_data_files(
    data_dir: str, gcs_client: gcs.Client | None = None
) -> list[str]:
    """Get file ids for which data are available in each data subdirectory."""
    file_ids = set()
    listdir: Callable[[str], list[str]]
    join: Callable[[str, str], str]
    if data_dir.startswith("gs://"):

        def listdir(path: str) -> list[str]:
            return _gcs_listdir(path, client=gcs_client)

        def join(base: str, sub: str) -> str:
            return base.rstrip("/") + "/" + sub
    else:

        def listdir(path: str) -> list[str]:
            return list(os.listdir(path))

        def join(base: str, sub: str) -> str:
            return os.path.join(base, sub)

    for i, data_subdir in enumerate(listdir(data_dir)):
        data_subdir_ids = set(
            [
                x.split(".")[0]
                for x in listdir(join(data_dir, data_subdir))
                if x.endswith(".parquet")
            ]
        )
        if i == 0:
            file_ids.update(data_subdir_ids)
        else:
            file_ids.intersection_update(data_subdir_ids)
    return sorted(list(file_ids))


## ------------------------------------------
## Utility functions for TFRecord data format
## ------------------------------------------
# def load_TFRecord(
#     datadir: str,
#     filename: str,
# ) -> tf.data.Dataset:
#     """Load TFRecord dataset."""
#     return tf.data.TFRecordDataset(
#         os.path.join(datadir, filename),
#         compression_type="",
#     )

# def extract_frames_from_TFRecord(
#     dataset: tf.data.Dataset,
#     max_n_frames: Optional[int] = None,
# ) -> List[wod.Frame]:
#     """Extract frames (sequences) from TFRecord dataset."""
#     # Validate max_n_frames arg
#     if max_n_frames is not None:
#         if (max_n_frames <= 0) or (not isinstance(max_n_frames, int)):
#             raise ValueError(
#                 f"max_n_frames argument ({max_n_frames}) must be positive integer"
#             )
#     # Extract frames
#     frames = []
#     for data in dataset:
#         frame = wod.Frame()
#         frame.ParseFromString(bytearray(data.numpy()))
#         frames.append(frame)
#         if (max_n_frames is not None) and (len(frames) >= max_n_frames):
#             break
#     return frames
