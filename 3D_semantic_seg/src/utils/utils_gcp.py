"""Utility functions for Google Cloud Platform (GCP) resources"""

from google.cloud import storage


def connect_to_gcp_storage(project: str) -> storage.Client | None:
    """Connect to Google Cloud Storage and return the client."""
    try:
        client = storage.Client(project=project)
        return client
    except Exception as e:
        print(f"Error connecting to Google Cloud Storage: {e}")
        return None


def connect_to_gcp_bucket(
    client: storage.Client, bucket_name: str
) -> storage.Bucket | None:
    """Connect to a specific GCP bucket and return the bucket object."""
    try:
        bucket = client.bucket(bucket_name)
        return bucket
    except Exception as e:
        print(f"Error connecting to GCP bucket '{bucket_name}': {e}")
        return None
