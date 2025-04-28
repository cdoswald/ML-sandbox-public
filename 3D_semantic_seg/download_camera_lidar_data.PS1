## Download Waymo open dataset from GCP to local system
$download_to_local_dir = "data"
if (-not (Test-Path -Path $download_to_local_dir)) {
	New-Item -ItemType Directory -Path $download_to_local_dir
}

# Partial folder download (due to large file size)
$partial_download_file_prefix = "11"
$partial_download_folders = @(
	"gs://waymo_open_dataset_v_2_0_1/training/camera_image",
	"gs://waymo_open_dataset_v_2_0_1/training/camera_segmentation",
	"gs://waymo_open_dataset_v_2_0_1/training/lidar_camera_projection",
	"gs://waymo_open_dataset_v_2_0_1/training/lidar_pose",
	"gs://waymo_open_dataset_v_2_0_1/training/lidar"
)

foreach ($partial_folder in $partial_download_folders) {
	Write-Host "Downloading $partial_folder..."
	gsutil -m cp -r "$partial_folder/$partial_download_file_prefix*" $download_to_local_dir
}

# Full folder download
$full_download_folders = @(
	"gs://waymo_open_dataset_v_2_0_1/training/camera_box",
	"gs://waymo_open_dataset_v_2_0_1/training/camera_calibration",
	"gs://waymo_open_dataset_v_2_0_1/training/camera_hkp",
	"gs://waymo_open_dataset_v_2_0_1/training/camera_to_lidar_box_association",
	"gs://waymo_open_dataset_v_2_0_1/training/lidar_box",
	"gs://waymo_open_dataset_v_2_0_1/training/lidar_calibration",
	"gs://waymo_open_dataset_v_2_0_1/training/lidar_camera_synced_box",
	"gs://waymo_open_dataset_v_2_0_1/training/hkp",
	"gs://waymo_open_dataset_v_2_0_1/training/lidar_segmentation",
	"gs://waymo_open_dataset_v_2_0_1/training/projected_lidar_box",
	"gs://waymo_open_dataset_v_2_0_1/training/stats",
	"gs://waymo_open_dataset_v_2_0_1/training/vehicle_pose"
)

foreach ($full_folder in $full_download_folders) {
	Write-Host "Downloading $full_folder..."
	gsutil -m cp -r $full_folder $download_to_local_dir
}