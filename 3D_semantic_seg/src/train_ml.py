"""ML model training"""

interactive_mode = True
if interactive_mode:
    import os
    os.chdir("/workspace/hostfiles/src")
    print(os.getcwd())

import os                                   #noqa: E402
import numpy as np                          #noqa: E402

from config import Config
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
    # only about 30-50 of the approx. 200 timestep observations in each file for the 
    # top-mounted laser have labels)
    labeled_file_obs: dict[str, list[str]] = {}
    for file_id in file_ids:
        lidar_segment_table_all = utl.load_parquet_data(args.data_dir, file_id, "lidar_segmentation")
        labeled_obs_ids = lidar_segment_table_all["index"].combine_chunks().to_pylist()
        labeled_file_obs[file_id] = labeled_obs_ids
 
    # Create PyTorch dataset and dataloader



    # # plot range image
    # new_shape = lidar_image_table["[LiDARComponent].range_image_return1.shape"].combine_chunks().to_pylist()[0]
    # temp = np.array(
    #     lidar_image_table["[LiDARComponent].range_image_return1.values"].combine_chunks().to_pylist()[0]
    # ).reshape(new_shape)
    # plt.imshow(temp[..., 0])
    