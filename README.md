## Self/Semi-supervised depth estimation code, CEA-Valeo

Mostly based on PackNet-Sfm code ([**3D Packing for Self-Supervised Monocular Depth Estimation (CVPR 2020 oral)**](https://arxiv.org/abs/1905.02693), *Vitor Guizilini, Rares Ambrus, Sudeep Pillai, Allan Raventos and Adrien Gaidon*).

### Install

Tested with PyTorch 1.4.0, Cuda 10.1, OpenMPI 4.0.0, Ubuntu 20.04.
```bash
conda create --name packnet-env
source activate packnet-env
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install -c conda-forge yacs
pip install pycuda
pip install onnxruntime awscli onnx opencv-python-headless pillow-simd
pip install mpi4py
pip install future typing numpy pandas matplotlib jupyter h5py boto3 tqdm termcolor path.py
conda install cython
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
pip install --no-cache-dir git+https://github.com/horovod/horovod.git@65de4c961d1e5ad2828f2f6c4329072834f27661
pip install wandb==0.8.21
export PYTHONPATH="${PYTHONPATH}:/path/to/packnet-sfm/"
conda install scipy=1.6.2

# If you want to visualize 3D reconstructed point clouds:
pip install Pillow
conda install -c open3d-admin open3d=0.10.0
```

### Datasets

This code is mainly intented to be used with the Valeo synced multiview dataset.

### Training
Any training, including fine-tuning, can be done by passing either a `.yaml` config file or a `.ckpt` model checkpoint to [scripts/train.py](./scripts/train.py):

```bash
python3 scripts/train_valeo.py <config.yaml or checkpoint.ckpt>
```

If you pass a config file, training will start from scratch using the parameters in that config file. Example config files are in [configs](./configs).
If you pass instead a `.ckpt` file, training will continue from the current checkpoint state.

Note that it is also possible to define checkpoints within the config file itself. These can be done either individually for the depth and/or pose networks or by defining a checkpoint to the model itself, which includes all sub-networks (setting the model checkpoint will overwrite depth and pose checkpoints). In this case, a new training session will start and the networks will be initialized with the model state in the `.ckpt` file(s). Below we provide the locations in the config file where these checkpoints are defined:

```yaml
checkpoint:
    # Folder where .ckpt files will be saved during training
    filepath: /path/to/where/checkpoints/will/be/saved
model:
    # Checkpoint for the model (depth + pose)
    checkpoint_path: /path/to/model.ckpt
    depth_net:
        # Checkpoint for the depth network
        checkpoint_path: /path/to/depth_net.ckpt
    pose_net:
        # Checkpoint for the pose network
        checkpoint_path: /path/to/pose_net.ckpt
```

Every aspect of the training configuration can be controlled by modifying the yaml config file. This include the model configuration (self-supervised, semi-supervised, loss parameters, etc), depth and pose networks configuration (choice of architecture and different parameters), optimizers and schedulers (learning rates, weight decay, etc), datasets (name, splits, depth types, etc) and much more. For a comprehensive list please refer to [configs/default_config.py](./configs/default_config.py).

A few examples of configuration files are given in `configs/`.

### Evaluation

Similar to the training case, to evaluate a trained model you need to provide a `.ckpt` checkpoint, followed optionally by a `.yaml` config file that overrides the configuration stored in the checkpoint.

```bash
python3 scripts/eval_valeo.py --checkpoint <checkpoint.ckpt> [--config <config.yaml>]
```

You can also directly run inference on a single image or folder:

```bash
python3 scripts/infer_valeo.py --checkpoint <checkpoint.ckpt> --input <image or folder> --output <image or folder> [--image_shape <input shape (h,w)>]
```

### License

The source code is released under the [MIT license](LICENSE.md).

## 3D pointcloud reconstruction and visualization

This script is made to visualize reconstructed 3D pointclouds, along with lidar data.

## Usage

```
usage: python3 scripts/viz3D.py [-h] 
                                [--checkpoints CHECKPOINTS [CHECKPOINTS ...]] 
                                [--input_folders INPUT_FOLDERS [INPUT_FOLDERS ...]]
                                [--input_imgs INPUT_IMGS [INPUT_IMGS ...]] 
                                [--output OUTPUT] 
                                [--image_shape IMAGE_SHAPE [IMAGE_SHAPE ...]] 
                                [--stop]
                                [--load_pred_masks] 
                                [--alpha_mask_semantic ALPHA_MASK_SEMANTIC] 
                                [--remove_close_points_lidar_semantic] 
                                [--print_lidar]
                                [--plot_pov_sequence_first_pic]
                                [--mix_depths] 
                                [--save_visualization]
                                [--clean_with_laplacian] 
                                [--lap_threshold LAP_THRESHOLD]
                                [--remove_neighbors_outliers] 
                                [--remove_neighbors_outliers_std REMOVE_NEIGHBORS_OUTLIERS_STD] 
                                [--voxel_downsample]
                                [--voxel_size_downsampling VOXEL_SIZE_DOWNSAMPLING] 
                                [--colorize_cameras] 
                                [--alpha_colorize_cameras ALPHA_COLORIZE_CAMERAS]



Recalibration tool, for a specific sequence from the Valeo dataset

optional arguments:
  -h, --help            
                        show this help message and exit
  --checkpoints CHECKPOINTS [CHECKPOINTS ...]
                        Checkpoint files (.ckpt), one file per camera
  --input_folders INPUT_FOLDERS [INPUT_FOLDERS ...]
                        Input base folders, one folder per camera
  --input_imgs INPUT_IMGS [INPUT_IMGS ...]
                        Input images, one image per camera
  --output OUTPUT
                        Where to save output
  --image_shape IMAGE_SHAPE [IMAGE_SHAPE ...]
                        Input and output image shape (default: checkpoint's config.datasets.augmentation.image_shape)
  --stop                
                        Whether to stop for visualization
  --load_pred_masks 
                        Display predicted semantic masks (need pre-computation)
  --alpha_mask_semantic ALPHA_MASK_SEMANTIC 
                        Weighting for semantic colors
  --remove_close_points_lidar_semantic 
                        Remove points that are not semantized and close to other semantized points, or points close to lidar
  --print_lidar
                        Display lidar points
  --plot_pov_sequence_first_pic
                        Plot a sequence of moving point of view for the first picture
  --mix_depths 
                        Mixing data into a unified depth map
  --save_visualization
                        Save visualization
  --clean_with_laplacian 
                        Clean data based on laplacian
  --lap_threshold LAP_THRESHOLD
                        Threshold on laplacian
  --remove_neighbors_outliers 
                        Cleaning outliers based on number of neighbors
  --remove_neighbors_outliers_std REMOVE_NEIGHBORS_OUTLIERS_STD 
                        How much standard deviation to keep in neighbors inliers
  --voxel_downsample
                        Downsample voxels
  --voxel_size_downsampling VOXEL_SIZE_DOWNSAMPLING 
                        Voxel size for downsampling (making very dense regions less dense)
  --colorize_cameras 
                        Colorizing each camera with a different color
  --alpha_colorize_cameras ALPHA_COLORIZE_CAMERAS
                        Weighting for colorizing cameras
```

Example:
```
python3 scripts/viz3D.py --checkpoints /home/vbelissen/Downloads/test/config50.ckpt /home/vbelissen/Downloads/test/config50.ckpt /home/vbelissen/Downloads/test/config50.ckpt /home/vbelissen/Downloads/test/config50.ckpt /home/vbelissen/Downloads/test/LR-A-semisup200-ep26.ckpt --input_folders /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/20180716_192137/cam_0 /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/20180716_192137/cam_1 /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/20180716_192137/cam_2 /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/20180716_192137/cam_3 /home/vbelissen/Downloads/test/images_multiview/fisheye/test_sync/20180716_192137/cam_4 --output /home/vbelissen/Downloads/test/results --stop --colorize_cameras --clean_with_laplacian
```


## Photometric recalibration algorithm

This script is made to recalibrate cameras, especially in the case of Valeo data. It should not be too difficult to adapt it to other datasets.

### Usage

```
usage: python3 scripts/recalib.py [-h] 
                                  [--checkpoint CHECKPOINT] 
                                  [--input_folder INPUT_FOLDER] 
                                  [--input_imgs INPUT_IMGS [INPUT_IMGS ...]]
                                  [--image_shape IMAGE_SHAPE [IMAGE_SHAPE ...]] 
                                  [--n_epochs N_EPOCHS] 
                                  [--every_n_files EVERY_N_FILES] 
                                  [--lr LR]
                                  [--scheduler_step_size SCHEDULER_STEP_SIZE] 
                                  [--scheduler_gamma SCHEDULER_GAMMA] 
                                  [--regul_weight_trans REGUL_WEIGHT_TRANS]
                                  [--regul_weight_rot REGUL_WEIGHT_ROT] 
                                  [--regul_weight_overlap REGUL_WEIGHT_OVERLAP] 
                                  [--save_pictures] 
                                  [--save_plots]
                                  [--save_rot_tab] 
                                  [--show_plots] 
                                  [--save_folder SAVE_FOLDER] 
                                  [--frozen_cams_trans FROZEN_CAMS_TRANS [FROZEN_CAMS_TRANS ...]]
                                  [--frozen_cams_rot FROZEN_CAMS_ROT [FROZEN_CAMS_ROT ...]]


Recalibration tool, for a specific sequence from the Valeo dataset

optional arguments:
  -h, --help            
                        show this help message and exit
  --checkpoint CHECKPOINT
                        Checkpoint file (.ckpt)
  --input_folder INPUT_FOLDER [INPUT_FOLDER ...]
                        Input base folder
  --input_imgs INPUT_IMGS [INPUT_IMGS ...]
                        Input images
  --image_shape IMAGE_SHAPE [IMAGE_SHAPE ...]
                        Input and output image shape (default: checkpoint's config.datasets.augmentation.image_shape)
  --n_epochs N_EPOCHS   
                        Number of epochs
  --every_n_files EVERY_N_FILES
                        Step in files if folders are used
  --lr LR               Learning rate
  --scheduler_step_size SCHEDULER_STEP_SIZE
  --scheduler_gamma SCHEDULER_GAMMA
  --regul_weight_trans REGUL_WEIGHT_TRANS
                        Regularization weight for position correction
  --regul_weight_rot REGUL_WEIGHT_ROT
                        Regularization weight for rotation correction
  --regul_weight_overlap REGUL_WEIGHT_OVERLAP
                        Regularization weight for the overlap between cameras
  --save_pictures
                        Whether to save original and reprojected pictures during optimization
  --save_plots
                        Whether to save plots showing evolution of loss and calibration correction
  --save_rot_tab
                        Whether to save rotation correction values in a numpy array
  --show_plots
                        Whether to display plots
  --save_folder SAVE_FOLDER
                        Where to save files
  --frozen_cams_trans FROZEN_CAMS_TRANS [FROZEN_CAMS_TRANS ...]
                        List of frozen cameras in translation
  --frozen_cams_rot FROZEN_CAMS_ROT [FROZEN_CAMS_ROT ...]
                        List of frozen cameras in rotation 

```

Example:
```
python3 scripts/recalib.py --checkpoint /home/vbelissen/Downloads/test/config50.ckpt --input_folder /home/vbelissen/Downloads/test/images_multiview/fisheye/test/20170320_163113 --n_epochs 100 --every_n_files 20 --frozen_cams_trans 0 1 2 3 --save_pictures --regul_weight_rot 0.001
```
