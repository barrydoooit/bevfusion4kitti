# Training BEVFusion with KITTI Dataset

## 1. Train Lidar-Only model
```
bash tools/dist_train.sh config/with_kitti/bevfusion_lidar_voxel0075_second_secfpn_4xb4-20e_kitti-3d.py 4
```

## 2. Train Camera-Only Model
```
bash tools/dist_train.sh config/with_kitti/bevfusion_cam_resnet50_4xb6-20e_kitti-3d.py 4
```

## 3. Train Fusion Model
### Merge the single-modal checkpoints
```
python tools/merge_checkpoints.py --ckpt1 work_dirs/bevfusion_lidar_voxel0075_second_secfpn_4xb4-20e_kitti-3d/epoch_20.pth --ckpt2 work_dirs/bevfusion_cam_resnet50_4xb6-20e_kitti-3d/epoch_40.pth --outfile checkpoints/k01s3_k02cs1.pth --exclude fusion_layer
```
###  Train the multi-modal model
```
bash tools/dist_train.sh config/with_kitti/bevfusion_lidar-cam_resnet50_voxel0075_secfpn_4xb4-40e_kitti-3d.py 4
```
