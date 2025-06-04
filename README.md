## Tuning BEVFusion for KITTI Dataset

> ⚠️ **Warning:** These files were clipped from one of my old projects. They may not be directly runnable. Please take care when merging them into your own mmdet3d-based projects.
> 
---

### Some hints and thoughts

- There are three things to do (If I remember correctly): Modify a loading function, fix a bug in transfusion head (originally carelessness that mess up the width and length of the heatmap because nuScenes usually has a square heatmap), write the config files.
- The sparse encoder seems not accurate in describing the output dimensions (just my understanding).
- My suggestion is, don't spend too much time on testing BEVFusion on KITTI; I suppose it is not designed for this kind of dataset and the results can rank 100+ on KITTI leaderboard. What might be the cause: low resolution in images; further x axis (front and back) range that overwhelms the LSS; lack of consecutive LiDAR frames.


### What might be useful
BTW, under `edge_plugins.ops`, there is a super-fast CUDA voxelization implementation from NVIDIA : [Lidar AI Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution). I once made a simple encapsulation so that it can be used in mmdet3d (like the lines commentted in bevfusion.py). Sadly, it only support fp16.

To use, just add the following to the setup.py:
````python
make_cuda_ext(
    name="voxel_layer",
    module="bevfusion.edge_plugins.ops.voxel",
    sources=[
        'src/voxelization-pybind.cpp',
        'src/lidar-voxelization.cu',
    ],
)