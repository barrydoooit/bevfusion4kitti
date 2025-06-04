import torch
import torch.nn as nn
from typing import Optional
from . import voxel_layer

BEVFUSION_CONFIG = {
    "min_range": [-54.0, -54.0, -5.0],
    "max_range": [54.0, 54.0, 3.0],
    "voxel_size": [0.075, 0.075, 0.2],
    "max_points_per_voxel": 10,
    "max_points": 300000,
    "max_voxels": 160000,
    "num_feature": 5,
}
#   voxelization.grid_size =
#       voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);

class VoxelizationParameterWrapper:
    def __init__(self, cfg: dict):
        self.vx_param = voxel_layer.VoxelizationParameter()
        self.vx_param.min_range = voxel_layer.Float3(cfg["min_range"][0], cfg["min_range"][1], cfg["min_range"][2])
        self.vx_param.max_range = voxel_layer.Float3(cfg["max_range"][0], cfg["max_range"][1], cfg["max_range"][2])
        self.vx_param.voxel_size = voxel_layer.Float3(cfg["voxel_size"][0], cfg["voxel_size"][1], cfg["voxel_size"][2])
        self.vx_param.grid_size = voxel_layer.VoxelizationParameter.compute_grid_size(self.vx_param.max_range, self.vx_param.min_range, self.vx_param.voxel_size)
        self.vx_param.max_points_per_voxel = cfg["max_points_per_voxel"]
        self.vx_param.max_points = cfg["max_points"]
        self.vx_param.max_voxels = cfg["max_voxels"]
        self.vx_param.num_feature = cfg["num_feature"]
    
    def unwrap(self):
        return self.vx_param

BEVFUSION_VXPARAM_WRAPPER = VoxelizationParameterWrapper(BEVFUSION_CONFIG)


class VoxelizationWrapper(nn.Module):
    INDEX_DIM = 4
    def __init__(self, vx_param_wrapper: Optional[VoxelizationParameterWrapper]=None):
        super(VoxelizationWrapper, self).__init__()
        if vx_param_wrapper is None:
            vx_param_wrapper = BEVFUSION_VXPARAM_WRAPPER
            print("Using BEVFusion default voxelization parameters")
        self.vx = voxel_layer.create_voxelization(vx_param_wrapper.unwrap())
    
    def unwrap(self):
        return self.vx
    
    def forward(self, points: torch.Tensor):
        if not points.is_contiguous():
            points = points.contiguous()
        self.vx.forward(points, points.size(0))
        return self._features(), self._indices()

    def _features(self):
        num_voxels = self.vx.num_voxels()
        # print("num_voxels:", num_voxels)
        voxel_dim = self.vx.voxel_dim()
        # print("voxel_dim:", voxel_dim)
        features_ptr = self.vx.features()
        features_tensor = voxel_layer.create_feat_tensor_from_raw_ptr(features_ptr, num_voxels, voxel_dim)
        return features_tensor

    def _indices(self):
        num_voxels = self.vx.num_voxels()
        indices_ptr = self.vx.indices()
        indices_tensor = voxel_layer.create_indices_tensor_from_raw_ptr(indices_ptr, num_voxels, self.INDEX_DIM)
        return indices_tensor


if __name__ == "__main__":
    points = torch.randn(200000, 5).to(torch.float16).to("cuda")
    vs = VoxelizationWrapper()
    import time
    time_start = time.time()
    feats, coords = vs(points)
    print("Time elapsed:", time.time() - time_start)