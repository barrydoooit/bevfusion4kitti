#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lidar-voxelization.hpp"

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
namespace py = pybind11;

void forward_wrapper(bevfusion::lidar::Voxelization& self, torch::Tensor points, int num_points) {
    // ensure the tensor is of the expected type and on the GPU
    TORCH_CHECK(points.dtype() == torch::kFloat16, "Tensor must be of type torch.float16");
    TORCH_CHECK(points.is_cuda(), "Tensor must be on CUDA device");

    // cast data_ptr() to nvtype::half*
    auto ptr = reinterpret_cast<nvtype::half*>(points.data_ptr());

    // call the actual implementation
    self.forward(ptr, num_points);
}

torch::Tensor create_feat_tensor_from_raw_ptr(void* ptr, int64_t num_voxels, int64_t voxel_dim) {
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA).requires_grad(false);
    auto tensor = torch::from_blob(ptr, {num_voxels, voxel_dim}, options);
    return tensor;
}

torch::Tensor create_indices_tensor_from_raw_ptr(void* ptr, int64_t num_voxels, int64_t indices_dim) {
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false);
    auto tensor = torch::from_blob(ptr, {num_voxels, indices_dim}, options);
    return tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<nvtype::Float3>(m, "Float3")
        .def(py::init<float, float, float>());
    
    py::class_<nvtype::Int3>(m, "Int3")
        .def(py::init<int, int, int>());
    
    py::class_<bevfusion::lidar::VoxelizationParameter>(m, "VoxelizationParameter")
        .def(py::init<>())
        .def_readwrite("min_range", &bevfusion::lidar::VoxelizationParameter::min_range)
        .def_readwrite("max_range", &bevfusion::lidar::VoxelizationParameter::max_range)
        .def_readwrite("voxel_size", &bevfusion::lidar::VoxelizationParameter::voxel_size)
        .def_readwrite("grid_size", &bevfusion::lidar::VoxelizationParameter::grid_size)
        .def_readwrite("num_feature", &bevfusion::lidar::VoxelizationParameter::num_feature)
        .def_readwrite("max_voxels", &bevfusion::lidar::VoxelizationParameter::max_voxels)
        .def_readwrite("max_points_per_voxel", &bevfusion::lidar::VoxelizationParameter::max_points_per_voxel)
        .def_readwrite("max_points", &bevfusion::lidar::VoxelizationParameter::max_points)
        .def_static("compute_grid_size", &bevfusion::lidar::VoxelizationParameter::compute_grid_size);

    py::class_<bevfusion::lidar::Voxelization, std::shared_ptr<bevfusion::lidar::Voxelization>>(m, "Voxelization")
        .def("forward", &forward_wrapper)
        .def("num_voxels", &bevfusion::lidar::Voxelization::num_voxels)
        .def("voxel_dim", &bevfusion::lidar::Voxelization::voxel_dim)
        .def("features", &bevfusion::lidar::Voxelization::features, py::return_value_policy::reference)
        .def("indices", &bevfusion::lidar::Voxelization::indices, py::return_value_policy::reference);
        

    m.def("create_voxelization", &bevfusion::lidar::create_voxelization, "Create voxelization object with given parameters",
          py::arg("param"));
    m.def("create_feat_tensor_from_raw_ptr", &create_feat_tensor_from_raw_ptr, "Create a tensor from a raw memory pointer");
    m.def("create_indices_tensor_from_raw_ptr", &create_indices_tensor_from_raw_ptr, "Create a tensor from a raw memory pointer");
    

}