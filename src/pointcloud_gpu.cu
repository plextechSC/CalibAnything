/*
 * GPU-Accelerated Point Cloud Processing Implementation
 * Host-side code for point filtering and downsampling
 */

#include "pointcloud_gpu.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <iostream>
#include <algorithm>

namespace pointcloud_cuda {

PointCloudProcessorGPU::PointCloudProcessorGPU()
    : initialized_(false),
      device_id_(0),
      d_input_points_(nullptr),
      d_valid_flags_(nullptr),
      d_valid_indices_(nullptr),
      d_scan_output_(nullptr),
      workspace_capacity_(0) {
}

PointCloudProcessorGPU::~PointCloudProcessorGPU() {
    cuda_utils::FreeDeviceMemory(d_input_points_);
    cuda_utils::FreeDeviceMemory(d_valid_flags_);
    cuda_utils::FreeDeviceMemory(d_valid_indices_);
    cuda_utils::FreeDeviceMemory(d_scan_output_);
}

bool PointCloudProcessorGPU::FilterPoints(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud,
    const Eigen::Matrix4f& extrinsic,
    const Eigen::MatrixXf& intrinsic,
    int img_width,
    int img_height,
    int margin,
    const std::vector<Eigen::Matrix4f>& search_transforms
) {
    int num_points = input_cloud->size();
    if (num_points == 0) {
        output_cloud->clear();
        return true;
    }

    // Allocate/resize workspace if needed
    if (num_points > workspace_capacity_) {
        cuda_utils::FreeDeviceMemory(d_input_points_);
        cuda_utils::FreeDeviceMemory(d_valid_flags_);
        cuda_utils::FreeDeviceMemory(d_valid_indices_);
        cuda_utils::FreeDeviceMemory(d_scan_output_);

        workspace_capacity_ = num_points * 1.5;  // 50% extra capacity

        if (!cuda_utils::AllocateDeviceMemory(&d_input_points_, workspace_capacity_) ||
            !cuda_utils::AllocateDeviceMemory(&d_valid_flags_, workspace_capacity_) ||
            !cuda_utils::AllocateDeviceMemory(&d_valid_indices_, workspace_capacity_) ||
            !cuda_utils::AllocateDeviceMemory(&d_scan_output_, workspace_capacity_)) {
            return false;
        }
    }

    // Prepare input points on host
    std::vector<float4> h_points(num_points);
    for (int i = 0; i < num_points; i++) {
        const auto& pt = (*input_cloud)[i];
        h_points[i] = make_float4(pt.x, pt.y, pt.z, pt.intensity);
    }

    // Upload to device
    if (!cuda_utils::CopyHostToDevice(d_input_points_, h_points.data(), num_points)) {
        return false;
    }

    // Prepare transformation matrices
    std::vector<Eigen::Matrix4f> all_transforms;
    all_transforms.push_back(extrinsic);
    all_transforms.insert(all_transforms.end(), search_transforms.begin(), search_transforms.end());

    std::vector<float> h_transforms;
    for (const auto& T : all_transforms) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                h_transforms.push_back(T(i, j));
            }
        }
    }

    float* d_transforms = nullptr;
    if (!cuda_utils::AllocateDeviceMemory(&d_transforms, h_transforms.size()) ||
        !cuda_utils::CopyHostToDevice(d_transforms, h_transforms.data(), h_transforms.size())) {
        return false;
    }

    // Prepare intrinsic matrix
    int intrinsic_size = (intrinsic.cols() == 3) ? 9 : 12;
    std::vector<float> h_intrinsic;
    for (int i = 0; i < intrinsic.rows(); i++) {
        for (int j = 0; j < intrinsic.cols(); j++) {
            h_intrinsic.push_back(intrinsic(i, j));
        }
    }

    float* d_intrinsic = nullptr;
    if (!cuda_utils::AllocateDeviceMemory(&d_intrinsic, intrinsic_size) ||
        !cuda_utils::CopyHostToDevice(d_intrinsic, h_intrinsic.data(), intrinsic_size)) {
        cuda_utils::FreeDeviceMemory(d_transforms);
        return false;
    }

    // Launch filtering kernel
    int block_size, grid_size;
    cuda_utils::CalculateLaunchConfig(num_points, grid_size, block_size);

    FilterPointsByProjectionKernel<<<grid_size, block_size>>>(
        d_input_points_,
        num_points,
        d_transforms,
        all_transforms.size(),
        d_intrinsic,
        intrinsic_size,
        img_width,
        img_height,
        margin,
        d_valid_flags_
    );
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();

    // Count valid points using Thrust
    thrust::device_ptr<bool> valid_ptr(d_valid_flags_);
    int num_valid = thrust::count(thrust::device, valid_ptr, valid_ptr + num_points, true);

    std::cout << "GPU Filtering: " << num_valid << " / " << num_points
              << " points passed" << std::endl;

    // Use Thrust for stream compaction (simpler than custom scan)
    thrust::device_ptr<float4> input_ptr(d_input_points_);
    thrust::device_ptr<float4> output_ptr(d_input_points_);  // In-place is fine

    auto new_end = thrust::copy_if(
        thrust::device,
        input_ptr,
        input_ptr + num_points,
        valid_ptr,
        output_ptr,
        thrust::identity<bool>()
    );

    // Download filtered points
    std::vector<float4> h_filtered(num_valid);
    if (!cuda_utils::CopyDeviceToHost(h_filtered.data(), d_input_points_, num_valid)) {
        cuda_utils::FreeDeviceMemory(d_transforms);
        cuda_utils::FreeDeviceMemory(d_intrinsic);
        return false;
    }

    // Convert back to PCL format
    output_cloud->clear();
    output_cloud->reserve(num_valid);
    for (int i = 0; i < num_valid; i++) {
        pcl::PointXYZI pt;
        pt.x = h_filtered[i].x;
        pt.y = h_filtered[i].y;
        pt.z = h_filtered[i].z;
        pt.intensity = h_filtered[i].w;
        output_cloud->push_back(pt);
    }

    // Cleanup
    cuda_utils::FreeDeviceMemory(d_transforms);
    cuda_utils::FreeDeviceMemory(d_intrinsic);

    return true;
}

bool PointCloudProcessorGPU::VoxelDownsample(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud,
    float voxel_size
) {
    int num_points = input_cloud->size();
    if (num_points == 0) {
        output_cloud->clear();
        return true;
    }

    // Compute bounding box
    float3 min_bound = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 max_bound = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (const auto& pt : input_cloud->points) {
        min_bound.x = fmin(min_bound.x, pt.x);
        min_bound.y = fmin(min_bound.y, pt.y);
        min_bound.z = fmin(min_bound.z, pt.z);
        max_bound.x = fmax(max_bound.x, pt.x);
        max_bound.y = fmax(max_bound.y, pt.y);
        max_bound.z = fmax(max_bound.z, pt.z);
    }

    // Compute grid size
    int3 grid_size;
    grid_size.x = (int)ceil((max_bound.x - min_bound.x) / voxel_size) + 1;
    grid_size.y = (int)ceil((max_bound.y - min_bound.y) / voxel_size) + 1;
    grid_size.z = (int)ceil((max_bound.z - min_bound.z) / voxel_size) + 1;
    int num_voxels = grid_size.x * grid_size.y * grid_size.z;

    std::cout << "GPU Downsampling: " << num_points << " points into "
              << grid_size.x << "x" << grid_size.y << "x" << grid_size.z
              << " voxel grid" << std::endl;

    // Prepare point data (SoA layout)
    std::vector<float> h_x(num_points), h_y(num_points), h_z(num_points), h_intensity(num_points);
    for (int i = 0; i < num_points; i++) {
        const auto& pt = (*input_cloud)[i];
        h_x[i] = pt.x;
        h_y[i] = pt.y;
        h_z[i] = pt.z;
        h_intensity[i] = pt.intensity;
    }

    // Allocate and upload
    float *d_x, *d_y, *d_z, *d_intensity;
    int* d_voxel_indices;

    if (!cuda_utils::AllocateDeviceMemory(&d_x, num_points) ||
        !cuda_utils::AllocateDeviceMemory(&d_y, num_points) ||
        !cuda_utils::AllocateDeviceMemory(&d_z, num_points) ||
        !cuda_utils::AllocateDeviceMemory(&d_intensity, num_points) ||
        !cuda_utils::AllocateDeviceMemory(&d_voxel_indices, num_points)) {
        return false;
    }

    if (!cuda_utils::CopyHostToDevice(d_x, h_x.data(), num_points) ||
        !cuda_utils::CopyHostToDevice(d_y, h_y.data(), num_points) ||
        !cuda_utils::CopyHostToDevice(d_z, h_z.data(), num_points) ||
        !cuda_utils::CopyHostToDevice(d_intensity, h_intensity.data(), num_points)) {
        return false;
    }

    // Compute voxel indices
    int block_size, grid_size_1d;
    cuda_utils::CalculateLaunchConfig(num_points, grid_size_1d, block_size);

    ComputeVoxelIndicesKernel<<<grid_size_1d, block_size>>>(
        d_x, d_y, d_z,
        num_points,
        voxel_size,
        min_bound,
        grid_size,
        d_voxel_indices
    );
    CUDA_CHECK_KERNEL();

    // Use atomic aggregation (simpler than sort+reduce)
    float *d_voxel_sums_x, *d_voxel_sums_y, *d_voxel_sums_z, *d_voxel_sums_intensity;
    int* d_voxel_counts;

    if (!cuda_utils::AllocateDeviceMemory(&d_voxel_sums_x, num_voxels) ||
        !cuda_utils::AllocateDeviceMemory(&d_voxel_sums_y, num_voxels) ||
        !cuda_utils::AllocateDeviceMemory(&d_voxel_sums_z, num_voxels) ||
        !cuda_utils::AllocateDeviceMemory(&d_voxel_sums_intensity, num_voxels) ||
        !cuda_utils::AllocateDeviceMemory(&d_voxel_counts, num_voxels)) {
        return false;
    }

    // Zero buffers
    cuda_utils::ZeroDeviceMemory(d_voxel_sums_x, num_voxels);
    cuda_utils::ZeroDeviceMemory(d_voxel_sums_y, num_voxels);
    cuda_utils::ZeroDeviceMemory(d_voxel_sums_z, num_voxels);
    cuda_utils::ZeroDeviceMemory(d_voxel_sums_intensity, num_voxels);
    cuda_utils::ZeroDeviceMemory(d_voxel_counts, num_voxels);

    // Aggregate
    AggregateVoxelsAtomicKernel<<<grid_size_1d, block_size>>>(
        d_x, d_y, d_z, d_intensity,
        d_voxel_indices,
        num_points,
        d_voxel_sums_x,
        d_voxel_sums_y,
        d_voxel_sums_z,
        d_voxel_sums_intensity,
        d_voxel_counts
    );
    CUDA_CHECK_KERNEL();

    // Compute centroids
    float *d_out_x, *d_out_y, *d_out_z, *d_out_intensity;
    bool* d_voxel_valid;

    int voxel_grid_size, voxel_block_size;
    cuda_utils::CalculateLaunchConfig(num_voxels, voxel_grid_size, voxel_block_size);

    if (!cuda_utils::AllocateDeviceMemory(&d_out_x, num_voxels) ||
        !cuda_utils::AllocateDeviceMemory(&d_out_y, num_voxels) ||
        !cuda_utils::AllocateDeviceMemory(&d_out_z, num_voxels) ||
        !cuda_utils::AllocateDeviceMemory(&d_out_intensity, num_voxels) ||
        !cuda_utils::AllocateDeviceMemory(&d_voxel_valid, num_voxels)) {
        return false;
    }

    ComputeCentroidsKernel<<<voxel_grid_size, voxel_block_size>>>(
        d_voxel_sums_x,
        d_voxel_sums_y,
        d_voxel_sums_z,
        d_voxel_sums_intensity,
        d_voxel_counts,
        num_voxels,
        d_out_x,
        d_out_y,
        d_out_z,
        d_out_intensity,
        d_voxel_valid
    );
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();

    // Count valid voxels
    thrust::device_ptr<bool> valid_ptr(d_voxel_valid);
    int num_valid = thrust::count(thrust::device, valid_ptr, valid_ptr + num_voxels, true);

    std::cout << "GPU Downsampling: " << num_valid << " voxels occupied" << std::endl;

    // Download results
    std::vector<float> h_out_x(num_voxels);
    std::vector<float> h_out_y(num_voxels);
    std::vector<float> h_out_z(num_voxels);
    std::vector<float> h_out_intensity(num_voxels);
    std::vector<bool> h_voxel_valid(num_voxels);

    cuda_utils::CopyDeviceToHost(h_out_x.data(), d_out_x, num_voxels);
    cuda_utils::CopyDeviceToHost(h_out_y.data(), d_out_y, num_voxels);
    cuda_utils::CopyDeviceToHost(h_out_z.data(), d_out_z, num_voxels);
    cuda_utils::CopyDeviceToHost(h_out_intensity.data(), d_out_intensity, num_voxels);
    cuda_utils::CopyDeviceToHost(h_voxel_valid.data(), d_voxel_valid, num_voxels);

    // Convert to PCL
    output_cloud->clear();
    output_cloud->reserve(num_valid);
    for (int i = 0; i < num_voxels; i++) {
        if (h_voxel_valid[i]) {
            pcl::PointXYZI pt;
            pt.x = h_out_x[i];
            pt.y = h_out_y[i];
            pt.z = h_out_z[i];
            pt.intensity = h_out_intensity[i];
            output_cloud->push_back(pt);
        }
    }

    // Cleanup
    cuda_utils::FreeDeviceMemory(d_x);
    cuda_utils::FreeDeviceMemory(d_y);
    cuda_utils::FreeDeviceMemory(d_z);
    cuda_utils::FreeDeviceMemory(d_intensity);
    cuda_utils::FreeDeviceMemory(d_voxel_indices);
    cuda_utils::FreeDeviceMemory(d_voxel_sums_x);
    cuda_utils::FreeDeviceMemory(d_voxel_sums_y);
    cuda_utils::FreeDeviceMemory(d_voxel_sums_z);
    cuda_utils::FreeDeviceMemory(d_voxel_sums_intensity);
    cuda_utils::FreeDeviceMemory(d_voxel_counts);
    cuda_utils::FreeDeviceMemory(d_out_x);
    cuda_utils::FreeDeviceMemory(d_out_y);
    cuda_utils::FreeDeviceMemory(d_out_z);
    cuda_utils::FreeDeviceMemory(d_out_intensity);
    cuda_utils::FreeDeviceMemory(d_voxel_valid);

    return true;
}

} // namespace pointcloud_cuda
