/*
 * GPU-Accelerated Point Cloud Processing
 * CUDA implementation for point filtering and downsampling
 */

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Core>

#include "cuda_utils.cuh"

// Forward declaration
struct PointXYZINS;

namespace pointcloud_cuda {

// GPU data for voxel downsampling
struct VoxelGridGPU {
    float* d_points_x;
    float* d_points_y;
    float* d_points_z;
    float* d_intensities;
    int* d_voxel_indices;
    int* d_point_counts_per_voxel;

    int num_points;
    int num_voxels;
    float voxel_size;

    VoxelGridGPU() : d_points_x(nullptr), d_points_y(nullptr), d_points_z(nullptr),
                     d_intensities(nullptr), d_voxel_indices(nullptr),
                     d_point_counts_per_voxel(nullptr),
                     num_points(0), num_voxels(0), voxel_size(0.1f) {}
};

// Point cloud processor class
class PointCloudProcessorGPU {
public:
    PointCloudProcessorGPU();
    ~PointCloudProcessorGPU();

    // Filter points based on projection bounds
    bool FilterPoints(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud,
        const Eigen::Matrix4f& extrinsic,
        const Eigen::MatrixXf& intrinsic,
        int img_width,
        int img_height,
        int margin,
        const std::vector<Eigen::Matrix4f>& search_transforms
    );

    // Voxel grid downsampling
    bool VoxelDownsample(
        const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
        pcl::PointCloud<pcl::PointXYZI>::Ptr& output_cloud,
        float voxel_size
    );

    // Check if GPU processor is initialized
    bool IsInitialized() const { return initialized_; }

private:
    bool initialized_;
    int device_id_;

    // Workspace memory
    float4* d_input_points_;
    bool* d_valid_flags_;
    int* d_valid_indices_;
    int* d_scan_output_;

    size_t workspace_capacity_;
};

} // namespace pointcloud_cuda

// CUDA kernel declarations

// Filters points based on projection validity
__global__ void FilterPointsByProjectionKernel(
    const float4* input_points,
    int num_points,
    const float* transforms,
    int num_transforms,
    const float* intrinsic,
    int intrinsic_size,
    int img_width,
    int img_height,
    int margin,
    bool* valid_flags
);

// Computes voxel indices for each point
__global__ void ComputeVoxelIndicesKernel(
    const float* points_x,
    const float* points_y,
    const float* points_z,
    int num_points,
    float voxel_size,
    float3 min_bound,
    int3 grid_size,
    int* voxel_indices
);

// Aggregates points within each voxel
__global__ void AggregateVoxelsKernel(
    const float* points_x,
    const float* points_y,
    const float* points_z,
    const float* intensities,
    const int* voxel_indices,
    const int* sorted_indices,
    int num_points,
    int num_voxels,
    float* output_x,
    float* output_y,
    float* output_z,
    float* output_intensity
);

// Parallel prefix sum (scan) for stream compaction
__global__ void ScanKernel(
    const bool* input,
    int* output,
    int num_elements
);

// Compacts valid points using scan results
__global__ void CompactKernel(
    const float4* input_points,
    const bool* valid_flags,
    const int* scan_output,
    int num_points,
    float4* output_points
);

#endif // USE_CUDA
