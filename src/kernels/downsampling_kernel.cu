/*
 * CUDA Kernel: Voxel Grid Downsampling
 * Hash-based voxel downsampling for point clouds
 */

#include "pointcloud_gpu.cuh"

/**
 * Computes voxel hash index for each point
 * Uses 3D grid hash: hash = x + y * grid_x + z * grid_x * grid_y
 *
 * @param points_x      X coordinates
 * @param points_y      Y coordinates
 * @param points_z      Z coordinates
 * @param num_points    Number of points
 * @param voxel_size    Voxel size in meters
 * @param min_bound     Minimum bound of point cloud
 * @param grid_size     Grid dimensions
 * @param voxel_indices Output: voxel index for each point
 */
__global__ void ComputeVoxelIndicesKernel(
    const float* points_x,
    const float* points_y,
    const float* points_z,
    int num_points,
    float voxel_size,
    float3 min_bound,
    int3 grid_size,
    int* voxel_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) {
        return;
    }

    // Load point
    float x = points_x[idx];
    float y = points_y[idx];
    float z = points_z[idx];

    // Compute voxel coordinates
    int vx = __float2int_rd((x - min_bound.x) / voxel_size);
    int vy = __float2int_rd((y - min_bound.y) / voxel_size);
    int vz = __float2int_rd((z - min_bound.z) / voxel_size);

    // Clamp to grid bounds
    vx = max(0, min(vx, grid_size.x - 1));
    vy = max(0, min(vy, grid_size.y - 1));
    vz = max(0, min(vz, grid_size.z - 1));

    // Compute linear voxel index
    int voxel_idx = vx + vy * grid_size.x + vz * grid_size.x * grid_size.y;
    voxel_indices[idx] = voxel_idx;
}

/**
 * Aggregates points within each voxel to compute centroid
 * Assumes points are sorted by voxel index
 *
 * @param points_x          Input X coordinates
 * @param points_y          Input Y coordinates
 * @param points_z          Input Z coordinates
 * @param intensities       Input intensities
 * @param voxel_indices     Voxel index for each point (sorted)
 * @param sorted_indices    Original point indices after sorting
 * @param num_points        Number of input points
 * @param num_voxels        Number of occupied voxels
 * @param output_x          Output X coordinates (one per voxel)
 * @param output_y          Output Y coordinates
 * @param output_z          Output Z coordinates
 * @param output_intensity  Output intensities
 */
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
) {
    int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (voxel_id >= num_voxels) {
        return;
    }

    // Find the start and end of this voxel's points
    // Binary search for first point with this voxel_id
    int start = 0;
    int end = num_points;

    // Find first occurrence
    while (start < end) {
        int mid = (start + end) / 2;
        if (voxel_indices[sorted_indices[mid]] < voxel_id) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    int voxel_start = start;

    // Find last occurrence
    start = voxel_start;
    end = num_points;

    while (start < end) {
        int mid = (start + end) / 2;
        if (voxel_indices[sorted_indices[mid]] <= voxel_id) {
            start = mid + 1;
        } else {
            end = mid;
        }
    }

    int voxel_end = start;

    // Compute centroid
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_z = 0.0f;
    float sum_intensity = 0.0f;
    int count = 0;

    for (int i = voxel_start; i < voxel_end; i++) {
        int point_idx = sorted_indices[i];
        if (voxel_indices[point_idx] == voxel_id) {
            sum_x += points_x[point_idx];
            sum_y += points_y[point_idx];
            sum_z += points_z[point_idx];
            sum_intensity += intensities[point_idx];
            count++;
        }
    }

    // Write averaged result
    if (count > 0) {
        output_x[voxel_id] = sum_x / count;
        output_y[voxel_id] = sum_y / count;
        output_z[voxel_id] = sum_z / count;
        output_intensity[voxel_id] = sum_intensity / count;
    }
}

/**
 * Alternative simpler approach using atomic operations
 * Each thread processes one point and atomically accumulates to voxel
 */
__global__ void AggregateVoxelsAtomicKernel(
    const float* points_x,
    const float* points_y,
    const float* points_z,
    const float* intensities,
    const int* voxel_indices,
    int num_points,
    float* voxel_sums_x,
    float* voxel_sums_y,
    float* voxel_sums_z,
    float* voxel_sums_intensity,
    int* voxel_counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) {
        return;
    }

    int voxel_idx = voxel_indices[idx];

    // Atomically accumulate to voxel
    atomicAdd(&voxel_sums_x[voxel_idx], points_x[idx]);
    atomicAdd(&voxel_sums_y[voxel_idx], points_y[idx]);
    atomicAdd(&voxel_sums_z[voxel_idx], points_z[idx]);
    atomicAdd(&voxel_sums_intensity[voxel_idx], intensities[idx]);
    atomicAdd(&voxel_counts[voxel_idx], 1);
}

/**
 * Computes voxel centroids from accumulated sums
 */
__global__ void ComputeCentroidsKernel(
    const float* voxel_sums_x,
    const float* voxel_sums_y,
    const float* voxel_sums_z,
    const float* voxel_sums_intensity,
    const int* voxel_counts,
    int num_voxels,
    float* output_x,
    float* output_y,
    float* output_z,
    float* output_intensity,
    bool* voxel_valid
) {
    int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (voxel_id >= num_voxels) {
        return;
    }

    int count = voxel_counts[voxel_id];

    if (count > 0) {
        output_x[voxel_id] = voxel_sums_x[voxel_id] / count;
        output_y[voxel_id] = voxel_sums_y[voxel_id] / count;
        output_z[voxel_id] = voxel_sums_z[voxel_id] / count;
        output_intensity[voxel_id] = voxel_sums_intensity[voxel_id] / count;
        voxel_valid[voxel_id] = true;
    } else {
        voxel_valid[voxel_id] = false;
    }
}
