/*
 * CUDA Kernel: Point Cloud Filtering
 * Filters points based on projection validity with multiple transforms
 */

#include "pointcloud_gpu.cuh"

/**
 * Filters points by checking if they project within image bounds
 * Tests each point against multiple transformation matrices
 * A point is valid if it projects successfully with ANY transform
 *
 * @param input_points      Input point cloud (x, y, z, intensity)
 * @param num_points        Number of points
 * @param transforms        Array of transformation matrices [num_transforms * 16]
 * @param num_transforms    Number of transforms to test
 * @param intrinsic         Intrinsic matrix (3x3 or 3x4)
 * @param intrinsic_size    9 for 3x3, 12 for 3x4
 * @param img_width         Image width
 * @param img_height        Image height
 * @param margin            Projection margin in pixels
 * @param valid_flags       Output: true if point is valid
 */
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) {
        return;
    }

    // Initialize as invalid
    valid_flags[idx] = false;

    // Load point
    float4 pt = input_points[idx];

    // Check if point is finite
    if (!isfinite(pt.x) || !isfinite(pt.y) || !isfinite(pt.z)) {
        return;
    }

    // Test against each transformation
    for (int t = 0; t < num_transforms; t++) {
        const float* T = &transforms[t * 16];

        // Apply transformation
        float cam_x = T[0] * pt.x + T[1] * pt.y + T[2] * pt.z + T[3];
        float cam_y = T[4] * pt.x + T[5] * pt.y + T[6] * pt.z + T[7];
        float cam_z = T[8] * pt.x + T[9] * pt.y + T[10] * pt.z + T[11];

        // Check if in front of camera
        if (cam_z <= 0.0f) {
            continue;
        }

        // Apply intrinsic
        float proj_x, proj_y, proj_z;

        if (intrinsic_size == 9) {
            proj_x = intrinsic[0] * cam_x + intrinsic[1] * cam_y + intrinsic[2] * cam_z;
            proj_y = intrinsic[3] * cam_x + intrinsic[4] * cam_y + intrinsic[5] * cam_z;
            proj_z = intrinsic[6] * cam_x + intrinsic[7] * cam_y + intrinsic[8] * cam_z;
        } else {
            proj_x = intrinsic[0] * cam_x + intrinsic[1] * cam_y + intrinsic[2] * cam_z + intrinsic[3];
            proj_y = intrinsic[4] * cam_x + intrinsic[5] * cam_y + intrinsic[6] * cam_z + intrinsic[7];
            proj_z = intrinsic[8] * cam_x + intrinsic[9] * cam_y + intrinsic[10] * cam_z + intrinsic[11];
        }

        // Perspective division
        if (proj_z <= 0.0f) {
            continue;
        }

        float u = proj_x / proj_z;
        float v = proj_y / proj_z;

        // Check bounds with margin
        int px = __float2int_rn(u);
        int py = __float2int_rn(v);

        if (px >= -margin && px < img_width + margin &&
            py >= -margin && py < img_height + margin) {
            // Point is valid with this transform
            valid_flags[idx] = true;
            return;
        }
    }
}

/**
 * Parallel prefix sum (scan) using Hillis-Steele algorithm
 * Used for stream compaction
 *
 * @param input         Input boolean flags (0 or 1)
 * @param output        Output scan (exclusive prefix sum)
 * @param num_elements  Number of elements
 */
__global__ void ScanKernel(
    const bool* input,
    int* output,
    int num_elements
) {
    extern __shared__ int temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    temp[tid] = (idx < num_elements) ? (int)input[idx] : 0;
    __syncthreads();

    // Up-sweep (reduce) phase
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            temp[index] += temp[index - stride];
        }
        __syncthreads();
    }

    // Down-sweep phase
    if (tid == 0) {
        temp[blockDim.x - 1] = 0;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < blockDim.x) {
            int t = temp[index - stride];
            temp[index - stride] = temp[index];
            temp[index] += t;
        }
        __syncthreads();
    }

    // Write results
    if (idx < num_elements) {
        output[idx] = temp[tid];
    }
}

/**
 * Compacts valid points using scan results
 * Removes invalid points from the point cloud
 *
 * @param input_points  Input point cloud
 * @param valid_flags   Validity flags for each point
 * @param scan_output   Exclusive prefix sum of valid flags
 * @param num_points    Number of input points
 * @param output_points Output compacted point cloud
 */
__global__ void CompactKernel(
    const float4* input_points,
    const bool* valid_flags,
    const int* scan_output,
    int num_points,
    float4* output_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) {
        return;
    }

    // If this point is valid, write it to the output at the compacted position
    if (valid_flags[idx]) {
        int output_idx = scan_output[idx];
        output_points[output_idx] = input_points[idx];
    }
}
