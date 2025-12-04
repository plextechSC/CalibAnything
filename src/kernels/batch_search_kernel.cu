/*
 * CUDA Kernels: Batch Random Search
 * Random transformation generation and parallel score reduction
 */

#include "random_search_gpu.cuh"
#include <float.h>

/**
 * Initialize cuRAND states for each thread
 * Each thread gets its own random number generator state
 *
 * @param states        Array of cuRAND states
 * @param num_states    Number of states to initialize
 * @param seed          Random seed
 */
__global__ void InitRandStatesKernel(
    curandState* states,
    int num_states,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_states) {
        return;
    }

    // Initialize state with unique seed per thread
    curand_init(seed, idx, 0, &states[idx]);
}

/**
 * Generate random transformation matrices
 * Each thread generates one random transformation (delta from base)
 *
 * Delta transformation format: [roll, pitch, yaw, x, y, z]
 * Ranges: rotation in radians, translation in meters
 *
 * @param rand_states       cuRAND states for random number generation
 * @param batch_size        Number of transformations to generate
 * @param xyz_range         Translation range [-xyz_range, xyz_range]
 * @param rpy_range         Rotation range in radians [-rpy_range, rpy_range]
 * @param base_transform    Base 4x4 transformation matrix
 * @param output_transforms Output: generated 4x4 matrices [batch_size * 16]
 * @param output_deltas     Output: delta parameters [batch_size * 6]
 */
__global__ void GenerateRandomTransformsKernel(
    curandState* rand_states,
    int batch_size,
    float xyz_range,
    float rpy_range,
    const float* base_transform,
    float* output_transforms,
    float* output_deltas
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= batch_size) {
        return;
    }

    // Get random state for this thread
    curandState local_state = rand_states[idx];

    // Generate random deltas
    float roll = curand_uniform(&local_state) * 2.0f * rpy_range - rpy_range;
    float pitch = curand_uniform(&local_state) * 2.0f * rpy_range - rpy_range;
    float yaw = curand_uniform(&local_state) * 2.0f * rpy_range - rpy_range;
    float dx = curand_uniform(&local_state) * 2.0f * xyz_range - xyz_range;
    float dy = curand_uniform(&local_state) * 2.0f * xyz_range - xyz_range;
    float dz = curand_uniform(&local_state) * 2.0f * xyz_range - xyz_range;

    // Store deltas
    output_deltas[idx * 6 + 0] = roll;
    output_deltas[idx * 6 + 1] = pitch;
    output_deltas[idx * 6 + 2] = yaw;
    output_deltas[idx * 6 + 3] = dx;
    output_deltas[idx * 6 + 4] = dy;
    output_deltas[idx * 6 + 5] = dz;

    // Update random state
    rand_states[idx] = local_state;

    // Compute delta transformation matrix
    // R = Rz(yaw) * Ry(pitch) * Rx(roll)
    float cr = cosf(roll), sr = sinf(roll);
    float cp = cosf(pitch), sp = sinf(pitch);
    float cy = cosf(yaw), sy = sinf(yaw);

    // Combined rotation matrix
    float R[9];
    R[0] = cy * cp;
    R[1] = cy * sp * sr - sy * cr;
    R[2] = cy * sp * cr + sy * sr;
    R[3] = sy * cp;
    R[4] = sy * sp * sr + cy * cr;
    R[5] = sy * sp * cr - cy * sr;
    R[6] = -sp;
    R[7] = cp * sr;
    R[8] = cp * cr;

    // Build 4x4 delta matrix
    float delta_T[16];
    delta_T[0] = R[0];  delta_T[1] = R[1];  delta_T[2] = R[2];  delta_T[3] = dx;
    delta_T[4] = R[3];  delta_T[5] = R[4];  delta_T[6] = R[5];  delta_T[7] = dy;
    delta_T[8] = R[6];  delta_T[9] = R[7];  delta_T[10] = R[8]; delta_T[11] = dz;
    delta_T[12] = 0.0f; delta_T[13] = 0.0f; delta_T[14] = 0.0f; delta_T[15] = 1.0f;

    // Multiply: output = base * delta
    float* out_T = &output_transforms[idx * 16];

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            float sum = 0.0f;
            for (int k = 0; k < 4; k++) {
                sum += base_transform[i * 4 + k] * delta_T[k * 4 + j];
            }
            out_T[i * 4 + j] = sum;
        }
    }
}

/**
 * Warp-level reduction to find minimum value and its index
 * Uses warp shuffle instructions for efficient reduction
 */
__device__ void WarpReduceMin(double& val, int& idx, int tid) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        double other_val = __shfl_down_sync(0xffffffff, val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, idx, offset);

        if (other_val < val) {
            val = other_val;
            idx = other_idx;
        }
    }
}

/**
 * Block-level reduction to find minimum value and its index
 * Uses shared memory and warp reduction
 */
__device__ void BlockReduceMin(double* sdata_val, int* sdata_idx, int tid) {
    __syncthreads();

    // Warp reduction
    if (tid < warpSize) {
        double val = sdata_val[tid];
        int idx = sdata_idx[tid];
        WarpReduceMin(val, idx, tid);
        sdata_val[tid] = val;
        sdata_idx[tid] = idx;
    }
}

/**
 * Find minimum score and its index using parallel reduction
 *
 * @param scores        Array of scores [num_scores]
 * @param num_scores    Number of scores
 * @param best_idx      Output: index of minimum score [1]
 * @param best_score    Output: minimum score value [1]
 */
__global__ void FindMinScoreKernel(
    const double* scores,
    int num_scores,
    int* best_idx,
    double* best_score
) {
    extern __shared__ char shared_mem[];
    double* sdata_val = (double*)shared_mem;
    int* sdata_idx = (int*)&sdata_val[blockDim.x];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    double val = (idx < num_scores) ? scores[idx] : DBL_MAX;
    int best_id = (idx < num_scores) ? idx : -1;

    sdata_val[tid] = val;
    sdata_idx[tid] = best_id;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > warpSize; s >>= 1) {
        if (tid < s) {
            if (sdata_val[tid + s] < sdata_val[tid]) {
                sdata_val[tid] = sdata_val[tid + s];
                sdata_idx[tid] = sdata_idx[tid + s];
            }
        }
        __syncthreads();
    }

    // Final warp reduction
    if (tid < warpSize) {
        double final_val = sdata_val[tid];
        int final_idx = sdata_idx[tid];

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            double other_val = __shfl_down_sync(0xffffffff, final_val, offset);
            int other_idx = __shfl_down_sync(0xffffffff, final_idx, offset);

            if (other_val < final_val) {
                final_val = other_val;
                final_idx = other_idx;
            }
        }

        // Thread 0 of each block writes to global memory
        if (tid == 0) {
            atomicMin((unsigned long long*)best_score, __double_as_longlong(final_val));

            // Check if this block found the global minimum
            if (final_val == *best_score) {
                *best_idx = final_idx;
            }
        }
    }
}
