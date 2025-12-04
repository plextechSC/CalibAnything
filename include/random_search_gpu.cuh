/*
 * GPU-Accelerated Random Search
 * CUDA implementation for batch evaluation and optimization
 */

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <eigen3/Eigen/Core>
#include <vector>
#include <memory>

#include "cuda_utils.cuh"
#include "calibration_gpu.cuh"

namespace search_cuda {

// Random search optimizer
class RandomSearchGPU {
public:
    RandomSearchGPU();
    ~RandomSearchGPU();

    // Initialize random search with GPU score calculator
    bool Initialize(
        calib_cuda::CalScoreGPU* score_calculator,
        int num_files
    );

    // Perform batch random search
    // Evaluates multiple transformation candidates in parallel
    bool BatchSearch(
        int batch_size,
        float xyz_range,
        float rpy_range,
        const Eigen::Matrix4f& current_transform,
        int file_index,
        Eigen::Matrix<double, 6, 1>& best_var,
        double& best_score
    );

    // Check if initialized
    bool IsInitialized() const { return initialized_; }

private:
    bool initialized_;
    int device_id_;

    // cuRAND state
    curandState* d_rand_states_;
    int num_rand_states_;

    // Batch evaluation workspace
    float* d_transform_samples_;    // [batch_size * 16] transformation matrices
    double* d_scores_;              // [batch_size] scores
    int* d_best_idx_;              // [1] index of best score
    double* d_best_score_;         // [1] best score value

    // Pinned host memory
    double* h_scores_pinned_;
    float* h_transforms_pinned_;

    // CUDA streams for concurrent execution
    cudaStream_t* streams_;
    int num_streams_;

    // Reference to score calculator
    calib_cuda::CalScoreGPU* score_calculator_;
    int num_files_;
};

} // namespace search_cuda

// CUDA kernel declarations

// Initialize cuRAND states for random number generation
__global__ void InitRandStatesKernel(
    curandState* states,
    int num_states,
    unsigned long long seed
);

// Generate random transformation samples
__global__ void GenerateRandomTransformsKernel(
    curandState* rand_states,
    int batch_size,
    float xyz_range,
    float rpy_range,
    const float* base_transform,
    float* output_transforms,
    float* output_deltas
);

// Find minimum score and its index using parallel reduction
__global__ void FindMinScoreKernel(
    const double* scores,
    int num_scores,
    int* best_idx,
    double* best_score
);

// Warp-level reduction for finding minimum
__device__ void WarpReduceMin(volatile double* sdata, volatile int* sidx, int tid);

// Block-level reduction for finding minimum
__device__ void BlockReduceMin(double* sdata, int* sidx, int tid);

#endif // USE_CUDA
