/*
 * GPU-Accelerated Random Search Implementation
 * Batch evaluation and parallel optimization
 */

#include "random_search_gpu.cuh"
#include <iostream>
#include <algorithm>
#include <ctime>

namespace search_cuda {

RandomSearchGPU::RandomSearchGPU()
    : initialized_(false),
      device_id_(0),
      d_rand_states_(nullptr),
      num_rand_states_(0),
      d_transform_samples_(nullptr),
      d_scores_(nullptr),
      d_best_idx_(nullptr),
      d_best_score_(nullptr),
      h_scores_pinned_(nullptr),
      h_transforms_pinned_(nullptr),
      streams_(nullptr),
      num_streams_(4),
      score_calculator_(nullptr),
      num_files_(0) {
}

RandomSearchGPU::~RandomSearchGPU() {
    cuda_utils::FreeDeviceMemory(d_rand_states_);
    cuda_utils::FreeDeviceMemory(d_transform_samples_);
    cuda_utils::FreeDeviceMemory(d_scores_);
    cuda_utils::FreeDeviceMemory(d_best_idx_);
    cuda_utils::FreeDeviceMemory(d_best_score_);

    cuda_utils::FreePinnedMemory(h_scores_pinned_);
    cuda_utils::FreePinnedMemory(h_transforms_pinned_);

    if (streams_) {
        for (int i = 0; i < num_streams_; i++) {
            cudaStreamDestroy(streams_[i]);
        }
        delete[] streams_;
    }
}

bool RandomSearchGPU::Initialize(
    calib_cuda::CalScoreGPU* score_calculator,
    int num_files
) {
    if (!score_calculator || !score_calculator->IsInitialized()) {
        std::cerr << "Score calculator not initialized" << std::endl;
        return false;
    }

    score_calculator_ = score_calculator;
    num_files_ = num_files;

    // Initialize cuRAND states (one per thread)
    num_rand_states_ = 1024;  // Typical batch size
    if (!cuda_utils::AllocateDeviceMemory(&d_rand_states_, num_rand_states_)) {
        return false;
    }

    // Initialize random states
    int block_size, grid_size;
    cuda_utils::CalculateLaunchConfig(num_rand_states_, grid_size, block_size);

    unsigned long long seed = time(NULL);
    InitRandStatesKernel<<<grid_size, block_size>>>(
        d_rand_states_,
        num_rand_states_,
        seed
    );
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();

    // Allocate workspace for batch evaluation
    int max_batch_size = 1024;
    if (!cuda_utils::AllocateDeviceMemory(&d_transform_samples_, max_batch_size * 16) ||
        !cuda_utils::AllocateDeviceMemory(&d_scores_, max_batch_size) ||
        !cuda_utils::AllocateDeviceMemory(&d_best_idx_, 1) ||
        !cuda_utils::AllocateDeviceMemory(&d_best_score_, 1)) {
        return false;
    }

    // Allocate pinned memory for fast transfers
    if (!cuda_utils::AllocatePinnedMemory(&h_scores_pinned_, max_batch_size) ||
        !cuda_utils::AllocatePinnedMemory(&h_transforms_pinned_, max_batch_size * 16)) {
        return false;
    }

    // Create CUDA streams for concurrent execution
    streams_ = new cudaStream_t[num_streams_];
    for (int i = 0; i < num_streams_; i++) {
        if (cudaStreamCreate(&streams_[i]) != cudaSuccess) {
            std::cerr << "Failed to create CUDA stream " << i << std::endl;
            return false;
        }
    }

    initialized_ = true;
    std::cout << "Random search GPU initialized with batch size " << max_batch_size << std::endl;

    return true;
}

bool RandomSearchGPU::BatchSearch(
    int batch_size,
    float xyz_range,
    float rpy_range,
    const Eigen::Matrix4f& current_transform,
    int file_index,
    Eigen::Matrix<double, 6, 1>& best_var,
    double& best_score
) {
    if (!initialized_ || batch_size > num_rand_states_) {
        std::cerr << "Batch search not initialized or batch size too large" << std::endl;
        return false;
    }

    // Prepare base transformation matrix
    std::vector<float> h_base_transform(16);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            h_base_transform[i * 4 + j] = current_transform(i, j);
        }
    }

    float* d_base_transform = nullptr;
    if (!cuda_utils::AllocateDeviceMemory(&d_base_transform, 16) ||
        !cuda_utils::CopyHostToDevice(d_base_transform, h_base_transform.data(), 16)) {
        return false;
    }

    // Allocate device memory for deltas
    float* d_deltas = nullptr;
    if (!cuda_utils::AllocateDeviceMemory(&d_deltas, batch_size * 6)) {
        cuda_utils::FreeDeviceMemory(d_base_transform);
        return false;
    }

    // Generate random transformations
    int block_size, grid_size;
    cuda_utils::CalculateLaunchConfig(batch_size, grid_size, block_size);

    GenerateRandomTransformsKernel<<<grid_size, block_size>>>(
        d_rand_states_,
        batch_size,
        xyz_range,
        rpy_range,
        d_base_transform,
        d_transform_samples_,
        d_deltas
    );
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();

    // Download transformations to host
    if (!cuda_utils::CopyDeviceToHost(h_transforms_pinned_, d_transform_samples_, batch_size * 16)) {
        cuda_utils::FreeDeviceMemory(d_base_transform);
        cuda_utils::FreeDeviceMemory(d_deltas);
        return false;
    }

    // Evaluate all transformations
    // Use multiple streams to overlap computation
    int batch_per_stream = (batch_size + num_streams_ - 1) / num_streams_;

    for (int i = 0; i < batch_size; i++) {
        // Convert to Eigen matrix
        Eigen::Matrix4f T;
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                T(r, c) = h_transforms_pinned_[i * 16 + r * 4 + c];
            }
        }

        // Evaluate score (this calls GPU kernels internally)
        double score = score_calculator_->ComputeScore(T, file_index);

        if (score < 0.0) {
            std::cerr << "Score evaluation failed for candidate " << i << std::endl;
            cuda_utils::FreeDeviceMemory(d_base_transform);
            cuda_utils::FreeDeviceMemory(d_deltas);
            return false;
        }

        h_scores_pinned_[i] = score;
    }

    // Upload scores to device
    if (!cuda_utils::CopyHostToDevice(d_scores_, h_scores_pinned_, batch_size)) {
        cuda_utils::FreeDeviceMemory(d_base_transform);
        cuda_utils::FreeDeviceMemory(d_deltas);
        return false;
    }

    // Find minimum score using parallel reduction
    double init_max = 1e10;
    int init_idx = -1;
    if (!cuda_utils::CopyHostToDevice(d_best_score_, &init_max, 1) ||
        !cuda_utils::CopyHostToDevice(d_best_idx_, &init_idx, 1)) {
        cuda_utils::FreeDeviceMemory(d_base_transform);
        cuda_utils::FreeDeviceMemory(d_deltas);
        return false;
    }

    int reduce_block_size = 256;
    int reduce_grid_size = (batch_size + reduce_block_size - 1) / reduce_block_size;
    size_t shared_mem_size = reduce_block_size * (sizeof(double) + sizeof(int));

    FindMinScoreKernel<<<reduce_grid_size, reduce_block_size, shared_mem_size>>>(
        d_scores_,
        batch_size,
        d_best_idx_,
        d_best_score_
    );
    CUDA_CHECK_KERNEL();
    CUDA_SYNC_CHECK();

    // Download best result
    int best_idx_host;
    double best_score_host;
    if (!cuda_utils::CopyDeviceToHost(&best_idx_host, d_best_idx_, 1) ||
        !cuda_utils::CopyDeviceToHost(&best_score_host, d_best_score_, 1)) {
        cuda_utils::FreeDeviceMemory(d_base_transform);
        cuda_utils::FreeDeviceMemory(d_deltas);
        return false;
    }

    // Download best delta
    std::vector<float> h_deltas(batch_size * 6);
    if (!cuda_utils::CopyDeviceToHost(h_deltas.data(), d_deltas, batch_size * 6)) {
        cuda_utils::FreeDeviceMemory(d_base_transform);
        cuda_utils::FreeDeviceMemory(d_deltas);
        return false;
    }

    if (best_idx_host >= 0 && best_idx_host < batch_size) {
        // Extract best delta
        for (int i = 0; i < 6; i++) {
            best_var(i) = h_deltas[best_idx_host * 6 + i];
        }
        best_score = best_score_host;
    } else {
        std::cerr << "Invalid best index: " << best_idx_host << std::endl;
        cuda_utils::FreeDeviceMemory(d_base_transform);
        cuda_utils::FreeDeviceMemory(d_deltas);
        return false;
    }

    // Cleanup
    cuda_utils::FreeDeviceMemory(d_base_transform);
    cuda_utils::FreeDeviceMemory(d_deltas);

    return true;
}

} // namespace search_cuda
