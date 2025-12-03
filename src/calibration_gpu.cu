/*
 * GPU-Accelerated Calibration Score Calculator Implementation
 * Host-side code for CUDA-accelerated lidar-camera calibration
 */

#include "calibration_gpu.cuh"
#include "calibration.hpp"
#include <iostream>
#include <algorithm>

namespace calib_cuda {

CalScoreGPU::CalScoreGPU()
    : d_intrinsic_(nullptr),
      d_transform_(nullptr),
      h_transform_pinned_(nullptr),
      h_score_pinned_(nullptr),
      intrinsic_size_(0),
      point_per_pixel_(0.05f),
      curvature_max_(0.0f),
      img_width_(0),
      img_height_(0),
      initialized_(false),
      device_id_(0) {
}

CalScoreGPU::~CalScoreGPU() {
    FreeGPUMemory();
}

bool CalScoreGPU::Initialize(
    const std::vector<pcl::PointCloud<PointXYZINS>::Ptr>& point_clouds,
    const std::vector<cv::Mat>& masks,
    const std::vector<std::vector<int>>& mask_point_nums,
    const std::vector<int>& n_mask,
    const std::vector<int>& n_seg,
    const Eigen::MatrixXf& intrinsic,
    int img_width,
    int img_height,
    float point_per_pixel,
    float curvature_max
) {
    std::cout << "Initializing GPU calibration..." << std::endl;

    // Check CUDA availability
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA devices available. Falling back to CPU." << std::endl;
        return false;
    }

    // Select device
    CUDA_CHECK(cudaSetDevice(device_id_));

    // Print device info
    if (!cuda_utils::PrintDeviceInfo(device_id_)) {
        return false;
    }

    // Store parameters
    img_width_ = img_width;
    img_height_ = img_height;
    point_per_pixel_ = point_per_pixel;
    curvature_max_ = curvature_max;

    // Determine intrinsic matrix size
    intrinsic_size_ = (intrinsic.cols() == 3) ? 9 : 12;

    // Estimate total GPU memory requirement
    size_t total_memory = 0;
    for (size_t i = 0; i < point_clouds.size(); i++) {
        size_t num_points = point_clouds[i]->size();
        size_t num_masks = n_mask[i];
        size_t max_segments = n_seg[i];

        size_t file_memory = 0;
        file_memory += num_points * sizeof(float4);  // points
        file_memory += num_points * sizeof(float3);  // normals
        file_memory += num_points * sizeof(float);   // intensities
        file_memory += num_points * sizeof(int);     // segments
        file_memory += num_points * sizeof(int) * 2; // projected coords
        file_memory += num_points * sizeof(bool);    // valid flags
        file_memory += img_width * img_height * sizeof(uchar4);  // masks
        file_memory += num_masks * sizeof(int);      // mask pixel counts
        file_memory += num_masks * 3 * sizeof(float);  // normal sums
        file_memory += num_masks * sizeof(float);    // intensity sums
        file_memory += num_masks * max_segments * sizeof(int);  // segment counts
        file_memory += num_masks * sizeof(int);      // point counts
        file_memory += num_masks * sizeof(float) * 3;  // scores
        file_memory += num_masks * sizeof(bool);     // valid flags

        total_memory += file_memory;
    }

    // Add shared memory
    total_memory += intrinsic_size_ * sizeof(float);  // intrinsic
    total_memory += 16 * sizeof(float);  // transform
    total_memory += 100 * 1024 * 1024;  // 100MB buffer

    std::cout << "Estimated GPU memory requirement: "
              << total_memory / (1024.0 * 1024.0) << " MB" << std::endl;

    // Check available memory
    if (!cuda_utils::CheckAvailableMemory(total_memory, device_id_)) {
        std::cerr << "Insufficient GPU memory. Falling back to CPU." << std::endl;
        return false;
    }

    // Allocate shared GPU memory
    if (!cuda_utils::AllocateDeviceMemory(&d_intrinsic_, intrinsic_size_)) {
        return false;
    }
    if (!cuda_utils::AllocateDeviceMemory(&d_transform_, 16)) {
        return false;
    }

    // Allocate pinned host memory for fast transfers
    if (!cuda_utils::AllocatePinnedMemory(&h_transform_pinned_, 16)) {
        return false;
    }
    if (!cuda_utils::AllocatePinnedMemory(&h_score_pinned_, 1)) {
        return false;
    }

    // Upload intrinsic matrix (row-major)
    std::vector<float> intrinsic_data;
    for (int i = 0; i < intrinsic.rows(); i++) {
        for (int j = 0; j < intrinsic.cols(); j++) {
            intrinsic_data.push_back(intrinsic(i, j));
        }
    }
    if (!cuda_utils::CopyHostToDevice(d_intrinsic_, intrinsic_data.data(),
                                      intrinsic_size_)) {
        return false;
    }

    // Upload data for each file
    gpu_data_.resize(point_clouds.size());

    for (size_t i = 0; i < point_clouds.size(); i++) {
        std::cout << "Uploading data for file " << i + 1 << "/"
                  << point_clouds.size() << "..." << std::endl;

        if (!UploadFileData(i, point_clouds[i], masks[i], mask_point_nums[i],
                           n_mask[i], n_seg[i])) {
            std::cerr << "Failed to upload data for file " << i << std::endl;
            FreeGPUMemory();
            return false;
        }
    }

    initialized_ = true;
    std::cout << "GPU initialization complete!" << std::endl;
    return true;
}

bool CalScoreGPU::UploadFileData(
    int file_index,
    const pcl::PointCloud<PointXYZINS>::Ptr& point_cloud,
    const cv::Mat& masks,
    const std::vector<int>& mask_point_nums,
    int n_mask,
    int n_seg
) {
    GPUCalibData& data = gpu_data_[file_index];

    int num_points = point_cloud->size();
    data.num_points = num_points;
    data.num_masks = n_mask;
    data.max_segments = n_seg;
    data.img_width = img_width_;
    data.img_height = img_height_;

    // Prepare host data (Structure of Arrays)
    std::vector<float4> h_points(num_points);
    std::vector<float3> h_normals(num_points);
    std::vector<float> h_intensities(num_points);
    std::vector<int> h_segments(num_points);

    for (int i = 0; i < num_points; i++) {
        const PointXYZINS& pt = (*point_cloud)[i];
        h_points[i] = make_float4(pt.x, pt.y, pt.z, 1.0f);
        h_normals[i] = make_float3(pt.normal_x, pt.normal_y, pt.normal_z);
        h_intensities[i] = pt.intensity;
        h_segments[i] = pt.segment;
    }

    // Allocate and upload point cloud data
    if (!cuda_utils::AllocateDeviceMemory(&data.d_points, num_points) ||
        !cuda_utils::CopyHostToDevice(data.d_points, h_points.data(), num_points)) {
        return false;
    }

    if (!cuda_utils::AllocateDeviceMemory(&data.d_normals, num_points) ||
        !cuda_utils::CopyHostToDevice(data.d_normals, h_normals.data(), num_points)) {
        return false;
    }

    if (!cuda_utils::AllocateDeviceMemory(&data.d_intensities, num_points) ||
        !cuda_utils::CopyHostToDevice(data.d_intensities, h_intensities.data(), num_points)) {
        return false;
    }

    if (!cuda_utils::AllocateDeviceMemory(&data.d_segments, num_points) ||
        !cuda_utils::CopyHostToDevice(data.d_segments, h_segments.data(), num_points)) {
        return false;
    }

    // Allocate workspace memory
    if (!cuda_utils::AllocateDeviceMemory(&data.d_projected_x, num_points) ||
        !cuda_utils::AllocateDeviceMemory(&data.d_projected_y, num_points) ||
        !cuda_utils::AllocateDeviceMemory(&data.d_valid, num_points)) {
        return false;
    }

    // Upload mask image
    size_t mask_pixels = img_width_ * img_height_;
    if (!cuda_utils::AllocateDeviceMemory(&data.d_masks, mask_pixels)) {
        return false;
    }

    // Convert cv::Mat to uchar4 array
    std::vector<uchar4> h_masks(mask_pixels);
    for (int y = 0; y < img_height_; y++) {
        for (int x = 0; x < img_width_; x++) {
            cv::Vec4b pixel = masks.at<cv::Vec4b>(y, x);
            h_masks[y * img_width_ + x] = make_uchar4(pixel[0], pixel[1], pixel[2], pixel[3]);
        }
    }

    if (!cuda_utils::CopyHostToDevice(data.d_masks, h_masks.data(), mask_pixels)) {
        return false;
    }

    // Upload mask pixel counts
    if (!cuda_utils::AllocateDeviceMemory(&data.d_mask_pixel_counts, n_mask) ||
        !cuda_utils::CopyHostToDevice(data.d_mask_pixel_counts, mask_point_nums.data(), n_mask)) {
        return false;
    }

    // Allocate accumulation buffers
    if (!cuda_utils::AllocateDeviceMemory(&data.d_mask_normal_sums, n_mask * 3) ||
        !cuda_utils::AllocateDeviceMemory(&data.d_mask_intensity_sums, n_mask) ||
        !cuda_utils::AllocateDeviceMemory(&data.d_mask_segment_counts, n_mask * n_seg) ||
        !cuda_utils::AllocateDeviceMemory(&data.d_mask_point_counts, n_mask)) {
        return false;
    }

    // Allocate score outputs
    if (!cuda_utils::AllocateDeviceMemory(&data.d_normal_scores, n_mask) ||
        !cuda_utils::AllocateDeviceMemory(&data.d_intensity_scores, n_mask) ||
        !cuda_utils::AllocateDeviceMemory(&data.d_segment_scores, n_mask) ||
        !cuda_utils::AllocateDeviceMemory(&data.d_mask_valid, n_mask)) {
        return false;
    }

    return true;
}

double CalScoreGPU::ComputeScore(
    const Eigen::Matrix4f& transform,
    int file_index
) {
    if (!initialized_ || file_index < 0 || file_index >= (int)gpu_data_.size()) {
        std::cerr << "GPU not initialized or invalid file index" << std::endl;
        return -1.0;
    }

    const GPUCalibData& data = gpu_data_[file_index];

    // Copy transformation matrix to pinned memory (row-major)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            h_transform_pinned_[i * 4 + j] = transform(i, j);
        }
    }

    // Upload transformation matrix
    if (!cuda_utils::CopyHostToDevice(d_transform_, h_transform_pinned_, 16)) {
        return -1.0;
    }

    // Calculate launch configurations
    int block_size, grid_size;
    cuda_utils::CalculateLaunchConfig(data.num_points, grid_size, block_size);

    // Step 1: Project points
    ProjectPointsKernel<<<grid_size, block_size>>>(
        data.d_points,
        d_transform_,
        d_intrinsic_,
        intrinsic_size_,
        data.num_points,
        data.img_width,
        data.img_height,
        data.d_projected_x,
        data.d_projected_y,
        data.d_valid
    );
    CUDA_CHECK_KERNEL();

    // Step 2: Zero accumulation buffers
    if (!cuda_utils::ZeroDeviceMemory(data.d_mask_normal_sums, data.num_masks * 3) ||
        !cuda_utils::ZeroDeviceMemory(data.d_mask_intensity_sums, data.num_masks) ||
        !cuda_utils::ZeroDeviceMemory(data.d_mask_segment_counts, data.num_masks * data.max_segments) ||
        !cuda_utils::ZeroDeviceMemory(data.d_mask_point_counts, data.num_masks)) {
        return -1.0;
    }

    // Step 3: Accumulate mask statistics
    AccumulateMaskStatsKernel<<<grid_size, block_size>>>(
        data.d_projected_x,
        data.d_projected_y,
        data.d_valid,
        data.d_normals,
        data.d_intensities,
        data.d_segments,
        data.d_masks,
        data.num_points,
        data.img_width,
        data.img_height,
        data.num_masks,
        data.max_segments,
        data.d_mask_normal_sums,
        data.d_mask_intensity_sums,
        data.d_mask_segment_counts,
        data.d_mask_point_counts
    );
    CUDA_CHECK_KERNEL();

    // Step 4: Compute per-mask scores
    int mask_grid_size, mask_block_size;
    cuda_utils::CalculateLaunchConfig(data.num_masks, mask_grid_size, mask_block_size);

    ComputeScoresKernel<<<mask_grid_size, mask_block_size>>>(
        data.d_mask_normal_sums,
        data.d_mask_intensity_sums,
        data.d_mask_segment_counts,
        data.d_mask_point_counts,
        data.d_mask_pixel_counts,
        data.num_masks,
        data.max_segments,
        point_per_pixel_,
        data.img_width,
        data.img_height,
        data.d_normal_scores,
        data.d_intensity_scores,
        data.d_segment_scores,
        data.d_mask_valid
    );
    CUDA_CHECK_KERNEL();

    // Step 5: Reduce scores (download to host and compute final score)
    // For simplicity, we do the final reduction on CPU
    std::vector<float> h_normal_scores(data.num_masks);
    std::vector<float> h_intensity_scores(data.num_masks);
    std::vector<float> h_segment_scores(data.num_masks);
    std::vector<bool> h_mask_valid(data.num_masks);

    if (!cuda_utils::CopyDeviceToHost(h_normal_scores.data(), data.d_normal_scores, data.num_masks) ||
        !cuda_utils::CopyDeviceToHost(h_intensity_scores.data(), data.d_intensity_scores, data.num_masks) ||
        !cuda_utils::CopyDeviceToHost(h_segment_scores.data(), data.d_segment_scores, data.num_masks) ||
        !cuda_utils::CopyDeviceToHost(h_mask_valid.data(), data.d_mask_valid, data.num_masks)) {
        return -1.0;
    }

    // Compute final score on host (matches CPU implementation)
    double normal_sum = 0.0, intensity_sum = 0.0, segment_sum = 0.0;
    int valid_count = 0;

    for (int i = 0; i < data.num_masks; i++) {
        if (h_mask_valid[i]) {
            normal_sum += h_normal_scores[i];
            intensity_sum += h_intensity_scores[i];
            segment_sum += h_segment_scores[i];
            valid_count++;
        }
    }

    if (valid_count == 0) {
        return 2.0;  // Maximum penalty
    }

    double normal_mean = normal_sum / valid_count;
    double intensity_mean = intensity_sum / valid_count;
    double segment_mean = segment_sum / valid_count;

    double score = 2.0 - 0.3 * normal_mean - 0.2 * intensity_mean -
                   0.5 * segment_mean - 0.0001 * valid_count;

    return score;
}

size_t CalScoreGPU::GetTotalGPUMemoryUsage() const {
    size_t total = 0;

    // Shared memory
    total += intrinsic_size_ * sizeof(float);
    total += 16 * sizeof(float);

    // Per-file memory
    for (const auto& data : gpu_data_) {
        total += data.num_points * sizeof(float4);
        total += data.num_points * sizeof(float3);
        total += data.num_points * sizeof(float) * 2;
        total += data.num_points * sizeof(int) * 3;
        total += data.num_points * sizeof(bool);
        total += data.img_width * data.img_height * sizeof(uchar4);
        total += data.num_masks * (sizeof(int) + sizeof(float) * 6 + sizeof(bool));
        total += data.num_masks * data.max_segments * sizeof(int);
    }

    return total;
}

void CalScoreGPU::FreeGPUMemory() {
    if (!initialized_) {
        return;
    }

    // Free per-file data
    for (auto& data : gpu_data_) {
        FreeFileData(data);
    }
    gpu_data_.clear();

    // Free shared memory
    cuda_utils::FreeDeviceMemory(d_intrinsic_);
    cuda_utils::FreeDeviceMemory(d_transform_);

    // Free pinned memory
    cuda_utils::FreePinnedMemory(h_transform_pinned_);
    cuda_utils::FreePinnedMemory(h_score_pinned_);

    d_intrinsic_ = nullptr;
    d_transform_ = nullptr;
    h_transform_pinned_ = nullptr;
    h_score_pinned_ = nullptr;

    initialized_ = false;
}

void CalScoreGPU::FreeFileData(GPUCalibData& data) {
    cuda_utils::FreeDeviceMemory(data.d_points);
    cuda_utils::FreeDeviceMemory(data.d_normals);
    cuda_utils::FreeDeviceMemory(data.d_intensities);
    cuda_utils::FreeDeviceMemory(data.d_segments);
    cuda_utils::FreeDeviceMemory(data.d_masks);
    cuda_utils::FreeDeviceMemory(data.d_mask_pixel_counts);
    cuda_utils::FreeDeviceMemory(data.d_projected_x);
    cuda_utils::FreeDeviceMemory(data.d_projected_y);
    cuda_utils::FreeDeviceMemory(data.d_valid);
    cuda_utils::FreeDeviceMemory(data.d_mask_normal_sums);
    cuda_utils::FreeDeviceMemory(data.d_mask_intensity_sums);
    cuda_utils::FreeDeviceMemory(data.d_mask_segment_counts);
    cuda_utils::FreeDeviceMemory(data.d_mask_point_counts);
    cuda_utils::FreeDeviceMemory(data.d_normal_scores);
    cuda_utils::FreeDeviceMemory(data.d_intensity_scores);
    cuda_utils::FreeDeviceMemory(data.d_segment_scores);
    cuda_utils::FreeDeviceMemory(data.d_mask_valid);
}

} // namespace calib_cuda
