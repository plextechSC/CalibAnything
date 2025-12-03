/*
 * GPU-Accelerated Calibration Score Calculator
 * CUDA implementation for high-performance lidar-camera calibration
 */

#pragma once

#ifdef USE_CUDA

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <eigen3/Eigen/Core>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>

#include "cuda_utils.cuh"

// Forward declaration of custom point type (defined in calibration.hpp)
struct PointXYZINS;

namespace calib_cuda {

// GPU data structure for per-file calibration data
struct GPUCalibData {
    // Point cloud data (Structure of Arrays for coalesced access)
    float4* d_points;              // (x, y, z, 1) in lidar frame
    float3* d_normals;             // (nx, ny, nz)
    float* d_intensities;          // Intensity values
    int* d_segments;               // Segment IDs

    // Mask data
    uchar4* d_masks;               // 4-channel mask image (up to 4 overlapping masks per pixel)
    int* d_mask_pixel_counts;      // Number of pixels per mask

    // Workspace memory (reused across iterations)
    int* d_projected_x;            // Projected pixel x coordinates
    int* d_projected_y;            // Projected pixel y coordinates
    bool* d_valid;                 // Projection validity flags

    // Accumulation buffers
    float* d_mask_normal_sums;     // Accumulated normals per mask [num_masks * 3]
    float* d_mask_intensity_sums;  // Accumulated intensities per mask
    int* d_mask_segment_counts;    // Segment histogram per mask [num_masks * max_segments]
    int* d_mask_point_counts;      // Point count per mask

    // Score computation outputs
    float* d_normal_scores;        // Normal consistency scores per mask
    float* d_intensity_scores;     // Intensity consistency scores per mask
    float* d_segment_scores;       // Segment consistency scores per mask
    bool* d_mask_valid;            // Mask validity flags

    // Metadata
    int num_points;
    int num_masks;
    int max_segments;
    int img_width;
    int img_height;

    GPUCalibData()
        : d_points(nullptr), d_normals(nullptr), d_intensities(nullptr),
          d_segments(nullptr), d_masks(nullptr), d_mask_pixel_counts(nullptr),
          d_projected_x(nullptr), d_projected_y(nullptr), d_valid(nullptr),
          d_mask_normal_sums(nullptr), d_mask_intensity_sums(nullptr),
          d_mask_segment_counts(nullptr), d_mask_point_counts(nullptr),
          d_normal_scores(nullptr), d_intensity_scores(nullptr),
          d_segment_scores(nullptr), d_mask_valid(nullptr),
          num_points(0), num_masks(0), max_segments(0),
          img_width(0), img_height(0) {}
};

// Main GPU calibration score calculator class
class CalScoreGPU {
public:
    CalScoreGPU();
    ~CalScoreGPU();

    // Initialize GPU resources and upload data
    // Returns true on success, false on failure (will fallback to CPU)
    bool Initialize(
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
    );

    // Compute calibration score for given transformation
    // Returns score value, or -1.0 on error
    double ComputeScore(
        const Eigen::Matrix4f& transform,
        int file_index
    );

    // Check if GPU is initialized and ready
    bool IsInitialized() const { return initialized_; }

    // Get memory usage information
    size_t GetTotalGPUMemoryUsage() const;

private:
    // Upload data for one file
    bool UploadFileData(
        int file_index,
        const pcl::PointCloud<PointXYZINS>::Ptr& point_cloud,
        const cv::Mat& masks,
        const std::vector<int>& mask_point_nums,
        int n_mask,
        int n_seg
    );

    // Free GPU memory for all files
    void FreeGPUMemory();

    // Free GPU memory for one file
    void FreeFileData(GPUCalibData& data);

    // GPU data for each file
    std::vector<GPUCalibData> gpu_data_;

    // Shared GPU memory across all files
    float* d_intrinsic_;           // 3x3 or 3x4 intrinsic matrix
    float* d_transform_;           // 4x4 transformation matrix (updated each iteration)

    // Pinned host memory for fast transfers
    float* h_transform_pinned_;    // 4x4 transformation matrix
    double* h_score_pinned_;       // Final score result

    // Parameters
    int intrinsic_size_;           // 9 (3x3) or 12 (3x4)
    float point_per_pixel_;
    float curvature_max_;
    int img_width_;
    int img_height_;

    // State
    bool initialized_;
    int device_id_;
};

} // namespace calib_cuda

// CUDA kernel declarations (implemented in separate .cu files)

// Projects point cloud onto image plane
__global__ void ProjectPointsKernel(
    const float4* points,
    const float* transform_matrix,
    const float* intrinsic_matrix,
    int intrinsic_size,
    int num_points,
    int img_width,
    int img_height,
    int* pixel_x,
    int* pixel_y,
    bool* valid
);

// Accumulates per-mask statistics (normals, intensities, segments)
__global__ void AccumulateMaskStatsKernel(
    const int* pixel_x,
    const int* pixel_y,
    const bool* valid,
    const float3* normals,
    const float* intensities,
    const int* segments,
    const uchar4* mask_image,
    int num_points,
    int img_width,
    int img_height,
    int num_masks,
    int max_segments,
    float* mask_normal_sums,
    float* mask_intensity_sums,
    int* mask_segment_counts,
    int* mask_point_counts
);

// Computes consistency scores per mask
__global__ void ComputeScoresKernel(
    const float* mask_normal_sums,
    const float* mask_intensity_sums,
    const int* mask_segment_counts,
    const int* mask_point_counts,
    const int* mask_pixel_counts,
    int num_masks,
    int max_segments,
    float point_per_pixel,
    int img_width,
    int img_height,
    float* normal_scores,
    float* intensity_scores,
    float* segment_scores,
    bool* mask_valid
);

// Reduces scores to final calibration score
__global__ void ReduceScoresKernel(
    const float* normal_scores,
    const float* intensity_scores,
    const float* segment_scores,
    const bool* mask_valid,
    int num_masks,
    float* final_score
);

#endif // USE_CUDA
