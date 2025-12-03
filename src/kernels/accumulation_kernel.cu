/*
 * CUDA Kernel: Mask Statistics Accumulation
 * Accumulates per-mask statistics for projected points
 */

#include "calibration_gpu.cuh"

/**
 * Accumulates statistics for each mask
 *
 * Each thread processes one projected point:
 * 1. Check if projection is valid
 * 2. Look up mask ID(s) at pixel location (up to 4 overlapping masks)
 * 3. Atomically accumulate normals, intensities, segment counts
 *
 * Uses atomic operations for thread-safe accumulation.
 * Memory layout: Structure of Arrays for coalesced access.
 *
 * @param pixel_x           Projected x coordinates
 * @param pixel_y           Projected y coordinates
 * @param valid             Projection validity flags
 * @param normals           Normal vectors (nx, ny, nz)
 * @param intensities       Intensity values
 * @param segments          Segment IDs
 * @param mask_image        4-channel mask image (each channel = mask ID, 0 = no mask)
 * @param num_points        Number of points
 * @param img_width         Image width
 * @param img_height        Image height
 * @param num_masks         Total number of masks
 * @param max_segments      Maximum segment ID + 1
 * @param mask_normal_sums  Output: accumulated normals per mask [num_masks * 3]
 * @param mask_intensity_sums  Output: accumulated intensities [num_masks]
 * @param mask_segment_counts  Output: segment histogram [num_masks * max_segments]
 * @param mask_point_counts    Output: point count [num_masks]
 */
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) {
        return;
    }

    // Skip invalid projections
    if (!valid[idx]) {
        return;
    }

    // Get pixel coordinates
    int px = pixel_x[idx];
    int py = pixel_y[idx];

    // Bounds check (should already be valid, but double-check)
    if (px < 0 || px >= img_width || py < 0 || py >= img_height) {
        return;
    }

    // Load point attributes
    float3 normal = normals[idx];
    float intensity = intensities[idx];
    int segment = segments[idx];

    // Look up mask IDs at this pixel (up to 4 overlapping masks)
    int pixel_idx = py * img_width + px;
    uchar4 mask_ids = mask_image[pixel_idx];

    // Process each mask channel (up to 4 masks can overlap at one pixel)
    for (int c = 0; c < 4; c++) {
        unsigned char mask_id_byte = 0;
        if (c == 0) mask_id_byte = mask_ids.x;
        else if (c == 1) mask_id_byte = mask_ids.y;
        else if (c == 2) mask_id_byte = mask_ids.z;
        else mask_id_byte = mask_ids.w;

        // mask_id is 1-indexed (0 = no mask)
        if (mask_id_byte == 0) {
            break;  // No more masks at this pixel
        }

        int mask_id = (int)mask_id_byte - 1;  // Convert to 0-indexed

        if (mask_id < 0 || mask_id >= num_masks) {
            continue;  // Invalid mask ID, skip
        }

        // Atomically accumulate normal components
        atomicAdd(&mask_normal_sums[mask_id * 3 + 0], normal.x);
        atomicAdd(&mask_normal_sums[mask_id * 3 + 1], normal.y);
        atomicAdd(&mask_normal_sums[mask_id * 3 + 2], normal.z);

        // Atomically accumulate intensity
        atomicAdd(&mask_intensity_sums[mask_id], intensity);

        // Atomically accumulate point count
        atomicAdd(&mask_point_counts[mask_id], 1);

        // Atomically accumulate segment histogram
        if (segment >= 0 && segment < max_segments) {
            atomicAdd(&mask_segment_counts[mask_id * max_segments + segment], 1);
        }
    }
}
