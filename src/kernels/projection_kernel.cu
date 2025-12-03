/*
 * CUDA Kernel: Point Cloud Projection
 * Projects 3D lidar points onto 2D image plane in parallel
 */

#include "calibration_gpu.cuh"

/**
 * Projects point cloud points onto image plane
 *
 * Each thread processes one point:
 * 1. Apply 4x4 transformation matrix (lidar to camera frame)
 * 2. Apply 3x3 or 3x4 intrinsic matrix (camera to pixel coordinates)
 * 3. Perspective division
 * 4. Check if projection is valid (within image bounds, positive depth)
 *
 * @param points        Input point cloud (x, y, z, 1)
 * @param transform_matrix  4x4 transformation matrix (lidar to camera)
 * @param intrinsic_matrix  3x3 or 3x4 intrinsic matrix
 * @param intrinsic_size    9 for 3x3, 12 for 3x4
 * @param num_points    Number of points to project
 * @param img_width     Image width in pixels
 * @param img_height    Image height in pixels
 * @param pixel_x       Output: projected x pixel coordinates
 * @param pixel_y       Output: projected y pixel coordinates
 * @param valid         Output: true if projection is valid
 */
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_points) {
        return;
    }

    // Initialize output
    valid[idx] = false;

    // Load point
    float4 pt = points[idx];

    // Apply 4x4 transformation: cam_pt = T * lidar_pt
    // T is stored in row-major order
    float cam_x = transform_matrix[0] * pt.x + transform_matrix[1] * pt.y +
                  transform_matrix[2] * pt.z + transform_matrix[3];
    float cam_y = transform_matrix[4] * pt.x + transform_matrix[5] * pt.y +
                  transform_matrix[6] * pt.z + transform_matrix[7];
    float cam_z = transform_matrix[8] * pt.x + transform_matrix[9] * pt.y +
                  transform_matrix[10] * pt.z + transform_matrix[11];

    // Check if point is in front of camera
    if (cam_z <= 0.0f) {
        return;
    }

    // Apply intrinsic matrix
    float proj_x, proj_y, proj_z;

    if (intrinsic_size == 9) {
        // 3x3 intrinsic matrix (pinhole model)
        // K is stored in row-major order
        proj_x = intrinsic_matrix[0] * cam_x + intrinsic_matrix[1] * cam_y +
                 intrinsic_matrix[2] * cam_z;
        proj_y = intrinsic_matrix[3] * cam_x + intrinsic_matrix[4] * cam_y +
                 intrinsic_matrix[5] * cam_z;
        proj_z = intrinsic_matrix[6] * cam_x + intrinsic_matrix[7] * cam_y +
                 intrinsic_matrix[8] * cam_z;
    } else {
        // 3x4 intrinsic matrix (projection matrix)
        proj_x = intrinsic_matrix[0] * cam_x + intrinsic_matrix[1] * cam_y +
                 intrinsic_matrix[2] * cam_z + intrinsic_matrix[3];
        proj_y = intrinsic_matrix[4] * cam_x + intrinsic_matrix[5] * cam_y +
                 intrinsic_matrix[6] * cam_z + intrinsic_matrix[7];
        proj_z = intrinsic_matrix[8] * cam_x + intrinsic_matrix[9] * cam_y +
                 intrinsic_matrix[10] * cam_z + intrinsic_matrix[11];
    }

    // Perspective division
    if (proj_z <= 0.0f) {
        return;
    }

    float u = proj_x / proj_z;
    float v = proj_y / proj_z;

    // Round to nearest pixel
    int px = __float2int_rn(u);
    int py = __float2int_rn(v);

    // Check if within image bounds (no margin)
    if (px >= 0 && px < img_width && py >= 0 && py < img_height) {
        pixel_x[idx] = px;
        pixel_y[idx] = py;
        valid[idx] = true;
    }
}
