import json
import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def project_and_save(calib, points, image_file, output_image_file, cx_offset, cy_offset, use_distortion, radius):
    image = cv2.imread(image_file)
    
    intrinsic_params = calib['intrinsic_params']
    extrinsic_params = calib['extrinsic_params']

    fx = intrinsic_params['fx']
    fy = intrinsic_params['fy']
    cx = intrinsic_params['cx'] + cx_offset
    cy = intrinsic_params['cy'] + cy_offset

    
    
    fisheye_model = intrinsic_params.get('camera_model') == 'fisheye'

    if use_distortion and fisheye_model:
        k1, k2, k3, k4 = intrinsic_params.get('k1', 0), intrinsic_params.get('k2', 0), intrinsic_params.get('k3', 0), intrinsic_params.get('k4', 0)
        dist_coeffs = np.array([k1, k2, k3, k4])
        project_func = cv2.fisheye.projectPoints
    elif use_distortion and not fisheye_model:  # Pinhole with distortion
        k1, k2, p1, p2 = intrinsic_params['k1'], intrinsic_params['k2'], intrinsic_params['p1'], intrinsic_params['p2']
        k3, k4, k5, k6 = intrinsic_params.get('k3', 0), intrinsic_params.get('k4', 0), intrinsic_params.get('k5', 0), intrinsic_params.get('k6', 0)
        dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6])
        project_func = cv2.projectPoints
    else:  # No distortion
        dist_coeffs = np.zeros(8)
        project_func = cv2.projectPoints

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])

    roll, pitch, yaw = extrinsic_params['roll'], extrinsic_params['pitch'], extrinsic_params['yaw']
    px, py, pz = extrinsic_params['px'], extrinsic_params['py'], extrinsic_params['pz']

    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
    rotation_matrix_cam_to_lidar = r.as_matrix()
    translation_vector_cam_to_lidar = np.array([px, py, pz])

    # Transform points from LiDAR to camera frame
    rotation_matrix = rotation_matrix_cam_to_lidar.T
    translation_vector = -np.dot(rotation_matrix, translation_vector_cam_to_lidar)

    transformed_points = (rotation_matrix @ points.T).T + translation_vector
    front_points = transformed_points[transformed_points[:, 2] > 0]
    
    if project_func == cv2.fisheye.projectPoints:
        front_points_reshaped = front_points.reshape(-1, 1, 3)
        image_points, _ = project_func(front_points_reshaped, np.zeros((3, 1)), np.zeros((3, 1)), camera_matrix, dist_coeffs)
    else:
        image_points, _ = project_func(front_points, np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)

    h, w, _ = image.shape
    depths = front_points[:, 2]
    normalized_depths = (255 * (depths % 10) / 10).astype(np.uint8)
    colors = cv2.applyColorMap(normalized_depths, cv2.COLORMAP_JET)

    for i, p in enumerate(image_points):
        x, y = int(p[0][0]), int(p[0][1])
        if 0 <= x < w and 0 <= y < h:
            color = colors[i][0].tolist()
            cv2.circle(image, (x, y), radius, color, -1)

    cv2.imwrite(output_image_file, image)
    print(f"✅ Projected image saved to {output_image_file}")
    print("camtolidar", rotation_matrix_cam_to_lidar)
    print("distortions", dist_coeffs)
    print("camera_mat", camera_matrix)

if __name__ == '__main__':
    # --- Set your own files and values here ---
    calibration_file = '/Users/mahitnamburu/Desktop/LucidMotors/CalibAnything/data/lucid/rnc_c/rnc_c.json'
    pcd_file = '/Users/mahitnamburu/Desktop/LucidMotors/CalibAnything/data/cam06/pc/000000.pcd'
    image_file = '/Users/mahitnamburu/Desktop/LucidMotors/CalibAnything/data/cam06/images/000000.png'
    output_image_file = '/Users/mahitnamburu/Desktop/LucidMotors/CalibAnything/projected_outputcam06.jpg'

    cx_offset = 0
    cy_offset = 0
    radius = 3
    use_distortion = False  # Set to False to ignore distortion

    # --- Load calibration and point cloud ---
    with open(calibration_file, 'r') as f:
        calib = json.load(f)
    
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # --- Run projection ---
    project_and_save(
        calib=calib,
        points=points,
        image_file=image_file,
        output_image_file=output_image_file,
        cx_offset=cx_offset,
        cy_offset=cy_offset,
        use_distortion=use_distortion,
        radius=radius
    )
