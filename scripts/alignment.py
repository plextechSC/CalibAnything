import cv2
import numpy as np
import open3d as o3d
import json
import sys

class LidarCameraAligner:
    def __init__(self, pcd_path, png_path, json_path=None):
        # 1. Load Data
        print(f"Loading cloud: {pcd_path}")
        self.pcd = o3d.io.read_point_cloud(pcd_path)
        if self.pcd.is_empty():
            print("Error: Point cloud is empty or file not found.")
            sys.exit(1)

        print(f"Loading image: {png_path}")
        self.img_orig = cv2.imread(png_path)
        if self.img_orig is None:
            print("Error: Image not found.")
            sys.exit(1)
        
        self.h, self.w = self.img_orig.shape[:2]

        # 2. Pre-process Cloud (Downsample for performance)
        # Adjust voxel_size if points are too sparse or too dense
        # self.pcd = self.pcd.voxel_down_sample(voxel_size=0.05)
        self.points = np.asarray(self.pcd.points)

        # 3. Initialization State
        # Translation (x, y, z)
        self.trans = np.array([0.0, 0.0, 0.0]) 
        # Rotation (Roll, Pitch, Yaw) in degrees
        self.rot_euler = np.array([0.0, 0.0, 0.0]) 

        # Load initial calibration if provided
        if json_path:
            self.load_initial_calibration(json_path)
        
        # Step sizes
        self.trans_step = 0.1  # meters
        self.rot_step = 1.0    # degrees

        # 4. Camera Intrinsics (CRITICAL)
        # You ideally need the real intrinsics of your camera.
        # Here we estimate a generic camera with ~60-90 deg FOV if unknown.
        # format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        fx = self.w  # Approximate focal length
        fy = self.w 
        cx = self.w / 2
        cy = self.h / 2
        
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # Distortion coefficients (assuming 0 for manual alignment, strictly pinhole)
        self.dist_coeffs = np.zeros((4,1))

    def load_initial_calibration(self, json_path):
        """ Loads T_lidar_to_cam from JSON and updates state """
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Parse the specific structure provided
            matrix_data = data["T_lidar_to_cam"]["data"]
            T = np.array(matrix_data)
            
            print(f"Loaded initial matrix from {json_path}")
            self.set_state_from_matrix(T)
            
        except Exception as e:
            print(f"Error loading JSON: {e}")
            sys.exit(1)

    def set_state_from_matrix(self, T):
        """ Decomposes 4x4 matrix into Trans (XYZ) and Euler (RPY) """
        # 1. Translation
        self.trans = T[:3, 3]
        
        # 2. Rotation (Decompose Rotation Matrix to Euler)
        # Assuming rotation order Rz * Ry * Rx (matches get_rotation_matrix)
        R = T[:3, :3]
        
        sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2,1] , R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0

        # Convert to degrees for internal state
        self.rot_euler = np.degrees(np.array([x, y, z]))

    def get_rotation_matrix(self):
        """ Converts current Euler angles to Rotation Matrix (R) """
        roll, pitch, yaw = np.radians(self.rot_euler)
        
        # Rotation matrices around axes
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        # Order: Rz * Ry * Rx (Standard extrinsic rotation order)
        R = Rz @ Ry @ Rx
        return R

    def get_transform_matrix(self):
        """ Combines R and T into a 4x4 Matrix """
        R = self.get_rotation_matrix()
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = self.trans
        return T

    def project_points(self):
        """ Projects 3D points to 2D image plane using current transforms """
        
        # Get current 4x4 transform
        T_lidar_to_cam = self.get_transform_matrix()
        
        # Convert points to Homogeneous coords (N, 4)
        ones = np.ones((self.points.shape[0], 1))
        points_hom = np.hstack((self.points, ones))
        
        # Transform points: P_cam = T * P_lidar
        # (Using matrix multiplication)
        points_cam = (T_lidar_to_cam @ points_hom.T).T  # Shape: (N, 4)
        
        # Extract XYZ in camera frame
        xyz_cam = points_cam[:, :3]
        
        # Filter points behind camera (Z < 0 usually implies behind in OpenCV frame)
        # OpenCV Camera Coords: Z-forward, X-right, Y-down
        mask = xyz_cam[:, 2] > 0.1
        xyz_cam = xyz_cam[mask]
        
        if len(xyz_cam) == 0:
            return self.img_orig.copy()

        # Project to Image Plane: P_img = K * P_cam
        # cv2.projectPoints handles perspective division (x/z, y/z)
        # We pass rvec=0, tvec=0 because we manually applied T earlier
        img_points, _ = cv2.projectPoints(
            xyz_cam, 
            np.zeros(3), # rvec (already applied)
            np.zeros(3), # tvec (already applied)
            self.K, 
            self.dist_coeffs
        )
        img_points = img_points.reshape(-1, 2)

        # Draw on image
        img_display = self.img_orig.copy()
        
        # Color mapping based on Depth (Z)
        depths = xyz_cam[:, 2]
        min_d, max_d = np.percentile(depths, 5), np.percentile(depths, 95)
        
        # Normalize depth for color
        norm_depths = np.clip((depths - min_d) / (max_d - min_d + 1e-6), 0, 1)
        norm_depths = (norm_depths * 255).astype(np.uint8)
        
        # Apply colormap
        colormap = cv2.applyColorMap(norm_depths, cv2.COLORMAP_JET)
        
        # Draw circles
        for i, (u, v) in enumerate(img_points):
            if 0 <= u < self.w and 0 <= v < self.h:
                # Get color from colormap
                color = colormap[i, 0].tolist()
                cv2.circle(img_display, (int(u), int(v)), 2, color, -1)

        return img_display

    def print_result(self):
        T = self.get_transform_matrix()
        
        # Construct JSON object
        output = {
            "T_lidar_to_cam": {
                "rows": 4,
                "cols": 4,
                "data": T.tolist() # Convert numpy array to list
            }
        }
        
        print("\n" + "="*30)
        print("FINAL TRANSFORMATION MATRIX:")
        print(json.dumps(output, indent=4))
        print("="*30 + "\n")

    def run(self):
        print("\n--- CONTROLS ---")
        print("Arrows:  Move X / Y")
        print("R / F :  Move Z (Depth) +/-")
        print("W / S :  Pitch (Rotate X)")
        print("A / D :  Yaw   (Rotate Y)")
        print("Q / E :  Roll  (Rotate Z)")
        print("+ / - :  Adjust Step Size")
        print("Enter :  Print Matrix (JSON)")
        print("ESC   :  Quit")
        print("----------------")

        while True:
            # Generate overlaid image
            frame = self.project_points()

            # Overlay UI text
            info_txt = f"XYZ: {self.trans} | RPY: {self.rot_euler}"
            step_txt = f"Step T: {self.trans_step:.2f} | Step R: {self.rot_step:.1f}"
            cv2.putText(frame, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, step_txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Lidar Overlay Tool", frame)
            
            key = cv2.waitKey(10)
            
            if key == 27: # ESC
                break
            elif key == 13: # Enter
                self.print_result()
                
            # Translation
            elif key == 82: # Up Arrow (OpenCV key code might vary by OS, this is common)
                self.trans[1] -= self.trans_step # Y Up (Image coords)
            elif key == 84: # Down Arrow
                self.trans[1] += self.trans_step
            elif key == 81: # Left Arrow
                self.trans[0] -= self.trans_step
            elif key == 83: # Right Arrow
                self.trans[0] += self.trans_step
            elif key == ord('r'): # Depth +
                self.trans[2] += self.trans_step
            elif key == ord('f'): # Depth -
                self.trans[2] -= self.trans_step

            # Rotation
            elif key == ord('w'): # Pitch down
                self.rot_euler[1] -= self.rot_step
            elif key == ord('s'): # Pitch up
                self.rot_euler[1] += self.rot_step
            elif key == ord('a'): # Yaw left
                self.rot_euler[2] -= self.rot_step
            elif key == ord('d'): # Yaw right
                self.rot_euler[2] += self.rot_step
            elif key == ord('q'): # Roll left
                self.rot_euler[0] -= self.rot_step
            elif key == ord('e'): # Roll right
                self.rot_euler[0] += self.rot_step

            # Sensitivity
            elif key == ord('+') or key == ord('='):
                self.trans_step *= 1.5
                self.rot_step *= 1.5
            elif key == ord('-') or key == ord('_'):
                self.trans_step /= 1.5
                self.rot_step /= 1.5

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Default file names (change these or pass as args)
    # You can create a dummy pcd and png to test if you don't have files ready.
    pcd_file = "./data/lucid/pc/000000.pcd" 
    png_file = "./data/lucid/images/000000.png"
    json_file = ""

    # Check if arguments provided
    if len(sys.argv) > 2:
        pcd_file = sys.argv[1]
        png_file = sys.argv[2]
    
    if len(sys.argv) > 3:
        json_file = sys.argv[3]

    try:
        app = LidarCameraAligner(pcd_file, png_file, json_file)
        app.run()
    except Exception as e:
        print(f"Initialization Failed: {e}")
        print("Usage: python lidar_camera_calibrator.py <path_to_pcd> <path_to_png> [optional_path_to_json]")
