#!/usr/bin/env python3
"""
Convert Lucid calibration JSON format to calib.json format.

Usage:
    python scripts/convert_lucid_calib.py --input data/lucid/fwc_c/fwc_c.json --output data/lucid_formatted/calib.json
    python scripts/convert_lucid_calib.py --input data/lucid/fnc/fnc.json --output data/lucid_formatted/calib.json --cx-scale 2 --cy-scale 2
    python scripts/convert_lucid_calib.py --input data/lucid/fwc_c/fwc_c.json --output output/calib.json --roll-offset 5.0 --pitch-offset -1.0
"""

import argparse
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def eulerangles_to_rotmat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """
    Convert Euler angles (in degrees) to rotation matrix.
    Uses scipy's 'xyz' extrinsic convention to match normal_cloud_visualizer.py
    """
    r = R.from_euler('xyz', [roll_deg, pitch_deg, yaw_deg], degrees=True)
    return r.as_matrix()


def compute_transformation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float,
                                    px: float, py: float, pz: float) -> np.ndarray:
    """
    Compute the 4x4 transformation matrix from lidar to camera frame.
    
    The transformation is: P_cam = R^(-1) * (P_lidar - t)
    So T_lidar_to_cam = [R^(-1) | -R^(-1)*t]
                        [  0    |     1    ]
    
    This matches the transformation in normal_cloud_visualizer.py:
        rotation_matrix = rotation_matrix_cam_to_lidar.T
        translation_vector = -np.dot(rotation_matrix, translation_vector_cam_to_lidar)
    """
    # Compute rotation matrix (cam-to-lidar) using scipy - same as normal_cloud_visualizer.py
    rot_cam_to_lidar = eulerangles_to_rotmat(roll_deg, pitch_deg, yaw_deg)
    t_cam_to_lidar = np.array([px, py, pz])
    
    # Compute inverse transformation (lidar-to-cam)
    rot_lidar_to_cam = rot_cam_to_lidar.T
    t_lidar_to_cam = -rot_lidar_to_cam @ t_cam_to_lidar
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = rot_lidar_to_cam
    T[:3, 3] = t_lidar_to_cam
    
    return T


def convert_lucid_to_calib(input_path: str, output_path: str = None,
                           fx_scale: float = 1.0, fy_scale: float = 1.0,
                           cx_scale: float = 1.0, cy_scale: float = 1.0,
                           roll_offset: float = 0.0, pitch_offset: float = 0.0, yaw_offset: float = 0.0,
                           tx_offset: float = 0.0, ty_offset: float = 0.0, tz_offset: float = 0.0,
                           file_names: list = None,
                           template_path: str = None) -> dict:
    """
    Convert Lucid calibration JSON to calib.json format.
    
    Args:
        input_path: Path to input Lucid calibration JSON file
        output_path: Path to output calib.json file (optional, if None returns dict only)
        fx_scale: Scale factor for fx (default 1.0)
        fy_scale: Scale factor for fy (default 1.0)
        cx_scale: Scale factor for cx (default 1.0)
        cy_scale: Scale factor for cy (default 1.0)
        roll_offset: Offset to add to roll angle in degrees (default 0.0)
        pitch_offset: Offset to add to pitch angle in degrees (default 0.0)
        yaw_offset: Offset to add to yaw angle in degrees (default 0.0)
        tx_offset: Offset to add to translation x (px) in meters (default 0.0)
        ty_offset: Offset to add to translation y (py) in meters (default 0.0)
        tz_offset: Offset to add to translation z (pz) in meters (default 0.0)
        file_names: List of file names to process (default ["000000", "000001", "000002"])
        template_path: Path to existing calib.json to use as template for params
    
    Returns:
        The converted calibration dictionary
    """
    # Load input calibration
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    intr = data['intrinsic_params']
    extr = data['extrinsic_params']
    
    # Extract intrinsics
    fx = intr['fx'] * fx_scale
    fy = intr['fy'] * fy_scale
    cx = intr['cx'] * cx_scale
    cy = intr['cy'] * cy_scale
    
    # Build 3x3 camera matrix
    cam_K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]
    
    # Distortion coefficients (fisheye uses k1-k4, add 0 for 5th)
    # NOTE, NO CAMERA IMAGES ARE DISTORTED ATM SO CAM_DIST IS ALL 0
    # k1 = intr.get('k1', 0)
    # k2 = intr.get('k2', 0)
    # k3 = intr.get('k3', 0)
    # k4 = intr.get('k4', 0)
    # cam_dist = [k1, k2, k3, k4, 0]
    cam_dist = [0, 0, 0, 0, 0]
    
    # Apply offsets to extrinsic parameters
    roll = extr['roll'] + roll_offset
    pitch = extr['pitch'] + pitch_offset
    yaw = extr['yaw'] + yaw_offset
    px = extr['px'] + tx_offset
    py = extr['py'] + ty_offset
    pz = extr['pz'] + tz_offset
    
    # Compute transformation matrix
    T = compute_transformation_matrix(roll, pitch, yaw, px, py, pz)
    T_list = T.tolist()
    
    # Default file names
    if file_names is None:
        file_names = ["000000"]
    
    # Load template params if provided
    params = {
        "min_plane_point_num": 2000,
        "cluster_tolerance": 0.25,
        "search_num": 4000,
        "search_range": {
            "rot_deg": 5,
            "trans_m": 0.5
        },
        "point_range": {
            "top": 0.0,
            "bottom": 1.0
        },
        "down_sample": {
            "is_valid": False,
            "voxel_m": 0.05
        },
        "thread": {
            "is_multi_thread": True,
            "num_thread": 8
        }
    }
    
    if template_path:
        try:
            with open(template_path, 'r') as f:
                template = json.load(f)
                if 'params' in template:
                    params = template['params']
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    # Build output calibration
    calib = {
        "cam_K": {
            "rows": len(cam_K),
            "cols": len(cam_K[0]),
            "data": cam_K
        },
        "cam_dist": {
            "cols": len(cam_dist),
            "data": cam_dist
        },
        "T_lidar_to_cam": {
            "rows": len(T_list),
            "cols": len(T_list[0]),
            "data": T_list
        },
        "T_lidar_to_cam_gt": {
            "available": False,
            "rows": 0,
            "cols": 0,
            "data": []
        },
        "img_folder": "images",
        "mask_folder": "processed_masks",
        "pc_folder": "pc",
        "img_format": ".png",
        "pc_format": ".pcd",
        "file_name": file_names,
        "params": params
    }
    
    # Write output if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(calib, f, indent=2)
        print(f"Converted calibration written to: {output_path}")
    
    return calib


def main():
    parser = argparse.ArgumentParser(
        description='Convert Lucid calibration JSON to calib.json format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_lucid_calib.py -i data/lucid/fwc_c/fwc_c.json -o data/lucid_formatted/calib.json
  python scripts/convert_lucid_calib.py -i data/lucid/fnc/fnc.json -o output/calib.json --files 000000 000001
  python scripts/convert_lucid_calib.py -i data/lucid/fwc_c/fwc_c.json -o output/calib.json --roll-offset 5.0 --pitch-offset -1.0
  python scripts/convert_lucid_calib.py -i data/lucid/fwc_c/fwc_c.json -o output/calib.json --tx-offset 0.1 --tz-offset -0.05
        """
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input Lucid calibration JSON file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output calib.json file')
    parser.add_argument('--fx-scale', type=float, default=1.0,
                        help='Scale factor for fx (default: 1.0)')
    parser.add_argument('--fy-scale', type=float, default=1.0,
                        help='Scale factor for fy (default: 1.0)')
    parser.add_argument('--cx-scale', type=float, default=1.0,
                        help='Scale factor for cx (default: 1.0)')
    parser.add_argument('--cy-scale', type=float, default=1.0,
                        help='Scale factor for cy (default: 1.0)')
    parser.add_argument('--roll-offset', type=float, default=0.0,
                        help='Offset to add to roll angle in degrees (default: 0.0)')
    parser.add_argument('--pitch-offset', type=float, default=0.0,
                        help='Offset to add to pitch angle in degrees (default: 0.0)')
    parser.add_argument('--yaw-offset', type=float, default=0.0,
                        help='Offset to add to yaw angle in degrees (default: 0.0)')
    parser.add_argument('--tx-offset', type=float, default=0.0,
                        help='Offset to add to translation x (px) in meters (default: 0.0)')
    parser.add_argument('--ty-offset', type=float, default=0.0,
                        help='Offset to add to translation y (py) in meters (default: 0.0)')
    parser.add_argument('--tz-offset', type=float, default=0.0,
                        help='Offset to add to translation z (pz) in meters (default: 0.0)')
    parser.add_argument('--files', nargs='+', default=None,
                        help='List of file names (default: 000000 000001 000002)')
    parser.add_argument('--template', default=None,
                        help='Path to existing calib.json to use as template for params')
    parser.add_argument('--print-transform', action='store_true',
                        help='Print the computed transformation matrix')
    
    args = parser.parse_args()
    
    calib = convert_lucid_to_calib(
        input_path=args.input,
        output_path=args.output,
        fx_scale=args.fx_scale,
        fy_scale=args.fy_scale,
        cx_scale=args.cx_scale,
        cy_scale=args.cy_scale,
        roll_offset=args.roll_offset,
        pitch_offset=args.pitch_offset,
        yaw_offset=args.yaw_offset,
        tx_offset=args.tx_offset,
        ty_offset=args.ty_offset,
        tz_offset=args.tz_offset,
        file_names=args.files,
        template_path=args.template
    )
    
    if args.print_transform:
        print("\nTransformation matrix (T_lidar_to_cam):")
        T = np.array(calib['T_lidar_to_cam']['data'])
        np.set_printoptions(precision=12, suppress=True)
        print(T)
        
        print("\nIntrinsics (cam_K):")
        K = np.array(calib['cam_K']['data'])
        print(K)
        
        print("\nDistortion (cam_dist):")
        print(calib['cam_dist']['data'])


if __name__ == '__main__':
    main()

