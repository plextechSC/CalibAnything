#!/usr/bin/env python3
"""
Convert Lucid calibration JSON format to calib.json format.

Usage:
    python scripts/convert_lucid_calib.py -i data/lucid/fnc/fnc.json -o output/calib.json
    python scripts/convert_lucid_calib.py -i data/lucid/fwc_l/fwc_l.json -o output/calib.json
    python scripts/convert_lucid_calib.py -i input.json -o output/calib.json -c FWC_R
"""

import argparse
import json
import re
import numpy as np
from scipy.spatial.transform import Rotation as R


# =============================================================================
# Camera Configuration
# =============================================================================

# Camera mapping: cam number -> camera name
CAMERA_MAP = {
    'cam-02': 'FWC_C',  # Front Wide Camera Center
    'cam-03': 'FNC',    # Front Narrow Camera
    'cam-04': 'RNC_R',  # Rear Narrow Camera Right
    'cam-05': 'FWC_R',  # Front Wide Camera Right
    'cam-06': 'RNC_C',  # Rear Narrow Camera Center
    'cam-07': 'FWC_L',  # Front Wide Camera Left
    'cam-08': 'RNC_L',  # Rear Narrow Camera Left
}

# Valid camera names (derived from CAMERA_MAP)
VALID_CAMERA_NAMES = list(CAMERA_MAP.values())

# Wide camera Y offset values
WIDE_OFFSET_Y = {
    'FWC_C': 500.0,
    'FWC_L': 300.0,
    'FWC_R': 300.0,
}


def detect_camera_from_path(path: str) -> str | None:
    """
    Try to detect camera name from a file path.
    
    Searches for camera names (FWC_C, FNC, etc.) or camera numbers (cam-05, cam05)
    in the given path.
    
    Args:
        path: File path to search for camera identifier
        
    Returns:
        Camera name (e.g., 'FWC_R') if found, None otherwise
    """
    path_upper = path.upper()
    
    # First, try to find camera names directly (FWC_C, FNC, RNC_R, etc.)
    for name in VALID_CAMERA_NAMES:
        # Match camera name with word boundaries (e.g., /fwc_c/ or _fwc_c. or fwc_c.json)
        pattern = rf'[/_\\]?{re.escape(name)}[/_\\.]'
        if re.search(pattern, path_upper):
            return name
    
    # Try to find cam-XX or camXX patterns
    cam_match = re.search(r'CAM[-_]?(\d{1,2})', path_upper)
    if cam_match:
        cam_num = cam_match.group(1).zfill(2)
        cam_key = f'cam-{cam_num}'
        if cam_key in CAMERA_MAP:
            return CAMERA_MAP[cam_key]
    
    return None


def parse_camera_identifier(camera_str: str) -> str:
    """
    Parse a camera identifier string and return the normalized camera name.
    
    Accepts:
        - Camera names: 'FWC_C', 'fwc_c', 'FNC', etc.
        - Camera numbers: 'cam-02', 'cam-05', '02', '5', etc.
    
    Args:
        camera_str: Camera identifier string
        
    Returns:
        Normalized camera name (e.g., 'FWC_R')
        
    Raises:
        ValueError: If camera identifier is not recognized
    """
    camera_input = camera_str.strip().upper()
    
    # Check if input is a camera number with 'cam-' prefix
    if camera_input.startswith('CAM-'):
        camera_key = camera_input.lower()
        if camera_key not in CAMERA_MAP:
            raise ValueError(
                f"Unknown camera number: {camera_str}. "
                f"Valid options: {list(CAMERA_MAP.keys())}"
            )
        return CAMERA_MAP[camera_key]
    
    # Check if input is just a number (e.g., "02", "2", "05")
    if camera_input.isdigit() or (len(camera_input) == 2 and camera_input[0] == '0'):
        camera_key = f"cam-{camera_input.zfill(2)}"
        if camera_key not in CAMERA_MAP:
            raise ValueError(
                f"Unknown camera number: {camera_str}. Valid options: 02-08"
            )
        return CAMERA_MAP[camera_key]
    
    # Assume it's a camera name (e.g., "FWC_C", "fwc_c")
    if camera_input not in VALID_CAMERA_NAMES:
        raise ValueError(
            f"Unknown camera name: {camera_str}. "
            f"Valid options: {VALID_CAMERA_NAMES}"
        )
    return camera_input


def resolve_camera(camera_arg: str | None, input_path: str, output_path: str) -> str:
    """
    Resolve the camera name from explicit argument or by auto-detecting from paths.
    
    Args:
        camera_arg: Explicit camera identifier (or None for auto-detect)
        input_path: Input file path (used for auto-detection)
        output_path: Output file path (used for auto-detection)
        
    Returns:
        Resolved camera name (e.g., 'FWC_R')
        
    Raises:
        ValueError: If camera cannot be determined
    """
    if camera_arg:
        return parse_camera_identifier(camera_arg)
    
    # Try to auto-detect from input path first, then output path
    camera_name = detect_camera_from_path(input_path)
    if not camera_name:
        camera_name = detect_camera_from_path(output_path)
    
    if not camera_name:
        raise ValueError(
            f"Unable to detect camera type from paths.\n"
            f"  Input: {input_path}\n"
            f"  Output: {output_path}\n"
            f"Please specify camera with -c/--camera. Valid options:\n"
            f"  Camera names: {VALID_CAMERA_NAMES}\n"
            f"  Camera numbers: {list(CAMERA_MAP.keys())}"
        )
    
    return camera_name


def get_camera_settings(camera_name: str) -> tuple[bool, float]:
    """
    Get camera settings based on camera name.
    
    Args:
        camera_name: Normalized camera name (e.g., 'FWC_R')
        
    Returns:
        Tuple of (is_wide_camera, wide_offset_y)
    """
    # Wide cameras have 'W' in name, narrow cameras have 'N'
    is_wide_camera = 'W' in camera_name
    wide_offset_y = WIDE_OFFSET_Y.get(camera_name, 0.0)
    return is_wide_camera, wide_offset_y


# =============================================================================
# Transformation Utilities
# =============================================================================

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
                           cx_offset: float = 0.0, cy_offset: float = 0.0,
                           roll_offset: float = 0.0, pitch_offset: float = 0.0, yaw_offset: float = 0.0,
                           tx_offset: float = 0.0, ty_offset: float = 0.0, tz_offset: float = 0.0,
                           file_names: list = None,
                           template_path: str = None,
                           is_wide_camera: bool = False,
                           wide_offset_y: float = 0.0,
                           custom_params: dict = None) -> dict:
    """
    Convert Lucid calibration JSON to calib.json format.
    
    Args:
        input_path: Path to input Lucid calibration JSON file
        output_path: Path to output calib.json file (optional, if None returns dict only)
        roll_offset: Offset to add to roll angle in degrees (default 0.0)
        pitch_offset: Offset to add to pitch angle in degrees (default 0.0)
        yaw_offset: Offset to add to yaw angle in degrees (default 0.0)
        tx_offset: Offset to add to translation x (px) in meters (default 0.0)
        ty_offset: Offset to add to translation y (py) in meters (default 0.0)
        tz_offset: Offset to add to translation z (pz) in meters (default 0.0)
        file_names: List of file names to process (default ["000000", "000001", "000002"])
        template_path: Path to existing calib.json to use as template for params
        is_wide_camera: If True, use wide camera intrinsics handling (default False)
        wide_offset_y: Y offset for wide camera cy calculation
        custom_params: Custom params dict to use instead of defaults (overrides template_path)
    
    Returns:
        The converted calibration dictionary
    """
    # Load input calibration
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    intr = data['intrinsic_params']
    extr = data['extrinsic_params']
    
    # Extract intrinsics - handle wide cameras differently from normal cameras
    if is_wide_camera:
        # Wide camera intrinsics handling (matches cloud_visualizer.py lines 109-129)
        # For wide cameras (fwc_c, fwc_l, fwc_r):
        # - fx and fy use scale of 1 (not 2)
        # - cx = 2 * original_cx
        # - cy = 2 * original_cy - offset (offset is 300 for fwc_l/fwc_r, 500/964 for fwc_c)
        fx = intr['fx']  # No scaling for wide cameras
        fy = intr['fy']  # No scaling for wide cameras
        cx = 2 * intr['cx']  # Double the cx
        cy = 2 * intr['cy'] - wide_offset_y  # Double the cy and subtract offset
    else:
        # Normal camera intrinsics (default behavior)
        fx = intr['fx']
        fy = intr['fy']
        cx = intr['cx'] + cx_offset
        cy = intr['cy'] + cy_offset
    
    # Build 3x3 camera matrix
    cam_K = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]
    
    # distortion coefficients
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
    
    ### BELOW IS CODE FOR OUTPUTTING THE CALIBRATION FILE

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
    
    # Priority: custom_params > template_path > defaults
    if custom_params:
        params = custom_params
    elif template_path:
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
  # Auto-detect camera from path (preferred):
  python scripts/convert_lucid_calib.py -i data/lucid/fnc/fnc.json -o output/calib.json
  python scripts/convert_lucid_calib.py -i data/lucid/fwc_l/fwc_l.json -o output/calib.json
  python scripts/convert_lucid_calib.py -i input.json -o data/cam05/calib.json
  
  # Explicit camera name (case-insensitive):
  python scripts/convert_lucid_calib.py -i input.json -o output/calib.json -c FNC
  python scripts/convert_lucid_calib.py -i input.json -o output/calib.json -c fwc_l
  
  # Explicit camera number:
  python scripts/convert_lucid_calib.py -i input.json -o output/calib.json -c cam-03
  python scripts/convert_lucid_calib.py -i input.json -o output/calib.json -c 05
  
  # With additional options:
  python scripts/convert_lucid_calib.py -i data/lucid/fnc/fnc.json -o output/calib.json --files 000000 000001
  python scripts/convert_lucid_calib.py -i data/lucid/fnc/fnc.json -o output/calib.json --roll-offset 5.0

  # Camera mapping:
  #   cam-02 = FWC_C (Front Wide Camera Center)  -> wide_offset_y=500
  #   cam-03 = FNC   (Front Narrow Camera)
  #   cam-04 = RNC_R (Rear Narrow Camera Right)
  #   cam-05 = FWC_R (Front Wide Camera Right)   -> wide_offset_y=300
  #   cam-06 = RNC_C (Rear Narrow Camera Center)
  #   cam-07 = FWC_L (Front Wide Camera Left)    -> wide_offset_y=300
  #   cam-08 = RNC_L (Rear Narrow Camera Left)
        """
    )
    parser.add_argument('-i', '--input', required=True,
                        help='Input Lucid calibration JSON file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output calib.json file')
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
    parser.add_argument('--params-file', default=None,
                        help='Path to JSON file containing file_names and params (e.g., test_params.json). Overrides --files and --template if those keys exist in the file.')
    parser.add_argument('--print-transform', action='store_true',
                        help='Print the computed transformation matrix')
    parser.add_argument('--cx-offset', type=float, default=0.0,
                        help='Offset to add to cx (default: 0.0)')
    parser.add_argument('--cy-offset', type=float, default=0.0,
                        help='Offset to add to cy (default: 0.0)')
    parser.add_argument('--camera', '-c', type=str, default=None,
                        help='Camera identifier: number (cam-02 to cam-08, or just 02-08) or name (FWC_C, FNC, RNC_R, etc.). If not provided, will try to auto-detect from input/output path.')
    
    args = parser.parse_args()

    # Load params file if provided (overrides --files and provides custom_params)
    file_names = args.files
    custom_params = None
    
    if args.params_file:
        try:
            with open(args.params_file, 'r') as f:
                params_data = json.load(f)
                # Override file_names if present in params file
                if 'file_names' in params_data:
                    file_names = params_data['file_names']
                # Load custom params if present
                if 'params' in params_data:
                    custom_params = params_data['params']
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load params file '{args.params_file}': {e}")

    # Resolve camera from argument or auto-detect from paths
    camera_name = resolve_camera(args.camera, args.input, args.output)
    is_wide_camera, wide_offset_y = get_camera_settings(camera_name)
    
    print(f"Camera: {camera_name} | Wide: {is_wide_camera} | wide_offset_y: {wide_offset_y}")
    
    calib = convert_lucid_to_calib(
        input_path=args.input,
        output_path=args.output,
        cx_offset=args.cx_offset,
        cy_offset=args.cy_offset,
        roll_offset=args.roll_offset,
        pitch_offset=args.pitch_offset,
        yaw_offset=args.yaw_offset,
        tx_offset=args.tx_offset,
        ty_offset=args.ty_offset,
        tz_offset=args.tz_offset,
        file_names=file_names,
        template_path=args.template,
        is_wide_camera=is_wide_camera,
        wide_offset_y=wide_offset_y,
        custom_params=custom_params,
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

