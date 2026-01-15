#!/usr/bin/env python3
"""
Script to test calibration robustness by adding random jitter to T_lidar_to_cam matrix.

Usage:
    python scripts/jitter_calibration.py <path_to_calib.json> [options]

Example:
    python scripts/jitter_calibration.py ./data/cam03/calib.json
    python scripts/jitter_calibration.py ./data/cam03/calib.json --roll 2.0 --pitch 1.5 --yaw 0.5 --tx 0.1 --ty 0.05 --tz 0.1
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import numpy as np
from pathlib import Path


def rotation_matrix_x(angle_rad):
    """Rotation matrix around X axis (roll)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rotation_matrix_y(angle_rad):
    """Rotation matrix around Y axis (pitch)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_z(angle_rad):
    """Rotation matrix around Z axis (yaw)."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])


def generate_jitter(roll_max_deg=0, pitch_max_deg=0, yaw_max_deg=0,
                    tx_max=0, ty_max=0, tz_max=0, seed=None):
    """
    Generate random jitter values within specified bounds.
    
    Args:
        roll_max_deg: Max roll error in degrees (random in [-roll_max, roll_max])
        pitch_max_deg: Max pitch error in degrees
        yaw_max_deg: Max yaw error in degrees
        tx_max: Max translation error in X (meters)
        ty_max: Max translation error in Y (meters)
        tz_max: Max translation error in Z (meters)
        seed: Random seed for reproducibility
    
    Returns:
        dict with roll, pitch, yaw (degrees), tx, ty, tz (meters)
    """
    if seed is not None:
        np.random.seed(seed)
    
    jitter = {
        'roll_deg': np.random.uniform(-roll_max_deg, roll_max_deg) if roll_max_deg > 0 else 0,
        'pitch_deg': np.random.uniform(-pitch_max_deg, pitch_max_deg) if pitch_max_deg > 0 else 0,
        'yaw_deg': np.random.uniform(-yaw_max_deg, yaw_max_deg) if yaw_max_deg > 0 else 0,
        'tx': np.random.uniform(-tx_max, tx_max) if tx_max > 0 else 0,
        'ty': np.random.uniform(-ty_max, ty_max) if ty_max > 0 else 0,
        'tz': np.random.uniform(-tz_max, tz_max) if tz_max > 0 else 0,
    }
    
    return jitter


def apply_jitter_to_transform(T_original, jitter):
    """
    Apply jitter to a 4x4 transformation matrix.
    
    The jitter rotation is applied as: T_jittered = T_jitter @ T_original
    This adds rotation/translation error to the original calibration.
    
    Args:
        T_original: 4x4 numpy array (original transformation matrix)
        jitter: dict with roll_deg, pitch_deg, yaw_deg, tx, ty, tz
    
    Returns:
        T_jittered: 4x4 numpy array (jittered transformation matrix)
    """
    # Convert angles to radians
    roll_rad = np.deg2rad(jitter['roll_deg'])
    pitch_rad = np.deg2rad(jitter['pitch_deg'])
    yaw_rad = np.deg2rad(jitter['yaw_deg'])
    
    # Build rotation jitter matrix (RPY order: roll -> pitch -> yaw)
    R_jitter = rotation_matrix_z(yaw_rad) @ rotation_matrix_y(pitch_rad) @ rotation_matrix_x(roll_rad)
    
    # Build 4x4 jitter transformation matrix
    T_jitter = np.eye(4)
    T_jitter[:3, :3] = R_jitter
    T_jitter[0, 3] = jitter['tx']
    T_jitter[1, 3] = jitter['ty']
    T_jitter[2, 3] = jitter['tz']
    
    # Apply jitter: T_jittered = T_jitter @ T_original
    T_jittered = T_jitter @ T_original
    
    return T_jittered


def load_calib(calib_path):
    """Load calibration JSON file."""
    with open(calib_path, 'r') as f:
        return json.load(f)


def save_calib(calib_data, calib_path):
    """Save calibration JSON file."""
    with open(calib_path, 'w') as f:
        json.dump(calib_data, f, indent=2)


def get_transform_matrix(calib_data):
    """Extract T_lidar_to_cam as numpy array from calib data."""
    data = calib_data['T_lidar_to_cam']['data']
    return np.array(data)


def set_transform_matrix(calib_data, T):
    """Set T_lidar_to_cam in calib data from numpy array."""
    calib_data['T_lidar_to_cam']['data'] = T.tolist()


def print_jitter_info(jitter):
    """Print the jitter values being applied."""
    print("\n" + "=" * 60)
    print("APPLYING CALIBRATION JITTER")
    print("=" * 60)
    
    rot_errors = []
    trans_errors = []
    
    if abs(jitter['roll_deg']) > 1e-6:
        rot_errors.append(f"Roll:  {jitter['roll_deg']:+.4f} deg")
    if abs(jitter['pitch_deg']) > 1e-6:
        rot_errors.append(f"Pitch: {jitter['pitch_deg']:+.4f} deg")
    if abs(jitter['yaw_deg']) > 1e-6:
        rot_errors.append(f"Yaw:   {jitter['yaw_deg']:+.4f} deg")
    
    if abs(jitter['tx']) > 1e-6:
        trans_errors.append(f"TX: {jitter['tx']:+.4f} m")
    if abs(jitter['ty']) > 1e-6:
        trans_errors.append(f"TY: {jitter['ty']:+.4f} m")
    if abs(jitter['tz']) > 1e-6:
        trans_errors.append(f"TZ: {jitter['tz']:+.4f} m")
    
    if rot_errors:
        print("\nRotation Errors:")
        for err in rot_errors:
            print(f"  {err}")
    else:
        print("\nRotation Errors: None")
    
    if trans_errors:
        print("\nTranslation Errors:")
        for err in trans_errors:
            print(f"  {err}")
    else:
        print("\nTranslation Errors: None")
    
    # Summary
    total_rot = np.sqrt(jitter['roll_deg']**2 + jitter['pitch_deg']**2 + jitter['yaw_deg']**2)
    total_trans = np.sqrt(jitter['tx']**2 + jitter['ty']**2 + jitter['tz']**2)
    print(f"\nTotal rotation magnitude: {total_rot:.4f} deg")
    print(f"Total translation magnitude: {total_trans:.4f} m")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Add random jitter to calibration and run lidar2camera calibration.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Random jitter with defaults (2 deg rotation, 0.1m translation max)
  python scripts/jitter_calibration.py ./data/cam03/calib.json
  
  # Custom max jitter values
  python scripts/jitter_calibration.py ./data/cam03/calib.json --roll 5 --pitch 3 --yaw 2 --tx 0.2 --ty 0.1 --tz 0.15
  
  # Only rotation errors
  python scripts/jitter_calibration.py ./data/cam03/calib.json --roll 3 --pitch 3 --yaw 3 --tx 0 --ty 0 --tz 0
  
  # Reproducible with seed
  python scripts/jitter_calibration.py ./data/cam03/calib.json --seed 42
        """
    )
    
    parser.add_argument('calib_path', type=str, help='Path to calib.json file')
    parser.add_argument('--roll', type=float, default=2.0, help='Max roll error in degrees (default: 2.0)')
    parser.add_argument('--pitch', type=float, default=2.0, help='Max pitch error in degrees (default: 2.0)')
    parser.add_argument('--yaw', type=float, default=2.0, help='Max yaw error in degrees (default: 2.0)')
    parser.add_argument('--tx', type=float, default=0.1, help='Max X translation error in meters (default: 0.1)')
    parser.add_argument('--ty', type=float, default=0.1, help='Max Y translation error in meters (default: 0.1)')
    parser.add_argument('--tz', type=float, default=0.1, help='Max Z translation error in meters (default: 0.1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--binary', type=str, default='./bin/run_lidar2camera', 
                        help='Path to lidar2camera binary (default: ./bin/run_lidar2camera)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Only show jitter values, do not modify files or run calibration')
    
    args = parser.parse_args()
    
    # Validate calib path exists
    calib_path = Path(args.calib_path)
    if not calib_path.exists():
        print(f"Error: Calibration file not found: {calib_path}")
        sys.exit(1)
    
    # Validate binary exists (unless dry run)
    binary_path = Path(args.binary)
    if not args.dry_run and not binary_path.exists():
        print(f"Error: Binary not found: {binary_path}")
        print("Make sure to build the project first with ./build.sh")
        sys.exit(1)
    
    # Define backup path
    calib_dir = calib_path.parent
    backup_path = calib_dir / 'calib_original.json'
    
    # Generate jitter
    jitter = generate_jitter(
        roll_max_deg=args.roll,
        pitch_max_deg=args.pitch,
        yaw_max_deg=args.yaw,
        tx_max=args.tx,
        ty_max=args.ty,
        tz_max=args.tz,
        seed=args.seed
    )
    
    # Print jitter info
    print_jitter_info(jitter)
    
    if args.dry_run:
        print("Dry run mode - no files modified, calibration not run.")
        return
    
    try:
        # Load original calibration
        print(f"Loading calibration from: {calib_path}")
        calib_data = load_calib(calib_path)
        
        # Backup original calibration
        print(f"Creating backup at: {backup_path}")
        shutil.copy2(calib_path, backup_path)
        
        # Get original transform matrix
        T_original = get_transform_matrix(calib_data)
        print("\nOriginal T_lidar_to_cam:")
        print(T_original)
        
        # Apply jitter
        T_jittered = apply_jitter_to_transform(T_original, jitter)
        print("\nJittered T_lidar_to_cam:")
        print(T_jittered)
        
        # Update and save calibration
        set_transform_matrix(calib_data, T_jittered)
        save_calib(calib_data, calib_path)
        print(f"\nSaved jittered calibration to: {calib_path}")
        
        # Run calibration binary
        print("\n" + "=" * 60)
        print("RUNNING CALIBRATION")
        print("=" * 60)
        cmd = [str(binary_path), str(calib_path)]
        print(f"Command: {' '.join(cmd)}\n")
        
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
        
        print("\n" + "=" * 60)
        print(f"CALIBRATION FINISHED (exit code: {result.returncode})")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during calibration: {e}")
        raise
        
    finally:
        # Always restore original calibration
        if backup_path.exists():
            print(f"\nRestoring original calibration from: {backup_path}")
            shutil.copy2(backup_path, calib_path)
            print("Original calibration restored.")
            # Delete backup file
            backup_path.unlink()
            print(f"Deleted backup file: {backup_path}")
        else:
            print("\nWarning: Backup file not found, could not restore original calibration!")


if __name__ == '__main__':
    main()
