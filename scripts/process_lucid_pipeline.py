#!/usr/bin/env python3
"""
Pipeline script to process Lucid data from datasampleformat to calibration-ready format.

This script:
1. Converts data structure from datasampleformat to data/camXX format
2. Converts lucid_calib.json files to calib.json using convert_lucid_calib.py
3. Processes masks if they exist using processed_mask.py
4. Runs calibration (./bin/run_lidar2camera) on each camera
5. Saves outputs (extrinsic.txt, refined_proj_seg.png, refined_proj.png) to output directory

Usage:
    python scripts/process_lucid_pipeline.py -i datasampleformat -o output
    python scripts/process_lucid_pipeline.py -i datasampleformat -o output --skip-calibration
"""

import argparse
import json
import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Tuple


def find_scenes(input_dir: Path) -> List[Path]:
    """Find all scene directories in the input directory."""
    scenes = []
    for item in input_dir.iterdir():
        # TODO: go through all folders not just scene
        if item.is_dir():
            scenes.append(item)
    return sorted(scenes)


def find_cameras(scene_dir: Path) -> List[str]:
    """Find all camera directories in a scene."""
    cameras = []
    for item in scene_dir.iterdir():
        # TODO: all subdirectories within scenes are assumed to be cameras
        if item.is_dir() and item.name.startswith('cam'):
            cameras.append(item.name)
    return sorted(cameras)


def find_pcd_files(scene_dir: Path) -> List[Path]:
    """Find all PCD files in a scene directory (not in subdirectories)."""
    pcd_files = []
    for item in scene_dir.iterdir():
        if item.is_file() and item.suffix == '.pcd':
            pcd_files.append(item)
    return sorted(pcd_files)


def get_file_name_from_path(pcd_path: Path) -> str:
    """Extract file name (without extension) from PCD path."""
    return pcd_path.stem  # e.g., "000000" from "000000.pcd"


def create_output_structure(output_dir: Path, scene_name: str, camera_name: str) -> Dict[str, Path]:
    """Create directory structure for a camera in the output format.
    
    Format: output/scene1_cam02/ or output/cam02/ if scene_name is None
    """
    if scene_name:
        cam_dir = output_dir / f"{scene_name}_{camera_name}"
    else:
        cam_dir = output_dir / camera_name
    cam_dir.mkdir(parents=True, exist_ok=True)
    
    images_dir = cam_dir / "images"
    masks_dir = cam_dir / "masks"
    pc_dir = cam_dir / "pc"
    processed_masks_dir = cam_dir / "processed_masks"
    
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)
    pc_dir.mkdir(exist_ok=True)
    processed_masks_dir.mkdir(exist_ok=True)
    
    return {
        'cam_dir': cam_dir,
        'images_dir': images_dir,
        'masks_dir': masks_dir,
        'pc_dir': pc_dir,
        'processed_masks_dir': processed_masks_dir
    }


def copy_images_and_pcds(scene_dir: Path, camera_name: str, 
                         dirs: Dict[str, Path], pcd_files: List[Path],
                         file_names: List[str]):
    """Copy images and PCD files from scene to output structure.
    
    Images are copied from the camera directory and all its subdirectories.
    All found image files (png, jpg, etc.) are copied to the images directory.
    """
    # Copy all images from camera directory and subdirectories
    cam_source_dir = scene_dir / camera_name
    
    if not cam_source_dir.exists():
        print(f"  Warning: Camera directory does not exist: {cam_source_dir}")
        return
    
    # Find all image files in camera directory and subdirectories
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = []
    
    # Search in the camera directory and all subdirectories
    for item in cam_source_dir.rglob('*'):
        if item.is_file() and item.suffix in image_extensions:
            image_files.append(item)
    
    # Sort image files for consistent ordering
    image_files.sort()
    
    if len(image_files) == 0:
        print(f"  Warning: No image files found in {cam_source_dir}")
    else:
        # Copy each image file to the images directory
        for image_file in image_files:
            # Use the original filename for the destination
            dest_image = dirs['images_dir'] / image_file.name
            shutil.copy2(image_file, dest_image)
            print(f"  Copied image: {image_file.name} -> {dest_image}")
    
    # Copy PCD files
    for pcd_file, file_name in zip(pcd_files, file_names):
        dest_pcd = dirs['pc_dir'] / f"{file_name}.pcd"
        shutil.copy2(pcd_file, dest_pcd)
        print(f"  Copied PCD to {dest_pcd}")


def convert_calib_file(lucid_calib_path: Path, output_calib_path: Path, 
                       camera_name: str, file_names: List[str],
                       convert_script: Path = None) -> bool:
    """Convert lucid_calib.json to calib.json using convert_lucid_calib.py."""
    if convert_script is None:
        # Assume script is in scripts/ directory relative to this script
        script_dir = Path(__file__).parent
        convert_script = script_dir / "convert_lucid_calib.py"
    
    if not convert_script.exists():
        print(f"Error: convert_lucid_calib.py not found at {convert_script}")
        return False
    
    # Build command
    cmd = [
        'python3', str(convert_script),
        '-i', str(lucid_calib_path),
        '-o', str(output_calib_path),
        '-c', camera_name,
        '--files'
    ] + file_names
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Converted calibration file: {output_calib_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error converting calibration file: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def process_masks(masks_dir: Path, processed_masks_dir: Path,
                  processed_mask_script: Path = None) -> bool:
    """Process masks using processed_mask.py if masks directory is not empty."""
    # Check if masks directory exists and has subdirectories
    if not masks_dir.exists():
        return True  # No masks to process, not an error
    
    mask_subdirs = [d for d in masks_dir.iterdir() if d.is_dir()]
    if len(mask_subdirs) == 0:
        print(f"  No mask subdirectories found in {masks_dir}, skipping mask processing")
        return True
    
    if processed_mask_script is None:
        # Look for processed_mask.py in the project root
        project_root = Path(__file__).parent.parent
        processed_mask_script = project_root / "processed_mask.py"
    
    if not processed_mask_script.exists():
        print(f"Warning: processed_mask.py not found at {processed_mask_script}, skipping mask processing")
        return False
    
    # Remove processed_masks_dir if it exists (processed_mask.py requires it to not exist)
    if processed_masks_dir.exists():
        print(f"  Removing existing processed_masks directory: {processed_masks_dir}")
        shutil.rmtree(processed_masks_dir)
    
    # Build command
    cmd = [
        'python3', str(processed_mask_script),
        '-i', str(masks_dir),
        '-o', str(processed_masks_dir)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Processed masks: {processed_masks_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error processing masks: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def run_calibration(calib_json_path: Path, executable_path: Path,
                    output_dir: Path, scene_name: str, camera_name: str,
                    file_index: int = 0) -> Tuple[bool, Path]:
    """
    Run calibration executable on a calib.json file.
    
    Args:
        calib_json_path: Path to calib.json file
        executable_path: Path to ./bin/run_lidar2camera executable
        output_dir: Base output directory for results
        scene_name: Name of the scene (e.g., "scene1")
        camera_name: Name of the camera (e.g., "cam02")
        file_index: Index of the file being processed (for naming output dir)
    
    Returns:
        Tuple of (success: bool, output_path: Path)
    """
    # Create output directory for this calibration run
    # Format: output/scene1_cam02_000000/ or output/scene1_cam02/ if single file
    calib_data = json.load(open(calib_json_path))
    file_names = calib_data.get('file_name', ['000000'])
    
    if len(file_names) == 1:
        output_subdir_name = f"{scene_name}_{camera_name}_{file_names[0]}"
    else:
        # For multiple files, we'll process each one separately
        output_subdir_name = f"{scene_name}_{camera_name}_{file_names[file_index]}"
    
    output_subdir = output_dir / "calibration_results" / output_subdir_name
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    # Get project root (parent of bin/)
    if executable_path.parent.name == 'bin':
        project_root = executable_path.parent.parent
    else:
        project_root = Path.cwd()
    
    # Change to project root before running (calibration outputs to current directory)
    original_cwd = Path.cwd()
    try:
        os.chdir(project_root)
        
        # Run calibration
        cmd = [str(executable_path), str(calib_json_path)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Copy outputs to output directory
        output_files = ['extrinsic.txt', 'refined_proj_seg.png', 'refined_proj.png']
        copied_files = []
        
        for filename in output_files:
            src = project_root / filename
            if src.exists():
                dst = output_subdir / filename
                shutil.copy2(src, dst)
                copied_files.append(filename)
        
        # Also copy init_proj.png and init_proj_seg.png if they exist
        for filename in ['init_proj.png', 'init_proj_seg.png']:
            src = project_root / filename
            if src.exists():
                dst = output_subdir / filename
                shutil.copy2(src, dst)
                copied_files.append(filename)
        
        print(f"  Calibration complete. Outputs saved to: {output_subdir}")
        print(f"    Copied files: {', '.join(copied_files)}")
        return True, output_subdir
        
    except subprocess.CalledProcessError as e:
        print(f"  Error running calibration: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False, output_subdir
    finally:
        os.chdir(original_cwd)


def process_scene(scene_dir: Path, output_dir: Path, 
                  convert_script: Path, processed_mask_script: Path,
                  executable_path: Path, skip_calibration: bool = False):
    """Process a single scene: convert structure, process masks, run calibration."""
    scene_name = scene_dir.name
    print(f"\n{'='*60}")
    print(f"Processing scene: {scene_name}")
    print(f"{'='*60}")
    
    # Find PCD files in scene
    pcd_files = find_pcd_files(scene_dir)
    if len(pcd_files) == 0:
        print(f"  Warning: No PCD files found in {scene_dir}")
        return
    
    file_names = [get_file_name_from_path(pcd) for pcd in pcd_files]
    print(f"  Found {len(pcd_files)} PCD files: {file_names}")
    
    # Find cameras
    cameras = find_cameras(scene_dir)
    if len(cameras) == 0:
        print(f"  Warning: No camera directories found in {scene_dir}")
        return
    
    print(f"  Found {len(cameras)} cameras: {cameras}")
    
    # Process each camera
    for camera_name in cameras:
        print(f"\n  Processing camera: {camera_name}")
        
        # Create output structure
        dirs = create_output_structure(output_dir, scene_name, camera_name)
        
        # Copy images and PCD files
        copy_images_and_pcds(scene_dir, camera_name, dirs, pcd_files, file_names)
        
        # Convert calibration file
        lucid_calib_path = scene_dir / camera_name / "lucid_calib.json"
        output_calib_path = dirs['cam_dir'] / "calib.json"
        
        if not lucid_calib_path.exists():
            print(f"  Warning: lucid_calib.json not found at {lucid_calib_path}")
            continue
        
        if not convert_calib_file(lucid_calib_path, output_calib_path, camera_name, 
                                 file_names, convert_script):
            print(f"  Failed to convert calibration file for {camera_name}")
            continue
        
        # Process masks if they exist
        process_masks(dirs['masks_dir'], dirs['processed_masks_dir'], processed_mask_script)
        
        # Update calib.json to use processed_masks if they exist
        if dirs['processed_masks_dir'].exists():
            mask_subdirs = [d for d in dirs['processed_masks_dir'].iterdir() if d.is_dir()]
            if len(mask_subdirs) > 0:
                calib_data = json.load(open(output_calib_path))
                calib_data['mask_folder'] = 'processed_masks'
                json.dump(calib_data, open(output_calib_path, 'w'), indent=2)
                print(f"  Updated calib.json to use processed_masks")
        
        # Run calibration if not skipped
        if not skip_calibration:
            print(f"  Running calibration...")
            # For now, run calibration once per camera (with all files in file_name list)
            # If you need separate calibration runs per file, this needs to be adjusted
            success, output_path = run_calibration(
                output_calib_path, executable_path, output_dir, 
                scene_name, camera_name, file_index=0
            )
        else:
            print(f"  Skipping calibration (--skip-calibration flag)")


def main():
    parser = argparse.ArgumentParser(
        description='Process Lucid data pipeline: convert format, process masks, run calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output
  
  # Skip calibration step
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output --skip-calibration
  
  # Specify custom executable path
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output --executable ./bin/run_lidar2camera
        """
    )
    parser.add_argument('-i', '--input', required=True, type=Path,
                        help='Input directory containing scene folders (e.g., datasampleformat)')
    parser.add_argument('-o', '--output', required=True, type=Path,
                        help='Output directory for converted data and calibration results')
    parser.add_argument('--convert-script', type=Path, default=None,
                        help='Path to convert_lucid_calib.py script (default: scripts/convert_lucid_calib.py)')
    parser.add_argument('--processed-mask-script', type=Path, default=None,
                        help='Path to processed_mask.py script (default: processed_mask.py in project root)')
    parser.add_argument('--executable', type=Path, default=Path('./bin/run_lidar2camera'),
                        help='Path to calibration executable (default: ./bin/run_lidar2camera)')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip the calibration step (only convert format and process masks)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        return 1
    
    if not args.executable.exists():
        print(f"Error: Calibration executable does not exist: {args.executable}")
        print(f"  Please build the project first or specify the correct path with --executable")
        return 1
    
    # Set default script paths
    if args.convert_script is None:
        args.convert_script = Path(__file__).parent / "convert_lucid_calib.py"
    if args.processed_mask_script is None:
        args.processed_mask_script = Path(__file__).parent.parent / "processed_mask.py"
    
    # Validate scripts exist
    if not args.convert_script.exists():
        print(f"Error: convert_lucid_calib.py not found at {args.convert_script}")
        return 1
    
    # Find scenes
    scenes = find_scenes(args.input)
    if len(scenes) == 0:
        print(f"Error: No scene directories found in {args.input}")
        return 1
    
    print(f"Found {len(scenes)} scene(s): {[s.name for s in scenes]}")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Process each scene
    for scene_dir in scenes:
        process_scene(
            scene_dir, args.output, 
            args.convert_script, args.processed_mask_script,
            args.executable, args.skip_calibration
        )
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Results saved to: {args.output}")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    exit(main())
