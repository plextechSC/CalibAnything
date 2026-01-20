#!/usr/bin/env python3
"""
Pipeline script to process Lucid data from datasampleformat to calibration-ready format.

This script:
1. Clones and sets up lucid-sam for automatic mask generation
2. Converts data structure from datasampleformat to data/camXX format
3. Converts lucid_calib.json files to calib.json using convert_lucid_calib.py
4. Generates masks using SAM2 (if not already present)
5. Processes masks using processed_mask.py
6. Runs calibration (./bin/run_lidar2camera) on each camera
7. Saves outputs (extrinsic.txt, refined_proj_seg.png, refined_proj.png) to output directory

Usage:
    python scripts/process_lucid_pipeline.py -i datasampleformat -o output
    python scripts/process_lucid_pipeline.py -i datasampleformat -o output --skip-calibration
    python scripts/process_lucid_pipeline.py -i datasampleformat -o output --force-sam
"""

import argparse
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# =============================================================================
# lucid-sam Configuration
# =============================================================================

LUCID_SAM_REPO = "https://github.com/plextechSC/lucid-sam.git"
LUCID_SAM_DIR = Path(__file__).parent.parent / "external" / "lucid-sam"

# Default max dimension for SAM when not using CUDA (to reduce memory usage on CPU/MPS)
SAM_NON_CUDA_MAX_DIMENSION = 1024


# =============================================================================
# Virtual Environment Check
# =============================================================================

def check_virtual_environment() -> bool:
    """Check if the script is running inside a virtual environment.
    
    Returns:
        True if in a virtual environment, False otherwise
    """
    # Check if sys.prefix differs from sys.base_prefix (standard venv detection)
    in_venv = sys.prefix != sys.base_prefix
    
    # Also check for VIRTUAL_ENV environment variable (set by most venv activations)
    has_venv_var = 'VIRTUAL_ENV' in os.environ
    
    return in_venv or has_venv_var


def enforce_virtual_environment():
    """Exit with an error if not running in a virtual environment."""
    if not check_virtual_environment():
        print("=" * 60)
        print("ERROR: This script must be run inside a virtual environment!")
        print("=" * 60)
        print()
        print("Please create and activate a virtual environment first:")
        print()
        print("  # Create virtual environment")
        print("  python3 -m venv .venv")
        print()
        print("  # Activate it")
        print("  source .venv/bin/activate  # On macOS/Linux")
        print("  .venv\\Scripts\\activate     # On Windows")
        print()
        print("  # Install dependencies")
        print("  pip install -r scripts/requirements.txt")
        print()
        print("  # Then run this script again")
        print("  python scripts/process_lucid_pipeline.py -i <input> -o <output>")
        print()
        sys.exit(1)


# =============================================================================
# lucid-sam Setup Functions
# =============================================================================

def clone_lucid_sam(lucid_sam_dir: Path = LUCID_SAM_DIR) -> bool:
    """Clone the lucid-sam repository if it doesn't exist.
    
    Args:
        lucid_sam_dir: Directory where lucid-sam should be cloned
        
    Returns:
        True if successful (or already exists), False on error
    """
    if lucid_sam_dir.exists():
        print(f"  lucid-sam already exists at {lucid_sam_dir}")
        return True
    
    print(f"  Cloning lucid-sam repository to {lucid_sam_dir}...")
    lucid_sam_dir.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        cmd = ['git', 'clone', LUCID_SAM_REPO, str(lucid_sam_dir)]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Successfully cloned lucid-sam")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error cloning lucid-sam: {e}")
        print(f"  stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  Error: git is not installed. Please install git and try again.")
        return False


def setup_lucid_sam(lucid_sam_dir: Path = LUCID_SAM_DIR) -> bool:
    """Set up lucid-sam by running setup.sh (creates venv, installs deps, downloads checkpoints).
    
    Args:
        lucid_sam_dir: Directory where lucid-sam is located
        
    Returns:
        True if successful, False on error
    """
    setup_script = lucid_sam_dir / "setup.sh"
    venv_dir = lucid_sam_dir / ".venv"
    
    if not setup_script.exists():
        print(f"  Error: setup.sh not found at {setup_script}")
        return False
    
    # Check if already set up (venv exists and has sam2 installed)
    if venv_dir.exists():
        sam2_check = venv_dir / "lib"
        if sam2_check.exists():
            print(f"  lucid-sam environment already set up at {venv_dir}")
            return True
    
    print(f"  Setting up lucid-sam environment (this may take a while)...")
    print(f"  Running: source {setup_script}")
    
    # Run setup.sh by sourcing it in bash
    # We need to source it to properly set up the venv and install dependencies
    try:
        # Use bash to source the setup script
        cmd = f'cd "{lucid_sam_dir}" && source setup.sh'
        result = subprocess.run(
            ['bash', '-c', cmd],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(lucid_sam_dir)
        )
        print(f"  Successfully set up lucid-sam environment")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error setting up lucid-sam: {e}")
        print(f"  stdout: {e.stdout}")
        print(f"  stderr: {e.stderr}")
        return False


def ensure_lucid_sam_ready(lucid_sam_dir: Path = LUCID_SAM_DIR) -> bool:
    """Ensure lucid-sam is cloned and set up.
    
    Args:
        lucid_sam_dir: Directory for lucid-sam
        
    Returns:
        True if lucid-sam is ready to use, False otherwise
    """
    print("\n  Checking lucid-sam setup...")
    
    if not clone_lucid_sam(lucid_sam_dir):
        return False
    
    if not setup_lucid_sam(lucid_sam_dir):
        return False
    
    return True


# =============================================================================
# Scene/Camera Discovery Functions
# =============================================================================

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


# =============================================================================
# SAM Mask Generation Functions
# =============================================================================

def generate_sam_masks_for_image(
    image_path: Path, 
    output_masks_dir: Path,
    lucid_sam_dir: Path = LUCID_SAM_DIR,
    max_dimension: Optional[int] = None
) -> bool:
    """Generate SAM masks for a single image using lucid-sam.
    
    Args:
        image_path: Path to the input image
        output_masks_dir: Directory to save masks (masks will be in output_masks_dir/image_stem/)
        lucid_sam_dir: Path to lucid-sam directory
        max_dimension: Maximum image dimension for SAM processing (None = no limit)
        
    Returns:
        True if successful, False on error
    """
    image_stem = image_path.stem  # e.g., "000000" from "000000.png"
    
    # Convert to absolute paths since subprocess runs from lucid-sam directory
    image_path_abs = image_path.resolve()
    mask_output_dir_abs = (output_masks_dir / image_stem).resolve()
    lucid_sam_dir_abs = lucid_sam_dir.resolve()
    
    # Build the command to run SAM using lucid-sam's venv
    venv_python = lucid_sam_dir_abs / ".venv" / "bin" / "python"
    sam_script = lucid_sam_dir_abs / "sam.py"
    
    if not venv_python.exists():
        print(f"    Error: lucid-sam venv not found at {venv_python}")
        return False
    
    if not sam_script.exists():
        print(f"    Error: sam.py not found at {sam_script}")
        return False
    
    # We need to call process_image_with_sam from sam.py
    # Create a small wrapper command that imports and calls the function
    # Auto-downsample if not using CUDA to reduce memory usage
    # Then upscale masks back to original resolution
    python_code = f'''
import sys
import os
import torch
import cv2
sys.path.insert(0, "{lucid_sam_dir_abs}")
from sam import process_image_with_sam
from models import SAM2Model

# Read original image dimensions
original_img = cv2.imread("{image_path_abs}")
if original_img is None:
    raise FileNotFoundError(f"Failed to read image: {image_path_abs}")
orig_h, orig_w = original_img.shape[:2]
print(f"Original image size: {{orig_w}}x{{orig_h}}")

# Determine max_dimension: use provided value, or auto-downsample if not on CUDA
max_dim = {max_dimension if max_dimension else 'None'}
needs_upscale = False
if max_dim is None and not torch.cuda.is_available():
    max_dim = {SAM_NON_CUDA_MAX_DIMENSION}
    if max(orig_h, orig_w) > max_dim:
        needs_upscale = True
        print(f"CUDA not available, processing at max dimension {{max_dim}} (will upscale masks)")
elif max_dim is not None and max(orig_h, orig_w) > max_dim:
    needs_upscale = True
    print(f"Processing at max dimension {{max_dim}} (will upscale masks)")

masks = process_image_with_sam(
    image_path="{image_path_abs}",
    selected_model=SAM2Model.LARGE,
    visualize=False,
    output_masks=True,
    visualization_output_path=None,
    output_masks_dir="{mask_output_dir_abs}",
    mask_name_digits=4,
    mask_start_index=1,
    max_dimension=max_dim
)
print(f"Generated {{len(masks)}} masks")

# Upscale masks to original resolution if needed
if needs_upscale:
    print(f"Upscaling masks to original resolution {{orig_w}}x{{orig_h}}...")
    mask_dir = "{mask_output_dir_abs}"
    for mask_file in os.listdir(mask_dir):
        if mask_file.endswith('.png'):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                # Use INTER_NEAREST to preserve binary mask values
                upscaled = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(mask_path, upscaled)
    print(f"Upscaled {{len(os.listdir(mask_dir))}} masks")
'''
    
    try:
        result = subprocess.run(
            [str(venv_python), '-c', python_code],
            check=True,
            capture_output=True,
            text=True,
            cwd=str(lucid_sam_dir_abs)
        )
        # Print SAM output
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                print(f"    [SAM] {line}")
        print(f"    Generated masks for {image_path.name} -> {mask_output_dir_abs}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"    Error generating masks for {image_path.name}: {e}")
        if e.stdout and e.stdout.strip():
            print(f"    stdout: {e.stdout}")
        if e.stderr and e.stderr.strip():
            print(f"    stderr: {e.stderr}")
        return False


def generate_sam_masks_for_camera(
    images_dir: Path,
    masks_dir: Path,
    lucid_sam_dir: Path = LUCID_SAM_DIR,
    max_dimension: Optional[int] = None,
    force: bool = False
) -> bool:
    """Generate SAM masks for all images in a camera's images directory.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory to save masks (masks/image_stem/)
        lucid_sam_dir: Path to lucid-sam directory
        max_dimension: Maximum image dimension for SAM processing
        force: If True, regenerate masks even if they exist
        
    Returns:
        True if all successful, False if any errors
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    image_files.sort()
    
    if not image_files:
        print(f"    No images found in {images_dir}")
        return True
    
    print(f"    Generating SAM masks for {len(image_files)} images...")
    
    all_success = True
    for image_file in image_files:
        image_stem = image_file.stem
        mask_subdir = masks_dir / image_stem
        
        # Check if masks already exist
        if not force and masks_exist_for_image(masks_dir, image_stem):
            print(f"    Masks already exist for {image_file.name}, skipping")
            continue
        
        if not generate_sam_masks_for_image(
            image_file, masks_dir, lucid_sam_dir, max_dimension
        ):
            all_success = False
    
    return all_success


# =============================================================================
# Skip Logic Functions
# =============================================================================

def masks_exist_for_image(masks_dir: Path, image_name: str) -> bool:
    """Check if masks already exist for an image.
    
    Args:
        masks_dir: Base masks directory
        image_name: Image name (stem, without extension)
        
    Returns:
        True if masks exist, False otherwise
    """
    mask_subdir = masks_dir / image_name
    if not mask_subdir.exists():
        return False
    # Check if there are any PNG files in the subdirectory
    png_files = list(mask_subdir.glob("*.png"))
    return len(png_files) > 0


def processed_masks_exist(processed_masks_dir: Path) -> bool:
    """Check if processed masks directory has content.
    
    Args:
        processed_masks_dir: Path to processed_masks directory
        
    Returns:
        True if processed masks exist, False otherwise
    """
    if not processed_masks_dir.exists():
        return False
    # Check if there are any subdirectories with PNG files
    for subdir in processed_masks_dir.iterdir():
        if subdir.is_dir():
            if any(subdir.glob("*.png")):
                return True
    return False


def all_masks_exist_for_images(images_dir: Path, masks_dir: Path) -> bool:
    """Check if all images have corresponding masks.
    
    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing mask subdirectories
        
    Returns:
        True if all images have masks, False otherwise
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'}
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix in image_extensions]
    
    if not image_files:
        return True  # No images means nothing to check
    
    for image_file in image_files:
        if not masks_exist_for_image(masks_dir, image_file.stem):
            return False
    return True


# =============================================================================
# Output Structure Functions
# =============================================================================

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
        # Look for processed_mask.py in the scripts folder
        script_dir = Path(__file__).parent
        processed_mask_script = script_dir / "processed_mask.py"
    
    if not processed_mask_script.exists():
        print(f"Warning: processed_mask.py not found at {processed_mask_script}, skipping mask processing")
        return False
    
    # Remove processed_masks_dir if it exists (processed_mask.py requires it to not exist)
    if processed_masks_dir.exists():
        print(f"  Removing existing processed_masks directory: {processed_masks_dir}")
        shutil.rmtree(processed_masks_dir)
    
    # Count mask subdirectories to show progress info
    mask_subdirs = [d for d in masks_dir.iterdir() if d.is_dir()]
    total_masks = sum(len(list(d.glob("*.png"))) for d in mask_subdirs)
    print(f"  Processing {len(mask_subdirs)} mask directories ({total_masks} total mask files)...")
    
    # Build command
    cmd = [
        'python3', str(processed_mask_script),
        '-i', str(masks_dir),
        '-o', str(processed_masks_dir)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # Print output from processed_mask.py
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
        print(f"  Processed masks saved to: {processed_masks_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Error processing masks: {e}")
        if e.stdout and e.stdout.strip():
            print(f"  stdout: {e.stdout}")
        if e.stderr and e.stderr.strip():
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
        
        # Show what we're processing
        print(f"    Processing {len(file_names)} file(s): {', '.join(file_names)}")
        print(f"    Calib file: {calib_json_path}")
        print(f"    Running calibration (output streamed below)...")
        
        # Run calibration with real-time output streaming
        cmd = [str(executable_path), str(calib_json_path)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(f"    [Calib] {line.rstrip()}")
        
        process.wait()
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        
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
        print(f"  Error running calibration (exit code: {e.returncode})")
        print(f"  Output was streamed above - check for error messages.")
        return False, output_subdir
    finally:
        os.chdir(original_cwd)


def process_scene(scene_dir: Path, output_dir: Path, 
                  convert_script: Path, processed_mask_script: Path,
                  executable_path: Path, skip_calibration: bool = False,
                  force_sam: bool = False, force_process_masks: bool = False,
                  sam_max_dimension: Optional[int] = None,
                  lucid_sam_dir: Path = LUCID_SAM_DIR):
    """Process a single scene: convert structure, generate masks, process masks, run calibration.
    
    Args:
        scene_dir: Path to scene directory
        output_dir: Path to output directory
        convert_script: Path to convert_lucid_calib.py
        processed_mask_script: Path to processed_mask.py
        executable_path: Path to calibration executable
        skip_calibration: If True, skip the calibration step
        force_sam: If True, regenerate SAM masks even if they exist
        force_process_masks: If True, reprocess masks even if processed_masks exist
        sam_max_dimension: Maximum image dimension for SAM processing
        lucid_sam_dir: Path to lucid-sam directory
    """
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
        
        # =====================================================================
        # SAM Mask Generation
        # =====================================================================
        # Check if we need to generate SAM masks
        need_sam = force_sam or not all_masks_exist_for_images(
            dirs['images_dir'], dirs['masks_dir']
        )
        
        if need_sam:
            print(f"  Generating SAM masks...")
            generate_sam_masks_for_camera(
                dirs['images_dir'],
                dirs['masks_dir'],
                lucid_sam_dir=lucid_sam_dir,
                max_dimension=sam_max_dimension,
                force=force_sam
            )
        else:
            print(f"  SAM masks already exist, skipping generation (use --force-sam to regenerate)")
        
        # =====================================================================
        # Process Masks
        # =====================================================================
        # Check if we need to process masks
        need_process = force_process_masks or not processed_masks_exist(dirs['processed_masks_dir'])
        
        if need_process:
            # Process masks if they exist
            process_masks(dirs['masks_dir'], dirs['processed_masks_dir'], processed_mask_script)
        else:
            print(f"  Processed masks already exist, skipping (use --force-process-masks to reprocess)")
        
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
    # Enforce virtual environment
    enforce_virtual_environment()
    
    parser = argparse.ArgumentParser(
        description='Process Lucid data pipeline: generate SAM masks, process masks, run calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (first run clones lucid-sam, generates masks, processes them, runs calibration)
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output
  
  # Re-run calibration only (auto-skips mask generation since masks exist)
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output
  
  # Force regenerate all SAM masks
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output --force-sam
  
  # Force reprocess masks
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output --force-process-masks
  
  # Skip calibration step (only generate and process masks)
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output --skip-calibration
  
  # Limit image size for faster SAM processing (useful for large images)
  python scripts/process_lucid_pipeline.py -i datasampleformat -o output --sam-max-dimension 1024
  
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
                        help='Path to processed_mask.py script (default: scripts/processed_mask.py)')
    parser.add_argument('--executable', type=Path, default=Path('./bin/run_lidar2camera'),
                        help='Path to calibration executable (default: ./bin/run_lidar2camera)')
    parser.add_argument('--skip-calibration', action='store_true',
                        help='Skip the calibration step (only convert format and process masks)')
    
    # SAM-related arguments
    parser.add_argument('--force-sam', action='store_true',
                        help='Force regenerate SAM masks even if they already exist')
    parser.add_argument('--force-process-masks', action='store_true',
                        help='Force reprocess masks even if processed_masks already exist')
    parser.add_argument('--sam-max-dimension', type=int, default=None,
                        help='Maximum image dimension for SAM processing (reduces memory usage)')
    parser.add_argument('--lucid-sam-dir', type=Path, default=LUCID_SAM_DIR,
                        help=f'Path to lucid-sam directory (default: {LUCID_SAM_DIR})')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input.exists():
        print(f"Error: Input directory does not exist: {args.input}")
        return 1
    
    if not args.skip_calibration and not args.executable.exists():
        print(f"Error: Calibration executable does not exist: {args.executable}")
        print(f"  Please build the project first or specify the correct path with --executable")
        print(f"  Or use --skip-calibration to skip the calibration step")
        return 1
    
    # Set default script paths
    if args.convert_script is None:
        args.convert_script = Path(__file__).parent / "convert_lucid_calib.py"
    if args.processed_mask_script is None:
        args.processed_mask_script = Path(__file__).parent / "processed_mask.py"
    
    # Validate scripts exist
    if not args.convert_script.exists():
        print(f"Error: convert_lucid_calib.py not found at {args.convert_script}")
        return 1
    
    # =========================================================================
    # Setup lucid-sam
    # =========================================================================
    print("=" * 60)
    print("Setting up lucid-sam for mask generation")
    print("=" * 60)
    
    if not ensure_lucid_sam_ready(args.lucid_sam_dir):
        print("Error: Failed to set up lucid-sam. Please check the error messages above.")
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
            args.executable, args.skip_calibration,
            force_sam=args.force_sam,
            force_process_masks=args.force_process_masks,
            sam_max_dimension=args.sam_max_dimension,
            lucid_sam_dir=args.lucid_sam_dir
        )
    
    print(f"\n{'='*60}")
    print(f"Pipeline complete! Results saved to: {args.output}")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    exit(main())
