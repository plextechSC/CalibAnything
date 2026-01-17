#!/usr/bin/env python3
"""
Parameter sweep script for CalibAnything calibration (No CLI version).

This script:
1. Finds all calib.json files in the data folder
2. Creates/edits calib_sweep.json in each directory
3. Sweeps specified parameters with min, max, and step values
4. Runs the executable for each parameter combination
5. Saves output images to directories named after the parameters used

CONFIGURATION:
Edit the configuration section below to specify:
- Executable path
- Data directory
- Specific calib.json files (optional)
- Parameter sweep ranges (min, max, step)
"""

import json
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import itertools

# ============================================================================
# CONFIGURATION - Edit these values to customize your parameter sweep
# ============================================================================

# Path to the calibration executable
EXECUTABLE_PATH = "./bin/run_lidar2camera"

# Path to data directory (will search for all calib.json files recursively)
DATA_DIR = "./data"

# Specific calib.json files to process (leave as None to process all found files)
# Example: CALIB_FILES = ["./data/cam03/calib.json", "./data/cam04/calib.json"]
CALIB_FILES = ["./data/cam03/calib.json"]

# Parameter sweep configuration
# Format: "parameter.path": {
#     "default": value,           # Default value (used when sweep=False)
#     "sweep": bool,              # Whether to sweep this parameter
#     "min": value,               # Minimum value for sweep (only used if sweep=True)
#     "max": value,               # Maximum value for sweep (only used if sweep=True)
#     "step": value               # Step size for sweep (only used if sweep=True)
# }
#
# Set sweep=True to sweep over the parameter range, or sweep=False to use the default value.
# All parameters are listed below - set sweep=True for parameters you want to sweep.
#
# ============================================================================
# PARAMETER DESCRIPTIONS AND BOUNDS:
# ============================================================================
#
# 1. params.min_plane_point_num (integer)
#    Description: Minimum number of points required to extract a plane from the point cloud.
#                 The algorithm iteratively extracts planes until no plane has enough points.
#                 Used in RANSAC-based plane segmentation.
#    Typical values: 100-5000 (depends on point cloud density)
#    Common range: 100-2000 for dense LiDARs (64-beam+), 2000-5000 for sparse LiDARs (32-beam)
#    Bounds: Must be > 0, typically 50-10000
#
# 2. params.cluster_tolerance (float, meters)
#    Description: Spatial distance threshold for Euclidean clustering of non-planar points.
#                 Points within this distance are grouped into the same cluster.
#                 Larger values create fewer, larger clusters (good for sparse point clouds).
#                 Smaller values create more, smaller clusters (good for dense point clouds).
#    Typical values: 0.05-0.5 meters
#    Common range: 0.1-0.3 for dense LiDARs, 0.2-0.5 for sparse LiDARs
#    Bounds: Must be > 0, typically 0.01-1.0 meters
#
# 3. params.search_num (integer)
#    Description: Number of random search iterations performed during calibration optimization.
#                 Each iteration tests a random transformation within the search range.
#                 Higher values = more thorough search but slower computation.
#    Typical values: 2000-10000
#    Common range: 4000-8000 (good balance of speed and accuracy)
#    Bounds: Must be > 0, typically 1000-50000
#
# 4. params.search_range.rot_deg (float, degrees)
#    Description: Maximum rotation search range in degrees for each axis (roll, pitch, yaw).
#                 The algorithm searches within [-rot_deg, +rot_deg] for each rotation axis.
#                 Used to refine the initial extrinsic guess.
#                 WARNING: Code enforces rot_deg <= 10 degrees (hard limit).
#    Typical values: 3-7 degrees
#    Common range: 3-10 degrees (must be <= 10)
#    Bounds: Must be > 0 and <= 10 degrees (hard limit in code)
#
# 5. params.search_range.trans_m (float, meters)
#    Description: Maximum translation search range in meters for each axis (x, y, z).
#                 The algorithm searches within [-trans_m, +trans_m] for each translation axis.
#                 Used to refine the initial extrinsic guess.
#                 WARNING: Code enforces trans_m <= 1.0 meters (hard limit).
#    Typical values: 0.3-0.7 meters
#    Common range: 0.1-1.0 meters (must be <= 1.0)
#    Bounds: Must be > 0 and <= 1.0 meters (hard limit in code)
#
# 6. params.point_range.top (float, normalized 0.0-1.0)
#    Description: Top boundary of the vertical image region used for calibration scoring.
#                 Expressed as fraction of image height (0.0 = top of image, 1.0 = bottom).
#                 Used to focus calibration on relevant image regions (e.g., ignore sky).
#                 Must be < point_range.bottom.
#    Typical values: 0.0-0.5 (0.0 = use full image from top)
#    Common range: 0.0-0.6
#    Bounds: 0.0-1.0, must be < point_range.bottom
#
# 7. params.point_range.bottom (float, normalized 0.0-1.0)
#    Description: Bottom boundary of the vertical image region used for calibration scoring.
#                 Expressed as fraction of image height (0.0 = top of image, 1.0 = bottom).
#                 Used to focus calibration on relevant image regions (e.g., ignore ground).
#                 Must be > point_range.top.
#    Typical values: 0.4-1.0 (1.0 = use full image to bottom)
#    Common range: 0.5-1.0
#    Bounds: 0.0-1.0, must be > point_range.top
#
# 8. params.down_sample.voxel_m (float, meters)
#    Description: Voxel size for point cloud downsampling (if downsampling is enabled).
#                 Points within each voxel are replaced by their centroid.
#                 Larger values = more aggressive downsampling = faster but less accurate.
#                 Only used if params.down_sample.is_valid = true.
#    Typical values: 0.02-0.1 meters
#    Common range: 0.05-0.1 for dense LiDARs, 0.1-0.2 for sparse LiDARs
#    Bounds: Must be > 0, typically 0.01-0.5 meters
#
# ============================================================================
PARAM_CONFIG = {
    "params.min_plane_point_num": {
        "default": 200,
        "sweep": True,
        "min": 1700,
        "max": 2300,
        "step": 100
    },
    "params.cluster_tolerance": {
        "default": 0.2,
        "sweep": False,
        "min": 0.1,
        "max": 0.3,
        "step": 0.2
    },
    "params.search_num": {
        "default": 8000,
        "sweep": True,
        "min": 8000,
        "max": 16000,
        "step": 8000
    },
    "params.search_range.rot_deg": {
        "default": 3,
        "sweep": False,
        "min": 3,
        "max": 7,
        "step": 1
    },
    "params.search_range.trans_m": {
        "default": 0.5,
        "sweep": False,
        "min": 0.3,
        "max": 0.7,
        "step": 0.1
    },
    "params.point_range.top": {
        "default": 0.5,
        "sweep": False,
        "min": 0.0,
        "max": 0.5,
        "step": 0.1
    },
    "params.point_range.bottom": {
        "default": 1.0,
        "sweep": False,
        "min": 0.5,
        "max": 1.0,
        "step": 0.1
    },
    "params.down_sample.voxel_m": {
        "default": 0.05,
        "sweep": False,
        "min": 0.01,
        "max": 1.0,
        "step": 0.1
    },
}

# ============================================================================
# END OF CONFIGURATION
# ============================================================================


def find_calib_files(data_dir: str) -> List[Path]:
    """Find all calib.json files in the data directory."""
    data_path = Path(data_dir)
    calib_files = list(data_path.rglob("calib.json"))
    return calib_files


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(filepath: Path, data: Dict[str, Any]):
    """Save JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def get_nested_value(data: Dict, path: str):
    """Get value from nested dictionary using dot notation (e.g., 'params.search_range.rot_deg')."""
    keys = path.split('.')
    value = data
    for key in keys:
        value = value[key]
    return value


def set_nested_value(data: Dict, path: str, value: Any):
    """Set value in nested dictionary using dot notation."""
    keys = path.split('.')
    for key in keys[:-1]:
        if key not in data:
            data[key] = {}
        data = data[key]
    data[keys[-1]] = value


def generate_parameter_combinations(param_config: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from configuration.
    
    Parameters with sweep=False use their default value.
    Parameters with sweep=True generate a range from min to max with step.
    """
    # Separate parameters into those to sweep and those to use defaults
    sweep_params = {}
    default_params = {}
    
    for param_name, config in param_config.items():
        if config.get('sweep', False):
            sweep_params[param_name] = config
        else:
            default_params[param_name] = config['default']
    
    # If no parameters to sweep, return single combination with all defaults
    if not sweep_params:
        return [default_params]
    
    # Generate ranges for parameters to sweep
    sweep_param_names = list(sweep_params.keys())
    param_ranges = []
    
    for param_name in sweep_param_names:
        config = sweep_params[param_name]
        min_val = config['min']
        max_val = config['max']
        step = config['step']
        
        # Generate range of values
        values = []
        current = min_val
        while current <= max_val:
            values.append(current)
            current += step
        
        param_ranges.append(values)
    
    # Generate all combinations for swept parameters
    combinations = []
    for combo in itertools.product(*param_ranges):
        # Start with default values for all parameters
        param_combo = default_params.copy()
        # Override with swept values
        for param_name, value in zip(sweep_param_names, combo):
            param_combo[param_name] = value
        combinations.append(param_combo)
    
    return combinations


def create_param_dir_name(param_combo: Dict[str, Any], param_config: Dict[str, Dict[str, Any]]) -> str:
    """
    Create a directory name from parameter combination.
    Only includes parameters that were swept (sweep=True).
    """
    parts = []
    for param_name, value in sorted(param_combo.items()):
        # Only include parameters that were swept
        if param_name in param_config and param_config[param_name].get('sweep', False):
            # Replace dots with underscores for directory name
            safe_name = param_name.replace('.', '_')
            # Format value appropriately
            if isinstance(value, float):
                if value.is_integer():
                    parts.append(f"{safe_name}_{int(value)}")
                else:
                    parts.append(f"{safe_name}_{value:.3f}".rstrip('0').rstrip('.'))
            elif isinstance(value, int):
                parts.append(f"{safe_name}_{value}")
            else:
                parts.append(f"{safe_name}_{value}")
    
    # If no swept parameters, use a default name
    if not parts:
        return "default_params"
    
    return "_".join(parts)


def run_executable(executable_path: str, calib_json_path: Path) -> bool:
    """Run the calibration executable and wait for completion."""
    try:
        result = subprocess.run(
            [executable_path, str(calib_json_path)],
            check=True,
            capture_output=True,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running executable: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def copy_output_images(output_dir: Path, possible_locations: List[Path]):
    """Copy refined_proj.png and refined_proj_seg.png to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_name in ['refined_proj.png', 'refined_proj_seg.png']:
        copied = False
        for location in possible_locations:
            src = location / image_name
            if src.exists():
                dst = output_dir / image_name
                shutil.copy2(src, dst)
                print(f"  Copied {image_name} from {src} to {output_dir}")
                copied = True
                break
        
        if not copied:
            print(f"  Warning: {image_name} not found in any expected location")


def sweep_parameters(
    data_dir: str,
    executable_path: str,
    param_config: Dict[str, Dict[str, Any]],
    calib_files: List[Path] = None
):
    """
    Main function to sweep parameters.
    
    Args:
        data_dir: Path to data directory
        executable_path: Path to the calibration executable
        param_config: Dictionary mapping parameter paths to {min, max, step}
        calib_files: Optional list of specific calib.json files to process
    """
    # Determine project root (where executable is located)
    # Resolve executable path relative to current directory
    if Path(executable_path).is_absolute():
        executable_path_obj = Path(executable_path)
    else:
        executable_path_obj = Path(executable_path).resolve()
    
    # If executable is in bin/, project root is parent of bin/
    if executable_path_obj.parent.name == 'bin':
        project_root = executable_path_obj.parent.parent
    else:
        # Otherwise, assume executable is in project root
        project_root = executable_path_obj.parent
    
    # Make executable path relative to project root for running
    try:
        exec_rel_to_root = executable_path_obj.relative_to(project_root)
    except ValueError:
        # If executable is not under project_root, use absolute path
        exec_rel_to_root = executable_path_obj
    current_dir = Path.cwd()
    
    # Find calib.json files
    if calib_files is None:
        calib_files = find_calib_files(data_dir)
    
    if not calib_files:
        print(f"No calib.json files found in {data_dir}")
        return
    
    print(f"Found {len(calib_files)} calib.json file(s)")
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations(param_config)
    print(f"Generated {len(param_combinations)} parameter combinations")
    
    # Process each calib.json file
    for calib_file in calib_files:
        print(f"\n{'='*60}")
        print(f"Processing: {calib_file}")
        print(f"{'='*60}")
        
        calib_dir = calib_file.parent
        calib_sweep_file = calib_dir / "calib_sweep.json"
        
        # Load original calib.json
        calib_data = load_json(calib_file)
        
        # Process each parameter combination
        for i, param_combo in enumerate(param_combinations, 1):
            print(f"\n[{i}/{len(param_combinations)}] Parameter combination:")
            for param_name, value in param_combo.items():
                print(f"  {param_name} = {value}")
            
            # Create a copy of calib_data for this combination
            sweep_data = json.loads(json.dumps(calib_data))  # Deep copy
            
            # Update parameters
            for param_name, value in param_combo.items():
                set_nested_value(sweep_data, param_name, value)
            
            # Save calib_sweep.json
            save_json(calib_sweep_file, sweep_data)
            print(f"  Created/updated {calib_sweep_file}")
            
            # Run executable (from project root to ensure outputs go to expected location)
            print(f"  Running executable: {executable_path}")
            original_cwd = os.getcwd()
            try:
                os.chdir(project_root)
                success = run_executable(str(exec_rel_to_root), calib_sweep_file)
            finally:
                os.chdir(original_cwd)
            
            if not success:
                print(f"  Failed to run executable, skipping this combination")
                continue
            
            # Create output directory name
            param_dir_name = create_param_dir_name(param_combo, param_config)
            output_dir = calib_dir / "sweep_results" / param_dir_name
            
            # Copy output images (check multiple possible locations)
            possible_locations = [project_root, current_dir, calib_dir]
            copy_output_images(output_dir, possible_locations)
            print(f"  Results saved to: {output_dir}")


def main():
    """Main entry point - uses configuration from top of file."""
    # Validate configuration
    if not PARAM_CONFIG:
        print("Error: PARAM_CONFIG is empty. Please specify at least one parameter to sweep.")
        sys.exit(1)
    
    # Validate parameter configuration
    for param_name, config in PARAM_CONFIG.items():
        # Check required keys
        if 'default' not in config:
            print(f"Error: Parameter '{param_name}' must have a 'default' value.")
            sys.exit(1)
        
        if 'sweep' not in config:
            print(f"Error: Parameter '{param_name}' must have a 'sweep' boolean flag.")
            sys.exit(1)
        
        # If sweeping, validate min, max, step
        if config.get('sweep', False):
            if 'min' not in config or 'max' not in config or 'step' not in config:
                print(f"Error: Parameter '{param_name}' has sweep=True but is missing 'min', 'max', or 'step'.")
                sys.exit(1)
            
            min_val = config['min']
            max_val = config['max']
            step_val = config['step']
            
            if min_val > max_val:
                print(f"Error: min ({min_val}) must be <= max ({max_val}) for parameter {param_name}")
                sys.exit(1)
            
            if step_val <= 0:
                print(f"Error: step ({step_val}) must be > 0 for parameter {param_name}")
                sys.exit(1)
    
    # Check executable exists
    if not os.path.exists(EXECUTABLE_PATH):
        print(f"Error: Executable not found: {EXECUTABLE_PATH}")
        sys.exit(1)
    
    # Get calib files
    calib_files = None
    if CALIB_FILES is not None:
        calib_files = [Path(f) for f in CALIB_FILES]
        for f in calib_files:
            if not f.exists():
                print(f"Error: Calib file not found: {f}")
                sys.exit(1)
    
    # Print configuration summary
    print("="*60)
    print("Parameter Sweep Configuration")
    print("="*60)
    print(f"Executable: {EXECUTABLE_PATH}")
    print(f"Data directory: {DATA_DIR}")
    if calib_files:
        print(f"Specific calib files: {len(calib_files)} file(s)")
    else:
        print("Calib files: All found in data directory")
    print(f"\nParameters configuration:")
    swept_params = []
    default_params = []
    for param_name, config in PARAM_CONFIG.items():
        if config.get('sweep', False):
            swept_params.append(param_name)
            print(f"  {param_name}: SWEEP - {config['min']} to {config['max']} (step: {config['step']})")
        else:
            default_params.append(param_name)
            print(f"  {param_name}: DEFAULT - {config['default']}")
    
    if not swept_params:
        print("\n  WARNING: No parameters are set to sweep (all have sweep=False)")
        print("  The script will run once with default values.")
    
    print("="*60)
    print()
    
    # Run sweep
    sweep_parameters(
        data_dir=DATA_DIR,
        executable_path=EXECUTABLE_PATH,
        param_config=PARAM_CONFIG,
        calib_files=calib_files
    )


if __name__ == "__main__":
    main()

