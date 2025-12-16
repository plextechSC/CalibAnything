#!/usr/bin/env python3
"""
Parameter sweep script for CalibAnything calibration.

This script:
1. Finds all calib.json files in the data folder
2. Creates/edits calib_sweep.json in each directory
3. Sweeps specified parameters with min, max, and step values
4. Runs the executable for each parameter combination
5. Saves output images to directories named after the parameters used
"""

import json
import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import itertools
import argparse


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


def generate_parameter_combinations(param_config: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    """Generate all parameter combinations from min, max, step configuration."""
    param_names = list(param_config.keys())
    param_ranges = []
    
    for param_name in param_names:
        config = param_config[param_name]
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
    
    # Generate all combinations
    combinations = []
    for combo in itertools.product(*param_ranges):
        combinations.append(dict(zip(param_names, combo)))
    
    return combinations


def create_param_dir_name(param_combo: Dict[str, float]) -> str:
    """Create a directory name from parameter combination."""
    parts = []
    for param_name, value in sorted(param_combo.items()):
        # Replace dots with underscores for directory name
        safe_name = param_name.replace('.', '_')
        # Format value appropriately
        if isinstance(value, float):
            if value.is_integer():
                parts.append(f"{safe_name}_{int(value)}")
            else:
                parts.append(f"{safe_name}_{value:.3f}".rstrip('0').rstrip('.'))
        else:
            parts.append(f"{safe_name}_{value}")
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
    param_config: Dict[str, Dict[str, float]],
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
            param_dir_name = create_param_dir_name(param_combo)
            output_dir = calib_dir / "sweep_results" / param_dir_name
            
            # Copy output images (check multiple possible locations)
            possible_locations = [project_root, current_dir, calib_dir]
            copy_output_images(output_dir, possible_locations)
            print(f"  Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Parameter sweep script for CalibAnything calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Sweep cluster_tolerance from 0.1 to 0.3 with step 0.05
  python parameter_sweep.py --executable ./bin/run_lidar2camera \\
    --param params.cluster_tolerance --min 0.1 --max 0.3 --step 0.05
  
  # Sweep multiple parameters
  python parameter_sweep.py --executable ./bin/run_lidar2camera \\
    --param params.cluster_tolerance --min 0.1 --max 0.3 --step 0.05 \\
    --param params.search_range.rot_deg --min 3 --max 7 --step 1 \\
    --param params.search_range.trans_m --min 0.3 --max 0.7 --step 0.1
  
  # Process specific calib.json files
  python parameter_sweep.py --executable ./bin/run_lidar2camera \\
    --data-dir ./data \\
    --calib-file ./data/cam03/calib.json \\
    --param params.cluster_tolerance --min 0.1 --max 0.3 --step 0.05
        """
    )
    
    parser.add_argument(
        '--executable',
        type=str,
        required=True,
        help='Path to the calibration executable (e.g., ./bin/run_lidar2camera)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Path to data directory (default: ./data)'
    )
    
    parser.add_argument(
        '--calib-file',
        type=str,
        action='append',
        help='Specific calib.json file(s) to process (can be specified multiple times)'
    )
    
    parser.add_argument(
        '--param',
        type=str,
        action='append',
        required=True,
        help='Parameter path to sweep (e.g., params.cluster_tolerance). Can be specified multiple times.'
    )
    
    parser.add_argument(
        '--min',
        type=float,
        action='append',
        required=True,
        help='Minimum value for parameter (must match number of --param arguments)'
    )
    
    parser.add_argument(
        '--max',
        type=float,
        action='append',
        required=True,
        help='Maximum value for parameter (must match number of --param arguments)'
    )
    
    parser.add_argument(
        '--step',
        type=float,
        action='append',
        required=True,
        help='Step size for parameter (must match number of --param arguments)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (len(args.param) == len(args.min) == len(args.max) == len(args.step)):
        parser.error("Number of --param, --min, --max, and --step arguments must match")
    
    # Build parameter configuration
    param_config = {}
    for param, min_val, max_val, step_val in zip(args.param, args.min, args.max, args.step):
        if min_val > max_val:
            parser.error(f"min ({min_val}) must be <= max ({max_val}) for parameter {param}")
        if step_val <= 0:
            parser.error(f"step ({step_val}) must be > 0 for parameter {param}")
        param_config[param] = {
            'min': min_val,
            'max': max_val,
            'step': step_val
        }
    
    # Check executable exists
    if not os.path.exists(args.executable):
        parser.error(f"Executable not found: {args.executable}")
    
    # Get calib files
    calib_files = None
    if args.calib_file:
        calib_files = [Path(f) for f in args.calib_file]
        for f in calib_files:
            if not f.exists():
                parser.error(f"Calib file not found: {f}")
    
    # Run sweep
    sweep_parameters(
        data_dir=args.data_dir,
        executable_path=args.executable,
        param_config=param_config,
        calib_files=calib_files
    )


if __name__ == "__main__":
    main()

