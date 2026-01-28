#!/usr/bin/env python3
"""
Script to crop distortion borders from camera images.

Camera 02 (PRIMAX-IMX728): 7680x2856 - larger black borders
Camera 05 & 07 (SKE-IMX623): 3840x2472 - similar smaller borders

Output is saved to a 'cropped' folder mirroring the original structure.
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


# Crop configurations for each camera type
# For fixed resolution cameras: (left, top, right, bottom) pixel coordinates
# For variable resolution cameras: percentage-based with "type": "percent"
CROP_CONFIGS = {
    # Camera 02 (PRIMAX-IMX728): Variable resolutions (6400x2382, 7680x2856, etc.)
    # Has larger black borders, especially at bottom - uses percentage-based cropping
    "cam02": {
        "type": "percent",
        "left": 7.5,      # ~7.5% from left
        "top": 16.0,      # ~16% from top  
        "right": 93.0,    # ~93% from left (7% from right)
        "bottom": 75.0,   # ~83.5% from top (crop more of bottom distortion)
    },
    # Camera 05 (SKE-IMX623): 3840x2472
    # Smaller, roughly equal borders
    "cam05": {
        "type": "fixed",
        "left": 175,
        "top": 200,
        "right": 3560,
        "bottom": 2260,
    },
    # Camera 07 (SKE-IMX623): 3840x2472
    # Same as Camera 05
    "cam07": {
        "type": "fixed",
        "left": 175,
        "top": 200,
        "right": 3560,
        "bottom": 2260,
    },
}

# Cameras to process
TARGET_CAMERAS = ["cam02", "cam05", "cam07"]


def get_crop_box(camera_name: str, img_width: int = None, img_height: int = None) -> tuple:
    """
    Get crop box for a camera, handling both fixed and percentage-based configs.
    
    Args:
        camera_name: Name of the camera (e.g., "cam02")
        img_width: Image width (required for percentage-based configs)
        img_height: Image height (required for percentage-based configs)
        
    Returns:
        Tuple of (left, top, right, bottom) pixel coordinates
    """
    config = CROP_CONFIGS.get(camera_name)
    if not config:
        raise ValueError(f"Unknown camera: {camera_name}")
    
    if config.get("type") == "percent":
        if img_width is None or img_height is None:
            raise ValueError(f"Image dimensions required for percentage-based crop config: {camera_name}")
        left = int(img_width * config["left"] / 100)
        top = int(img_height * config["top"] / 100)
        right = int(img_width * config["right"] / 100)
        bottom = int(img_height * config["bottom"] / 100)
        return (left, top, right, bottom)
    else:
        # Fixed pixel coordinates
        return (config["left"], config["top"], config["right"], config["bottom"])


def crop_image(input_path: Path, output_path: Path, camera_name: str) -> tuple:
    """
    Crop a single image and save to output path.
    
    Args:
        input_path: Path to input image
        output_path: Path to save cropped image
        camera_name: Camera name for crop configuration
        
    Returns:
        Tuple of (input_path, success, message)
    """
    try:
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open, crop, and save
        with Image.open(input_path) as img:
            crop_box = get_crop_box(camera_name, img.width, img.height)
            cropped = img.crop(crop_box)
            cropped.save(output_path, quality=95)
        
        return (input_path, True, f"Cropped to {cropped.size}")
    except Exception as e:
        return (input_path, False, str(e))


def process_scenario(data_dir: Path, output_dir: Path, scenario_id: str, 
                     cameras: list = None, max_workers: int = 4) -> dict:
    """
    Process all target cameras in a scenario.
    
    Args:
        data_dir: Base data directory
        output_dir: Base output directory
        scenario_id: Scenario folder name
        cameras: List of cameras to process (default: TARGET_CAMERAS)
        max_workers: Number of parallel workers
        
    Returns:
        Dict with processing statistics
    """
    cameras = cameras or TARGET_CAMERAS
    scenario_path = data_dir / scenario_id
    
    if not scenario_path.exists():
        print(f"Scenario not found: {scenario_path}")
        return {"processed": 0, "failed": 0, "skipped": 0}
    
    stats = {"processed": 0, "failed": 0, "skipped": 0}
    tasks = []
    
    # Collect all images to process
    for camera in cameras:
        camera_path = scenario_path / camera
        if not camera_path.exists():
            print(f"  Camera folder not found: {camera_path}")
            continue
        
        # Find all PNG images
        for img_file in camera_path.glob("*.png"):
            output_path = output_dir / scenario_id / camera / img_file.name
            tasks.append((img_file, output_path, camera))
    
    if not tasks:
        print(f"  No images found for scenario {scenario_id}")
        return stats
    
    print(f"  Processing {len(tasks)} images...")
    
    # Process images in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(crop_image, inp, out, cam): inp 
            for inp, out, cam in tasks
        }
        
        for future in as_completed(futures):
            input_path, success, message = future.result()
            if success:
                stats["processed"] += 1
            else:
                stats["failed"] += 1
                print(f"    Failed: {input_path.name} - {message}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Crop distortion borders from camera images"
    )
    parser.add_argument(
        "--data-dir", 
        type=Path, 
        default=Path("data"),
        help="Input data directory (default: data)"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("cropped"),
        help="Output directory (default: cropped)"
    )
    parser.add_argument(
        "--scenarios", 
        nargs="+",
        help="Specific scenarios to process (default: all)"
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        choices=TARGET_CAMERAS,
        default=TARGET_CAMERAS,
        help=f"Cameras to process (default: {TARGET_CAMERAS})"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        return 1
    
    # Get scenarios to process
    if args.scenarios:
        scenarios = args.scenarios
    else:
        # Find all scenario directories
        scenarios = [
            d.name for d in args.data_dir.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ]
    
    if not scenarios:
        print("No scenarios found to process")
        return 1
    
    print(f"Crop configurations:")
    for cam in args.cameras:
        config = CROP_CONFIGS[cam]
        if config.get("type") == "percent":
            w_pct = config["right"] - config["left"]
            h_pct = config["bottom"] - config["top"]
            print(f"  {cam}: {w_pct:.1f}% x {h_pct:.1f}% (percentage-based, variable resolution)")
        else:
            w = config["right"] - config["left"]
            h = config["bottom"] - config["top"]
            print(f"  {cam}: crop to {w}x{h} pixels (fixed)")
    
    print(f"\nProcessing {len(scenarios)} scenario(s): {', '.join(scenarios)}")
    print(f"Cameras: {', '.join(args.cameras)}")
    print(f"Output directory: {args.output_dir}")
    
    if args.dry_run:
        print("\n[DRY RUN - no files will be processed]")
        for scenario in scenarios:
            scenario_path = args.data_dir / scenario
            for camera in args.cameras:
                camera_path = scenario_path / camera
                if camera_path.exists():
                    count = len(list(camera_path.glob("*.png")))
                    print(f"  {scenario}/{camera}: {count} images")
        return 0
    
    print()
    
    # Process each scenario
    total_stats = {"processed": 0, "failed": 0, "skipped": 0}
    
    for scenario in sorted(scenarios):
        print(f"Scenario: {scenario}")
        stats = process_scenario(
            args.data_dir, 
            args.output_dir, 
            scenario,
            args.cameras,
            args.workers
        )
        for key in total_stats:
            total_stats[key] += stats[key]
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Processed: {total_stats['processed']}")
    print(f"  Failed: {total_stats['failed']}")
    print(f"  Output: {args.output_dir.absolute()}")
    
    return 0 if total_stats["failed"] == 0 else 1


if __name__ == "__main__":
    exit(main())
