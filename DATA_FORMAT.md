# Data Format Specification

This document describes the expected directory structure and file formats for LiDAR-camera calibration data used by CalibAnything.

## Directory Structure

### Multi-Camera Scene Format

For scenes with multiple cameras, organize your data as follows:

```
YOUR_DATA_FOLDER/
├── scene1/
│   ├── 000000.pcd              # Point cloud file(s)
│   ├── 000001.pcd              # (optional additional frames)
│   ├── ...
│   ├── cam02/
│   │   ├── lucid_calib.json    # Camera calibration parameters
│   │   ├── sample_image.png    # RGB image(s)
│   │   └── ...
│   ├── cam03/
│   │   ├── lucid_calib.json
│   │   ├── sample_image.png
│   │   └── ...
│   └── cam04/
│       ├── lucid_calib.json
│       ├── sample_image.png
│       └── ...
├── scene2/
│   └── ...
└── ...
```

### Time-Series Format

For time-synchronized sequences with a single camera:

```
YOUR_DATA_FOLDER/
├── daytimetraffic/
│   ├── 000000.pcd              # Point cloud frame 0
│   ├── 000001.pcd              # Point cloud frame 1
│   ├── ...
│   └── cam03/
│       ├── lucid_calib.json    # Camera calibration parameters
│       ├── 000000.png          # Image frame 0
│       ├── 000001.png          # Image frame 1
│       └── ...
```

## File Formats

### Point Cloud Files (`.pcd`)

Standard PCD (Point Cloud Data) format with intensity values. The calibration algorithm requires intensity information for plane extraction and matching.

### Image Files (`.png`)

RGB images in PNG format. File naming should match the corresponding point cloud frames for time-synchronized data (e.g., `000000.pcd` corresponds to `000000.png`).

### Calibration File (`lucid_calib.json`)

JSON file containing camera intrinsic and extrinsic parameters:

```json
{
  "intrinsic_params": {
    "fx": 4567.32,          // Focal length X (pixels)
    "fy": 4566.75,          // Focal length Y (pixels)
    "cx": 1915.45,          // Principal point X (pixels)
    "cy": 1103.16,          // Principal point Y (pixels)
    "k1": -0.056582,        // Radial distortion coefficient 1
    "k2": 0.091158,         // Radial distortion coefficient 2
    "k3": 0.103344,         // Radial distortion coefficient 3
    "k4": 0.261091,         // Radial distortion coefficient 4
    "k5": 0,                // Radial distortion coefficient 5
    "k6": 0,                // Radial distortion coefficient 6
    "p1": -0.000124,        // Tangential distortion coefficient 1
    "p2": 0.000981,         // Tangential distortion coefficient 2
    "camera": "fnc_c",      // Camera identifier
    "camera_model": "normal" // Camera model type
  },
  "extrinsic_params": {
    "roll": -89.22,         // Roll angle (degrees)
    "pitch": 0.10,          // Pitch angle (degrees)
    "yaw": -89.71,          // Yaw angle (degrees)
    "px": 2.46,             // Translation X (meters)
    "py": -0.01,            // Translation Y (meters)
    "pz": -0.53,            // Translation Z (meters)
    "quaternion": {         // Rotation as quaternion (optional)
      "x": -0.497,
      "y": 0.500,
      "z": -0.502,
      "w": 0.501
    },
    "camera_coordinate": "OPTICAL",  // Coordinate frame convention
    "translation_error": 0.043,      // (optional) Translation error metric
    "rotation_error": 2.52,          // (optional) Rotation error metric
    "reprojection_error": 34.44      // (optional) Reprojection error
  }
}
```

#### Intrinsic Parameters

| Parameter | Description |
|-----------|-------------|
| `fx`, `fy` | Focal lengths in pixels |
| `cx`, `cy` | Principal point coordinates in pixels |
| `k1`-`k6` | Radial distortion coefficients (OpenCV convention) |
| `p1`, `p2` | Tangential distortion coefficients |
| `camera` | Camera identifier string |
| `camera_model` | Camera model type (`"normal"` for standard pinhole) |

#### Extrinsic Parameters

| Parameter | Description |
|-----------|-------------|
| `roll`, `pitch`, `yaw` | Euler angles in degrees |
| `px`, `py`, `pz` | Translation vector in meters |
| `quaternion` | Alternative rotation representation (x, y, z, w) |
| `camera_coordinate` | Coordinate frame convention (`"OPTICAL"` for standard camera frame) |

## Naming Conventions

- **Scene directories**: Use descriptive names (e.g., `scene1`, `daytimetraffic`, `parking_lot`)
- **Camera directories**: Use consistent naming (e.g., `cam01`, `cam02`, `cam03`)
- **Frame numbering**: Use zero-padded 6-digit numbers (e.g., `000000`, `000001`)
- **File extensions**: `.pcd` for point clouds, `.png` for images, `.json` for calibration

## Converting to Main Calibration Format

To use this data with the main calibration pipeline, you may need to convert the `lucid_calib.json` format to the `calib.json` format expected by `run_lidar2camera`. See the main [README.md](README.md#edit-the-json-file) for the `calib.json` specification, and use `scripts/convert_lucid_calib.py` for automated conversion.
