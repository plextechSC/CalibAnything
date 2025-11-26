#!/usr/bin/env python3
"""
Convert a Lucid dataset calibration JSON into CalibAnything fwc_c.json format.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to the Lucid JSON file (e.g., fwc_c (1).json).",
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Destination path for the converted JSON (e.g., fwc_c.json).",
    )
    parser.add_argument("--img-folder", default="images")
    parser.add_argument("--mask-folder", default="processed_masks")
    parser.add_argument("--pc-folder", default="pc")
    parser.add_argument("--img-format", default=".png")
    parser.add_argument("--pc-format", default=".pcd")
    return parser.parse_args()


def build_cam_K(intrinsic: Dict[str, Any]) -> Dict[str, Any]:
    fx = intrinsic.get("fx", 0.0)
    fy = intrinsic.get("fy", 0.0)
    cx = intrinsic.get("cx", 0.0)
    cy = intrinsic.get("cy", 0.0)
    return {
        "rows": 3,
        "cols": 3,
        "data": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
    }


def build_cam_dist(intrinsic: Dict[str, Any]) -> Dict[str, Any]:
    coef_order: Sequence[str] = (
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "k4",
        "k5",
        "k6",
    )
    data: List[float] = []
    for key in coef_order:
        if key in intrinsic:
            data.append(intrinsic[key])
    return {
        "cols": len(data),
        "data": data,
    }


def quaternion_to_rotation_matrix(quat: Dict[str, float]) -> List[List[float]]:
    x = quat.get("x", 0.0)
    y = quat.get("y", 0.0)
    z = quat.get("z", 0.0)
    w = quat.get("w", 1.0)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ]


def build_extrinsic(extrinsic: Dict[str, Any]) -> Dict[str, Any]:
    quat = extrinsic.get("quaternion", {})
    rotation = quaternion_to_rotation_matrix(quat)
    px = extrinsic.get("px", 0.0)
    py = extrinsic.get("py", 0.0)
    pz = extrinsic.get("pz", 0.0)

    data = [
        [*rotation[0], px],
        [*rotation[1], py],
        [*rotation[2], pz],
        [0.0, 0.0, 0.0, 1.0],
    ]

    return {
        "rows": 4,
        "cols": 4,
        "data": data,
    }


def build_template(
    cam_K: Dict[str, Any],
    cam_dist: Dict[str, Any],
    extrinsic: Dict[str, Any],
    img_folder: str,
    mask_folder: str,
    pc_folder: str,
    img_format: str,
    pc_format: str,
) -> Dict[str, Any]:
    return {
        "cam_K": cam_K,
        "cam_dist": cam_dist,
        "T_lidar_to_cam": extrinsic,
        "T_lidar_to_cam_gt": {
            "available": False,
            "rows": 0,
            "cols": 0,
            "data": [],
        },
        "img_folder": img_folder,
        "mask_folder": mask_folder,
        "pc_folder": pc_folder,
        "img_format": img_format,
        "pc_format": pc_format,
        "file_name": [],
        "params": {
            "min_plane_point_num": None,
            "cluster_tolerance": None,
            "search_num": None,
            "search_range": {
                "rot_deg": None,
                "trans_m": None,
            },
            "point_range": {
                "top": None,
                "bottom": None,
            },
            "down_sample": {
                "is_valid": None,
                "voxel_m": None,
            },
            "thread": {
                "is_multi_thread": None,
                "num_thread": None,
            },
        },
    }


def main() -> None:
    args = parse_args()
    with args.input_json.open("r", encoding="utf-8") as f:
        source = json.load(f)

    intrinsic = source.get("intrinsic_params", {})
    extrinsic = source.get("extrinsic_params", {})

    cam_K = build_cam_K(intrinsic)
    cam_dist = build_cam_dist(intrinsic)
    T_lidar_to_cam = build_extrinsic(extrinsic)

    template = build_template(
        cam_K=cam_K,
        cam_dist=cam_dist,
        extrinsic=T_lidar_to_cam,
        img_folder=args.img_folder,
        mask_folder=args.mask_folder,
        pc_folder=args.pc_folder,
        img_format=args.img_format,
        pc_format=args.pc_format,
    )

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()


