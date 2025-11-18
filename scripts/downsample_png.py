#!/usr/bin/env python3
"""
Utility to downsample PNG images.

Examples:
    # Downsample a single image by 50% and write next to the original
    python scripts/downsample_png.py --input data/cam02/images/000000.png --scale 0.5

    # Downsample an entire directory so the longest side is 1024 px
    python scripts/downsample_png.py --input data/cam02/images --longest-side 1024 --output /tmp/cam02_small

    # Force every image to exactly 640x480 (might change aspect ratio)
    python scripts/downsample_png.py --input data/cam02/images --size 640 480
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - guidance for user
    raise SystemExit(
        "Pillow is required. Install it with `pip install Pillow`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample PNG images.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a PNG image or a directory containing PNGs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory. If omitted, downsampled files are stored next to "
            "the originals (single file) or under <input>_downsampled (directory)."
        ),
    )
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument(
        "--scale",
        type=float,
        help="Uniform scaling factor (e.g., 0.5 halves both width and height).",
    )
    size_group.add_argument(
        "--longest-side",
        type=int,
        dest="longest_side",
        help="Resize so the longest edge equals this many pixels.",
    )
    size_group.add_argument(
        "--size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        help="Resize to an exact size (width height). Aspect ratio may change.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing files in the output location.",
    )
    return parser.parse_args()


def iter_pngs(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() != ".png":
            raise SystemExit(f"Input file must be a PNG: {path}")
        yield path
        return

    if not path.is_dir():
        raise SystemExit(f"Input path does not exist: {path}")

    for candidate in sorted(path.rglob("*.png")):
        if candidate.is_file():
            yield candidate


def compute_size(
    img: Image.Image,
    scale: float | None,
    longest: int | None,
    explicit_size: Tuple[int, int] | None,
) -> Tuple[int, int]:
    if explicit_size:
        width, height = explicit_size
        if width <= 0 or height <= 0:
            raise SystemExit("--size WIDTH HEIGHT requires positive integers.")
        return width, height

    if scale:
        if scale <= 0:
            raise SystemExit("--scale must be > 0.")
        width = max(1, int(round(img.width * scale)))
        height = max(1, int(round(img.height * scale)))
        return width, height

    if not longest or longest <= 0:
        raise SystemExit("--longest-side must be a positive integer when provided.")

    current_longest = max(img.width, img.height)
    if current_longest <= longest:
        return img.width, img.height

    ratio = longest / current_longest
    return (
        max(1, int(round(img.width * ratio))),
        max(1, int(round(img.height * ratio))),
    )


def resolve_output_path(src: Path, input_root: Path, output_root: Path | None) -> Path:
    if output_root is None:
        if input_root.is_file():
            # place alongside original
            return src.with_stem(src.stem + "_downsampled")
        base = input_root.parent / f"{input_root.name}_downsampled"
        return base / src.relative_to(input_root)
    if output_root.is_file():
        raise SystemExit("--output must be a directory when processing multiple images.")
    if input_root.is_file():
        return output_root
    return output_root / src.relative_to(input_root)


def downsample_image(src: Path, dst: Path, size: Tuple[int, int], overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise SystemExit(f"Output file exists (use --overwrite to replace): {dst}")

    with Image.open(src) as img:
        if img.size == size:
            if src == dst:
                return  # nothing to do
            img.save(dst)
            return
        resized = img.resize(size, Image.Resampling.LANCZOS)
        resized.save(dst)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output.resolve() if args.output else None

    pngs = list(iter_pngs(input_path))
    if not pngs:
        raise SystemExit("No PNG files found to process.")

    for src in pngs:
        with Image.open(src) as img:
            target_size = compute_size(
                img,
                args.scale,
                args.longest_side,
                tuple(args.size) if args.size else None,
            )
        dst = resolve_output_path(src, input_path, output_root)
        downsample_image(src, dst, target_size, args.overwrite)
        print(f"{src} -> {dst} ({target_size[0]}x{target_size[1]})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)

