from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .colorspace import rgb_to_yuv, yuv_to_rgb
from .io import load_image_with_alpha, save_image_gray, save_image_rgb, save_image_rgba
from .pca_stretch import (
    apply_crgb_approximation,
    apply_lxx_pca_stretch,
    apply_yxx_approximation,
    apply_yxx_pca_stretch,
)
from .utils import ensure_parent, list_images


def _default_input() -> Path:
    images = list_images(Path("output"))
    if not images:
        raise FileNotFoundError("No input image found in output/.")
    return images[0]


def _default_output(input_path: Path, scale: float, mode: str) -> Path:
    name = input_path.stem
    ext = input_path.suffix if input_path.suffix else ".png"
    out_name = f"{name}_{mode}_s{scale}{ext}"
    return Path("results") / out_name


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PCA-based decorrelation stretch in CRGB (RGB PCA) or YUV (YCrCb)."
    )
    parser.add_argument("--input", type=str, default=None, help="Input image path")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--scale", type=float, default=None, help="Stretch scale")
    parser.add_argument(
        "--mode",
        type=str,
        default="crgb",
        choices=["crgb", "yxx-aproximation", "yxx", "lxx"],
        help=(
            "Processing mode: CRGB PCA in RGB, YXX-Aproximation PCA, YXX PCA, LXX PCA"
        ),
    )
    parser.add_argument(
        "--factors",
        type=float,
        nargs=3,
        default=None,
        metavar=("Y", "U", "V"),
        help="Per-component stretch factors (3 values)",
    )
    fc_group = parser.add_mutually_exclusive_group()
    fc_group.add_argument(
        "--false-color",
        action="store_true",
        help="CRGB false-color mode (maps PCs to RGB)",
    )
    fc_group.add_argument(
        "--natural",
        action="store_true",
        help="CRGB natural-color mode (inverse PCA)",
    )
    fc_group.add_argument(
        "--export-channels",
        action="store_true",
        help="Export YUV as greyscale",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    mode = args.mode

    if mode == "crgb" and args.factors:
        raise ValueError("CRGB mode does not support --factors.")
    if mode != "crgb" and (args.false_color or args.natural):
        raise ValueError("False-color and natural options are only for CRGB mode.")
    
    factors: tuple[float, float, float] | None
    if "yxx" in mode and args.factors is not None:
        factors = (float(args.factors[0]), float(args.factors[1]), float(args.factors[2]))
    elif "lxx" in mode and args.factors is not None:
        factors = (float(args.factors[0]), float(args.factors[0]), float(args.factors[1]), float(args.factors[2]))
    else:
        factors = (1.0, 1.0, 1.0)

    if args.scale is None:
        scale = 1.0
    else:
        scale = float(args.scale)

    input_path = Path(args.input) if args.input else _default_input()
    output_path = (
        Path(args.output) if args.output else _default_output(input_path, scale, mode)
    )

    rgb, alpha = load_image_with_alpha(input_path)

    img_rgb = rgb.astype(np.float32) / 255.0
    if mode == "crgb":
        false_color = True if not args.natural else False
        rgb_out_f = apply_crgb_approximation(
            img_rgb, scale=scale, false_color=false_color
        )
        rgb_out = np.clip(np.rint(rgb_out_f * 255.0), 0, 255).astype(np.uint8)
    elif mode == "yxx-approximation":
        # Convert input RGB image to float32 in [0, 1] range for processing
        rgb_out_f = apply_yxx_approximation(img_rgb, scale=scale, channel_scales=factors)
        # rgb_out_f * 255.0: Scales the floating-point RGB values (typically in the range [0, 1]) up to the standard 8-bit range [0, 255].
        # np.rint(...): Rounds the scaled values to the nearest integer, which helps avoid truncation errors.
        # np.clip(..., 0, 255): Ensures all values stay within the valid 8-bit range (0 to 255), preventing overflow or underflow.
        # .astype(np.uint8): Converts the resulting array to 8-bit unsigned integers, which is the standard format for image data.
        rgb_out = np.clip(np.rint(rgb_out_f * 255.0), 0, 255).astype(np.uint8)
        # export_channels = True if not args.export_channels else False
        # if(export_channels):
    elif mode == "yxx":
        rgb_out_f = apply_yxx_pca_stretch(
            img_rgb, 
            scale=scale, 
            yxx_scales=factors)
    elif mode == "lxx":
        rgb_out_f = apply_lxx_pca_stretch(
            img_rgb, 
            scale=scale, 
            lxx_scales=factors)
        
        rgb_out = np.clip(np.rint(rgb_out_f * 255.0), 0, 255).astype(np.uint8)

    ensure_parent(output_path)
    if alpha is None:
        save_image_rgb(output_path, rgb_out)
    else:
        save_image_rgba(output_path, rgb_out, alpha)

    return 0
