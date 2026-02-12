from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from .utils import IMAGE_EXTS


def load_image_rgb(path: str | Path) -> np.ndarray:
    rgb, _ = load_image_with_alpha(path)
    return rgb


def load_image_with_alpha(path: str | Path) -> Tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 2:
        raise ValueError("Grayscale images are not supported; expected 3-channel RGB.")

    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb, None

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, alpha

    raise ValueError(f"Unsupported channel count: {img.shape[2]}")


def load_image_with_alpha(path: str | Path) -> Tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    if path.suffix.lower() not in IMAGE_EXTS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if img.ndim == 2:
        raise ValueError("Grayscale images are not supported; expected 3-channel RGB.")

    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb, None

    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, alpha

    raise ValueError(f"Unsupported channel count: {img.shape[2]}")
    
def save_image_rgb(path: str | Path, img_rgb: np.ndarray) -> None:
    path = Path(path)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise IOError(f"Failed to write image: {path}")


def save_image_rgba(path: str | Path, img_rgb: np.ndarray, alpha: np.ndarray) -> None:
    path = Path(path)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    bgra = np.dstack([bgr, alpha])
    if not cv2.imwrite(str(path), bgra):
        raise IOError(f"Failed to write image: {path}")


def save_image_gray(path: str | Path, img_gray: np.ndarray) -> None:
    """
    Saves a 2D numpy array (like a YUV channel) as a grayscale image.
    """
    # Convert Path object to string for OpenCV compatibility
    save_path = str(path)
    
    # Ensure the directory exists if a Path was provided
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)

    # OpenCV expects uint8 for standard image formats (0-255)
    # If your YUV data is float or int16, you might need to scale/clip it first.
    if img_gray.dtype != np.uint8:
        img_gray = np.clip(img_gray, 0, 255).astype(np.uint8)

    # Write the image to disk
    success = cv2.imwrite(save_path, img_gray)
    
    if not success:
        raise IOError(f"Failed to save image to {save_path}")
