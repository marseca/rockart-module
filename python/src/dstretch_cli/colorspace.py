from __future__ import annotations

import cv2
import numpy as np

_EPS = 1e-8


# def rgb_to_yuv(img_rgb: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)

# def yuv_to_rgb(img_yuv: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

XFORM_MATRIX_BT601 = np.array(
    [
        [0.299, 0.587, 0.114],        # Y (Luminance)
        [-0.168736, -0.331264, 0.5],  # U (Cb, Chroma Blue)
        [0.5, -0.418688, -0.081312],  # V (Cr, Chroma Red)
    ],
    dtype=np.float32,
)

XFORM_INV_MATRIX_BT601 = np.array(
   [
        [1.0, 0.0, 1.402],
        [1.0, -0.344136, -0.714136],
        [1.0, 1.772, 0.0],
    ],
    dtype=np.float32,
)

# DStretch uses D65 white
_D65_XN = 95.047
_D65_YN = 100.0
_D65_ZN = 108.883

def _srgb_to_linear(c):
    """
    Converts sRGB color values to linear RGB values.

    Parameters:
        c (numpy.ndarray or float): Input sRGB color value(s), expected in the range [0.0, 1.0].

    Returns:
        numpy.ndarray or float: Linear RGB value(s) corresponding to the input sRGB value(s).

    Notes:
        - Values outside the [0.0, 1.0] range are clipped before conversion.
        - The conversion follows the standard sRGB to linear RGB formula.
    """
    c = np.clip(c, 0.0, 1.0)
    return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)

def _linear_to_srgb(c):
    c = np.clip(c, 0.0, None)
    return np.where(c > 0.0031308, 1.055 * (c ** (1.0 / 2.4)) - 0.055, 12.92 * c)

def _f_lab(t):
    return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)

def _f_lab_inv(t):
    t3 = t ** 3
    return np.where(t3 > 0.008856, t3, (t - 16.0 / 116.0) / 7.787)

def rgb_to_xyz(img_rgb):
    """
    Converts an sRGB image to the CIE XYZ color space (D65).

    Parameters:
        img_rgb (np.ndarray): Input image in sRGB color space, shape (H, W, 3), values in [0.0, 1.0].

    Returns:
        np.ndarray: Image in XYZ color space, shape (H, W, 3), values scaled to 0..100.
    """
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    r = _srgb_to_linear(r)
    g = _srgb_to_linear(g)
    b = _srgb_to_linear(b)
    # sRGB D65 matrix, output scaled to 0..100 like DStretch
    x = (0.4124 * r + 0.3576 * g + 0.1805 * b) * 100.0
    y = (0.2126 * r + 0.7152 * g + 0.0722 * b) * 100.0
    z = (0.0193 * r + 0.1192 * g + 0.9505 * b) * 100.0
    return np.stack([x, y, z], axis=-1)

def xyz_to_rgb(img_xyz):
    x, y, z = img_xyz[..., 0] / 100.0, img_xyz[..., 1] / 100.0, img_xyz[..., 2] / 100.0
    # sRGB D65 inverse matrix
    r =  3.2406 * x + -1.5372 * y + -0.4986 * z
    g = -0.9689 * x +  1.8758 * y +  0.0415 * z
    b =  0.0557 * x + -0.2040 * y +  1.0570 * z
    rgb = np.stack([r, g, b], axis=-1)
    return _linear_to_srgb(rgb)

def rgb_to_lxx(
    img_rgb: np.ndarray,
    *,
    lxxmul1: float = 1.0,
    lxxmul2: float = 1.0,
    lxxmula: float = 1.0,
    lxxmulb: float = 1.0,
) -> np.ndarray:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Expected an image with shape (H, W, 3).")

    xyz = rgb_to_xyz(img_rgb)
    x = xyz[..., 0] / _D65_XN
    y = xyz[..., 1] / _D65_YN
    z = xyz[..., 2] / _D65_ZN

    fx = _f_lab(x)
    fy = _f_lab(y)
    fz = _f_lab(z)

    l = 116.0 * fy - 16.0
    a = (1.0 / lxxmul1) * 250.0 * (fx - lxxmula * fy)
    b = (1.0 / lxxmul2) * 100.0 * (lxxmulb * fy - fz)
    return np.stack([l, a, b], axis=-1)

def lxx_to_rgb(
    img_lxx: np.ndarray,
    *,
    lxxmul1: float = 1.0,
    lxxmul2: float = 1.0,
    lxxmula: float = 1.0,
    lxxmulb: float = 1.0,
) -> np.ndarray:
    if img_lxx.ndim != 3 or img_lxx.shape[2] != 3:
        raise ValueError("Expected an image with shape (H, W, 3).")

    l = img_lxx[..., 0]
    a = img_lxx[..., 1]
    b = img_lxx[..., 2]

    fy = (l + 16.0) / 116.0
    fx = lxxmul1 * a * 0.004 + lxxmula * fy
    fz = fy * lxxmulb - lxxmul2 * b * 0.01

    x = _f_lab_inv(fx) * _D65_XN
    y = _f_lab_inv(fy) * _D65_YN
    z = _f_lab_inv(fz) * _D65_ZN
    xyz = np.stack([x, y, z], axis=-1)
    return xyz_to_rgb(xyz)

def rgb_to_yuv(img_rgb: np.ndarray) -> np.ndarray:
    xform_matrix = XFORM_MATRIX_BT601
    return img_rgb @ xform_matrix.T
    

def yuv_to_rgb(img_yuv: np.ndarray) -> np.ndarray:
    # Inverse matrix for YUV (BT.601) to RGB conversion
    inverse_matrix = XFORM_INV_MATRIX_BT601

    img_out = img_yuv @ inverse_matrix.T
    return np.clip(img_out, 0.0, 1.0)


def rgb_to_yxx(
    img_rgb: np.ndarray,
    *,
    yxxmuly: float = 1.0,
    yxxmulu: float = 0.8,
    yxxmulv: float = 0.4,
) -> np.ndarray:
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("Expected an image with shape (H, W, 3).")

    r = img_rgb[..., 0]
    g = img_rgb[..., 1]
    b = img_rgb[..., 2]

    # DStretch YXX forward transform
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = yxxmuly * (b - yxxmulu * y)
    v = yxxmuly * (r - yxxmulv * y)

    return np.stack([y, u, v], axis=-1)


def yxx_to_rgb(
    img_yxx: np.ndarray,
    *,
    yxxmuly: float = 1.0,
    yxxmulu: float = 0.8,
    yxxmulv: float = 0.4,
) -> np.ndarray:
    if img_yxx.ndim != 3 or img_yxx.shape[2] != 3:
        raise ValueError("Expected an image with shape (H, W, 3).")

    if abs(yxxmuly) < _EPS:
        raise ValueError("yxxmuly must be non-zero for inverse YXX transform.")

    y = img_yxx[..., 0]
    u = img_yxx[..., 1]
    v = img_yxx[..., 2]

    # DStretch YXX inverse transform
    r = v / yxxmuly + yxxmulv * y
    b = u / yxxmuly + yxxmulu * y
    g = (y - 0.299 * r - 0.114 * b) / 0.587

    return np.stack([r, g, b], axis=-1)