from __future__ import annotations

import numpy as np

from dstretch_cli.colorspace import _EPS, XFORM_MATRIX_BT601, lxx_to_rgb, rgb_to_lxx, rgb_to_yuv, rgb_to_yxx, yxx_to_rgb

def _pca_components(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Validate expected 3-channel image input.
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("Expected an image with shape (H, W, 3).")

    h, w, _ = x.shape
    # Flatten image into N x 3 for PCA.
    flat = x.reshape(h * w, 3)
    # Mean per channel for centering.
    mu = flat.mean(axis=0)
    # Centered data.
    xc = flat - mu

    n = flat.shape[0]
    # Covariance of centered data (handle tiny images safely).
    if n <= 1:
        c = np.zeros((3, 3), dtype=np.float64)
    else:
        c = (xc.T @ xc) / (n - 1)

    # Eigen decomposition of the covariance matrix.
    eigvals, eigvecs = np.linalg.eigh(c)
    # Sort eigenvectors by descending eigenvalues.
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # Project centered data into PCA space.
    y = xc @ eigvecs
    return y, eigvecs, mu

def pca_decorrelate_and_stretch(
    img3: np.ndarray,
    scale: float,
    channel_scales: tuple[float, float, float] | None = None,
) -> np.ndarray:
    # Compute PCA components for the 3-channel image.
    y, eigvecs, mu = _pca_components(img3)

    # Standard deviation per principal component (avoid divide-by-zero).
    sigma = y.std(axis=0)
    sigma = np.where(sigma < _EPS, 1.0, sigma)
    # Determine the target stretch per component.
    if channel_scales is None:
        # Uniform stretch based on the average variance.
        target = sigma.mean() * scale
    else:
        # Per-component weighting (e.g., Y/U/V emphasis).
        factors = np.asarray(channel_scales, dtype=np.float32)
        if factors.shape != (3,):
            raise ValueError("channel_scales must have exactly 3 values.")
        target = sigma.mean() * scale * factors
    # Normalize to unit variance, then scale to the target.
    y_stretched = (y / sigma) * target

    # Inverse PCA and re-add the mean to restore the original center.
    x2 = y_stretched @ eigvecs.T + mu
    out = x2.reshape(img3.shape)
    return out

def apply_crgb_approximation(
    img_rgb: np.ndarray,
    scale: float = 2.50,
    false_color: bool = True,
) -> np.ndarray:
    """
    Approximates DStretch CRGB using PCA directly in RGB space.

    Args:
        img_rgb: Input image (float, [0, 1]).
        scale: Stretch factor (DStretch default is roughly 2.50 for this logic).
        false_color:
            If True, maps PCs directly to RGB (false-color).
            If False, applies inverse transform (natural colors).
    """
    y, eigvecs, mu = _pca_components(img_rgb)

    sigma = y.std(axis=0)
    sigma = np.where(sigma < _EPS, 1.0, sigma)
    target = sigma.mean() * scale
    y_stretched = (y / sigma) * target

    if false_color:
        out_flat = y_stretched + 0.5
        out = out_flat.reshape(img_rgb.shape)
    else:
        x2 = y_stretched @ eigvecs.T + mu
        out = x2.reshape(img_rgb.shape)

    return np.clip(out, 0.0, 1.0)

def apply_yxx_approximation(img_rgb: np.ndarray, scale: float = 2.50, channel_scales: tuple[float, float, float] = [1.0, 1.0, 1.0]) -> np.ndarray:
    """
    Approximates DStretch YXX by converting to YUV, applying PCA,
    and converting back.
    """
    """
        BT.601 | SD TV / Old JPEGs | Y = 0.299R + 0.587G + 0.114B
        BT.709 HD TV / Modern Web | Y = 0.2126R + 0.7152G + 0.0722B
        BT.2020 | Ultra HD / HDR | Y = 0.2627R + 0.6780G + 0.0593B
    """

     # The coefficients for the $Y$ (Luminance) channel in the BT.601 
    # standard (the one used for standard-definition video) are:
    """
        Here is why those specific weights exist:

        Green (58.7%): Our eyes are most sensitive to green light because our "M" (medium-wavelength) and "L" (long-wavelength) cones have significant overlap in the green-yellow part of the spectrum.

        Red (29.9%): We have a moderate sensitivity to red.

        Blue (11.4%): We are surprisingly "blind" to blue brightness. We use blue mostly for color detail, not for defining shapes or edges.
    """

    """
        The $U$ and $V$ channels (often called $Cb$ and $Cr$ in digital systems) 
        are designed to store only color information by subtracting the brightness ($Y$) 
        from the original $R$ and $B$ signals.
        U (Chroma Blue): Proportional to $(B - Y)$. Since $Y$ already contains the "brightness" part of blue, subtracting it leaves only the "blueness.
        "V (Chroma Red): Proportional to $(R - Y)$. This leaves only the "redness."

        If you have $Y$, $U$, and $V$, you can mathematically reconstruct Green
    """

    ycbcr= rgb_to_yuv(img_rgb)
    # ycbcr[:, :, 1:] += 0.5

    ycbcr_stretched = pca_decorrelate_and_stretch(ycbcr, scale=scale, channel_scales=channel_scales)
    # ycbcr_stretched[:, :, 1:] -= 0.5
    inverse_matrix = np.array(
        [
            [1.0, 0.0, 1.402],
            [1.0, -0.34414, -0.71414],
            [1.0, 1.772, 0.0],
        ],
        dtype=np.float32,
    )

    img_out = ycbcr_stretched @ inverse_matrix.T
    return np.clip(img_out, 0.0, 1.0)

def apply_yxx_pca_stretch(
    img_rgb: np.ndarray,
    scale: float = 2.50,
    yxx_scales: tuple[float, float, float] = [1.0, 0.8, 0.4],
    clip: bool = True,
) -> np.ndarray:
    """
    DStretch-like YXX: RGB -> YXX -> PCA decorrelate + stretch -> RGB.

    `yxx_scales` is optional and keeps your existing PCA weighting logic.
    Leave it as None to mimic standard YXX behavior.
    """
    yxxmuly, yxxmulu, yxxmulv = yxx_scales
    yxx = rgb_to_yxx(
        img_rgb,
        yxxmuly=yxxmuly,
        yxxmulu=yxxmulu,
        yxxmulv=yxxmulv,
    )

    yxx_stretched = pca_decorrelate_and_stretch(
        yxx,
        scale=scale,
    )

    out = yxx_to_rgb(
        yxx_stretched,
        yxxmuly=yxxmuly,
        yxxmulu=yxxmulu,
        yxxmulv=yxxmulv,
    )

    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out

def apply_lxx_pca_stretch(
    img_rgb: np.ndarray,
    scale: float = 2.50,
    lxx_scales: tuple[float, float, float, float] | None = None,
    clip: bool = True,
) -> np.ndarray:
   
    lxxmul1, lxxmul2, lxxmula, lxxmulb = lxx_scales
    
    lxx = rgb_to_lxx(
        img_rgb,
        lxxmul1=lxxmul1,
        lxxmul2=lxxmul2,
        lxxmula=lxxmula,
        lxxmulb=lxxmulb,
    )

    lxx_stretched = pca_decorrelate_and_stretch(
        lxx,
        scale=scale
    )

    out = lxx_to_rgb(
        lxx_stretched,
        lxxmul1=lxxmul1,
        lxxmul2=lxxmul2,
        lxxmula=lxxmula,
        lxxmulb=lxxmulb,
    )

    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out
