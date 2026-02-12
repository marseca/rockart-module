import numpy as np

from dstretch_cli.pca_stretch import (
    _pca_components,
    apply_crgb_approximation,
    apply_dstretch_from_log,
    apply_yrd_exact,
    apply_yxx_approximation,
    pca_decorrelate_and_stretch,
)


def _offdiag_sum(cov: np.ndarray) -> float:
    return float(np.sum(np.abs(cov)) - np.sum(np.abs(np.diag(cov))))


def test_pca_stretch_basic():
    base = np.linspace(0, 255, 16, dtype=np.float32).reshape(4, 4)
    ch0 = base
    ch1 = base * 0.8 + 10
    ch2 = base * 0.5 + 20
    img = np.stack([ch0, ch1, ch2], axis=2)

    out = pca_decorrelate_and_stretch(img, scale=2.0)
    assert out.shape == img.shape
    assert np.isfinite(out).all()

    clipped = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    assert clipped.min() >= 0 and clipped.max() <= 255

    flat_in = img.reshape(-1, 3)
    cov_in = np.cov(flat_in, rowvar=False)
    off_in = _offdiag_sum(cov_in)

    y, eigvecs, mu = _pca_components(img)
    sigma = y.std(axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    y_stretched = (y / sigma) * (sigma.mean() * 2.0)
    cov_y = np.cov(y_stretched, rowvar=False)
    off_y = _offdiag_sum(cov_y)

    assert off_y <= off_in * 0.1 + 1e-5


def test_pca_stretch_with_factors():
    img = np.random.default_rng(0).uniform(0, 255, size=(4, 4, 3)).astype(np.float32)
    out = pca_decorrelate_and_stretch(img, scale=1.0, channel_scales=(2.0, 1.0, 0.4))
    assert out.shape == img.shape
    assert np.isfinite(out).all()


def test_crgb_approximation_range():
    img = np.random.default_rng(1).uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    out = apply_crgb_approximation(img, scale=2.5, false_color=True)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_yrd_approximation_range():
    img = np.random.default_rng(2).uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    out = apply_yxx_approximation(img, scale=2.5)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_yrd_exact_range():
    img = np.random.default_rng(3).uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    out = apply_yrd_exact(img, scale=1.0)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_dstretch_log_range():
    img = np.random.default_rng(4).uniform(0, 1, size=(4, 4, 3)).astype(np.float32)
    out = apply_dstretch_from_log(img)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    assert out.min() >= 0.0 and out.max() <= 1.0
