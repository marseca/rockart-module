# dstretch-cli

A fast Python CLI that performs a decorrelation stretch using PCA (Karhunen–Loeve transform). It supports CRGB (PCA directly in RGB), YUV-like space via OpenCV's YCrCb, and both YRD approximation, weighted YRD exact, and a fixed DStretch log mode.

## Install

```bash
pip install -e .
```

## Usage

```bash
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode crgb --scale 2.5
```

If you don't want to install yet, you can run with:

```bash
PYTHONPATH=src python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode crgb --scale 2.5
```

### DStretch YRD approximation (BT.601 YCbCr)

Approximate DStretch YRD using BT.601 YCbCr and PCA:

```bash
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode yrd
```

You can also call it via the preset name (alias):

```bash
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --preset yrd
```

When using YRD without `--scale`, the default is 2.5.

### DStretch YRD exact (weighted PCA)

Weighted YRD variant using BT.601 YCbCr and per-component PCA weights (2.0, 1.0, 0.4):

```bash
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode yrd-exact
```

When using YRD exact without `--scale`, the default is 1.0.

### DStretch fixed log (dstretch.txt)

Use the coefficients and means from a specific DStretch log:

```bash
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode dstretch-log
```

### YUV decorrelation (YCrCb)

Run PCA in YUV-like space using OpenCV's YCrCb conversion:

```bash
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode yuv --scale 2.0
```

Optional per-component factors (Y, U, V):

```bash
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode yuv --factors 2.0 1.0 0.4
```

When using `--factors` without `--scale`, the global scale defaults to 1.0.

### CRGB approximation

CRGB runs PCA directly in RGB (no YUV conversion). Use `--natural` for a more natural look or default false-color mode.

```bash
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode crgb
python -m dstretch_cli --input input/in.jpg --output results/out.jpg --mode crgb --natural
```

Defaults:

- If `--input` is omitted, the first image in `input/` (sorted by name) is used.
- If `--output` is omitted, the file is written to `results/<input_name>_<mode>_s{scale}<input_ext>`.
- Default `--mode` is `crgb`.
- Default `--scale` is `2.5` for `crgb` and `yrd`, `1.0` for `yrd-exact` and `dstretch-log`, `2.0` for `yuv`.
- `--preset yrd` is an alias for `--mode yrd`, `--preset yrd-exact` is an alias for `--mode yrd-exact`, `--preset dstretch-log` is an alias for `--mode dstretch-log`.
- Default `--space` is `yuv` (implemented with OpenCV's YCrCb conversion) when using `--mode yuv`.
- Alpha channels are preserved if present.

## Notes on YUV/YCrCb

OpenCV exposes YCrCb conversions (`cv2.COLOR_RGB2YCrCb` and `cv2.COLOR_YCrCb2RGB`). This tool uses those conversions as a practical YUV-like space for decorrelation.

## Scale

`--scale` controls the strength of the stretch. Higher values increase color separation but can clip highlights or shadows. Start with 2.5 for CRGB or 2.0 for YUV and adjust.

## Tests

```bash
pytest
```
