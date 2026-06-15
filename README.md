# bgnorm

## Overview

**bgnorm** is a Gaussian mixture model (GMM)-based method for background correction and normalization of spatial imaging data, particularly multiplexed spatial proteomics images. The package models intensity distributions using a three-component mixture model and performs marker- and pixel- or cell-level signal adjustment to remove technical background while preserving biological variation.

bgnorm provides tools for processing multi-channel imaging intensities and generating background-adjusted measurements suitable for downstream spatial and single-cell analyses.

### Theoretical Background

bgnorm assumes three types of pixels: - **Background** ($X_1 = U_1$): Only background signals - **Non-specific binding** ($X_2 = U_1 + U_2$): Background + non-specific binding - **Signal** ($X_3 = U_1 + U_2 + U_3$): Background + non-specific + true biological signal

![](assets/bgnorm.png)

bgnorm uses a statistical mixture model to separate these three signal types and estimate how likely each pixel is to contain real biological signal. For pixels that likely contain biological signal, bgnorm subtracts estimated background and non-specific contributions and rescales the remaining signal. Because each pixel is only probabilistically assigned to a signal type, bgnorm performs a soft correction rather than hard thresholding. This produces smooth, background-adjusted intensities at the pixel level, which can then be aggregated to cells for downstream spatial and single-cell analyses.

## Installation

Install the development version from GitHub, with either `pip` or `uv`:

**pip**

```bash
# latest on the default branch
pip install "git+https://github.com/BhuvaLab/bgnormPy.git"

# pin to a branch, tag, or commit
pip install "git+https://github.com/BhuvaLab/bgnormPy.git@main"
pip install "git+https://github.com/BhuvaLab/bgnormPy.git@v0.1.0"
```

**uv**

```bash
uv add "git+https://github.com/BhuvaLab/bgnormPy.git" 
uv add "git+https://github.com/BhuvaLab/bgnormPy.git" --tag v0.1.0
uv pip install "git+https://github.com/BhuvaLab/bgnormPy.git" # or into current env 
```

Or declare it via uv's git source in `pyproject.toml`:

```toml
[project]
dependencies = ["bgnorm"]

[tool.uv.sources]
bgnorm = { git = "https://github.com/BhuvaLab/bgnormPy.git", branch = "main" }
```

**Local / development** — install from a clone of this repo (assuming uv tooling):

```bash
git clone https://github.com/BhuvaLab/bgnormPy.git 
uv pip install -e . 
uv sync
```

### Requirements
- Python >= 3.12
- dask >= 2026.6.0
- dask-image >= 2026.5.0
- numpy >= 2.0
- pandas >= 3.0.3
- pydantic >= 2.13.4
- pyyaml >= 6.0
- scikit-learn >= 1.9.0
- scipy >= 1.17.1
- xarray >= 2026.4.0

## Quick Start

### Processing Pipeline

#### GMM Normalization Pipeline:

1.  **Read & Filter**: Load intensity data and apply median filtering
2.  **Log Transform**: Apply log2(x / cofactor + 1) transformation
3.  **Fit GMM**: Fit 3-component Gaussian mixture model to identify:
    -   Component 1: Background pixels
    -   Component 2: Non-specific binding pixels
    -   Component 3: Signal pixels
4.  **Classify**: Assign each pixel to a mixture component
5.  **Adjust**: Apply variance-weighted deconvolution to remove background from singal
6.  **Normalize**: Apply quantile normalization for cross-sample comparability
7.  **Aggregate**: Compute per-cell median intensities

### High-Level Workflow

The easiest way to run a complete GMM normalization workflow with a single function entrypoint:

```{python}
```

### Scikit-Learn Composition
BgNorm steps exist as scikit-learn compatible modules and can be used to compose the bgnorm function as a scikit-learn Pipeline. We recommend using the `BgNormConfig` schema to validate parameters:

```{python}
from bgnorm import (
    BgNormConfig,
    MedianFilter,
    Log2Transform,
    BgNormChannel,
    PostHocQuantile
)
from sklearn.pipeline import Pipeline

cfg = BgNormConfig(
    median_filter_radius=3,
    image_cofactor=150,
    n_components=3,
    n_pixels_to_sample=1e5,
    pixel_sampling_seed=42,
    quantile_post_hoc_value=0.75
)

steps = [
    ("median_filter", MedianFilter(cfg.median_filter_radius)),
    ("log2_transform", Log2Transform(cfg.image_cofactor)),
    ("bgnorm", BgNormChannel(
        cfg.n_components, 
        cfg.n_pixels_to_sample, 
        cfg.pixel_sampling_seed
        )
    ),
    ("post_hoc_quantile", PostHocQuantile(cfg.quantile_post_hoc_value))
]

pp = Pipeline(steps)
adjusted_image = pp.fit_transform(...)
```