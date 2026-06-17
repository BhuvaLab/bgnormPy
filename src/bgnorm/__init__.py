from importlib.metadata import PackageNotFoundError, version

from bgnorm.src.core import (
    ArcsinhTransform,
    BgNormArrays,
    BgNormChannel,
    BgNormConfig,
    ImageLike,
    Log2Transform,
    MedianFilter,
    Moments,
    PostHocQuantile,
    ScoreMetrics,
    bgnorm,
    default_pipeline,
)

try:
    __version__ = version("bgnorm")
except PackageNotFoundError:  # not installed (e.g. running from source tree)
    __version__ = "0.0.0"

__all__ = [
    "ArcsinhTransform",
    "BgNormArrays",
    "BgNormChannel",
    "BgNormConfig",
    "ImageLike",
    "Log2Transform",
    "MedianFilter",
    "Moments",
    "PostHocQuantile",
    "ScoreMetrics",
    "bgnorm",
    "default_pipeline",
]