from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

import numpy as np
import dask.array as da
import pandas as pd
import scipy.stats as stats
from dask_image.ndfilters import median_filter
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
import xarray as xr

# type helpers
ImageLike = xr.DataArray | np.ndarray | da.Array
PipelineFactory = Callable[[], Pipeline]


class BgNormConfig(BaseModel):
    """Validated, user-facing parameters for the bgnorm pipeline."""
    # TODO: yaml parser for larger experiement sweeps
    model_config = ConfigDict(frozen=True, extra="forbid")

    median_filter_radius: int = Field(3, ge=1, description="Square median-filter radius (px).")
    image_cofactor: int = Field(150, ge=1, description="log2(x / cofactor + 1) divisor.")
    n_components: int = Field(3, ge=2, description="GMM mixture components.")
    n_pixels_to_sample: int = Field(int(1e5), ge=1, description="Pixels sampled to fit the GMM.")
    pixel_sampling_seed: int = Field(0, ge=0, description="Seed for pixel sampling.")
    quantile_post_hoc_value: float | None = Field(
        None, gt=0, lt=1, description="If set, append PostHocQuantile(q); else omit it."
    )


def _resolve_config(config: BgNormConfig | None, overrides: dict[str, object]) -> BgNormConfig:
    """Merge keyword overrides onto a config (or defaults), re-validating the result."""
    base = config.model_dump() if config is not None else {}
    return BgNormConfig(**{**base, **overrides})


class Moments(TypedDict):
    """GMM-derived quantities the adjustment + post-hoc steps depend on."""
    mean_signal: float
    mean_background: float
    var_signal: float
    var_background: float
    weight_signal: float


class ScoreMetrics(TypedDict):
    """Per-channel goodness-of-fit metrics."""
    cohens_d: float
    bic: float
    log_likelihood: float
    signal_weight: float


def _values(X: ImageLike) -> np.ndarray | da.Array:
    """Underlying array from a DataArray, or the array itself if already raw."""
    return X.data if isinstance(X, xr.DataArray) else X


def _as_dask(X: ImageLike) -> da.Array:
    """Underlying array as a (lazy) dask array; accepts DataArray / numpy / dask."""
    arr = _values(X)
    return arr if isinstance(arr, da.Array) else da.asarray(arr)


def _rewrap(X: ImageLike, data, attrs: dict | None = None) -> ImageLike:
    """Return `data` in the same flavour as the input X. Mainly to accomodate
    multiple entryp oints.
    """
    if isinstance(X, xr.DataArray):
        out = X.copy(data=data)
        if attrs:
            out.attrs.update(attrs)
        return out
    if isinstance(X, da.Array):
        return data
    return np.asarray(data)

class MedianFilter(BaseEstimator, TransformerMixin):
    """radiusxradius (one-connectivity) square median filter over a single channel. """

    def __init__(self, radius: int = 3) -> None:
        self.radius = radius

    def fit(self, X: ImageLike, y: None = None) -> MedianFilter:
        return self

    def transform(self, X: ImageLike) -> ImageLike:
        filtered = median_filter(_as_dask(X), size=(self.radius, self.radius))
        return _rewrap(X, filtered)

class Log2Transform(BaseEstimator, TransformerMixin):
    """log2(X / cofactor + 1) scalar transform"""

    def __init__(self, cofactor: int = 150) -> None:
        self.cofactor = cofactor

    def fit(self, X: ImageLike, y: None = None) -> Log2Transform:
        if self.cofactor < 1:
            raise ValueError("Cofactor must be 1 or greater")
        return self

    def transform(self, X: ImageLike) -> ImageLike:
        return _rewrap(X, da.log2(_as_dask(X) / self.cofactor + 1))

class ArcsinhTransform(BaseEstimator, TransformerMixin):
    """Arcsinh transform with cofactor"""
    def __init__(self, cofactor: int = 150) -> None:
        self.cofactor = cofactor

    def fit(self, X: ImageLike, y: None = None) -> ArcsinhTransform:
        return self

    def transform(self, X: ImageLike) -> ImageLike:
        return _rewrap(X, da.arcsinh(_as_dask(X) / self.cofactor))


class BgNormChannel(BaseEstimator, TransformerMixin):
    """
    BGNorm method on one channel.

    .fit(X); fits the conv of gaussians using GMMs
    .transform(X); returns adjusted DataArray
    .score(X); metrics of goodness fits {cohens_d, bic, log_likelihood, signal_weight}.
    """

    # fitted attributes (set in fit/transform), declared for type checkers
    gmm_: GaussianMixture
    rho_: float
    moments_: Moments
    sample_: np.ndarray
    signal_component_: int
    background_component_: int
    labels_: np.ndarray
    probabilities_: np.ndarray

    def __init__(
        self,
        n_components: int = 3, # NOTE: this might become static variable sicne bgnorm core assumption is ncomp=3
        n_pixels_to_sample: int = int(1e5),
        pixel_sampling_seed: int = 0,
    ) -> None:
        self.n_components = n_components
        self.n_pixels_to_sample = n_pixels_to_sample
        self.pixel_sampling_seed = pixel_sampling_seed

    def fit_transform(self, X: xr.DataArray, y: None = None, **fit_params: object) -> xr.DataArray:
        if hasattr(X, "compute"):
            X = X.compute()
        return self.fit(X, **fit_params).transform(X)

    def fit(
        self,
        X: ImageLike,
        y: None = None,
        mask: np.ndarray | None = None,
    ) -> BgNormChannel:
        flat = np.asarray(_values(X)).flatten()

        positive = flat[flat >= 0] if mask is None else flat[(flat >= 0) & mask.flatten()]
        if len(positive) < self.n_components + 1:
            raise ValueError("Not enough positive pixels to fit GMM")

        nsample = min(self.n_pixels_to_sample, len(positive))
        rng = np.random.default_rng(self.pixel_sampling_seed)
        sample = rng.choice(positive, int(nsample), replace=False)

        gmm = GaussianMixture(n_components=self.n_components, covariance_type="full")
        gmm.fit(sample.reshape(-1, 1))

        # reorder components by mean (ascending)
        order = np.argsort(gmm.means_.flatten())
        gmm.means_ = gmm.means_[order]
        gmm.covariances_ = gmm.covariances_[order]
        gmm.weights_ = gmm.weights_[order]

        # decide which of the top-two components is signal vs background by
        # comparing densities at the 0.99 quantile of the top component
        xvec = stats.norm.ppf(0.99, gmm.means_[-1][0], np.sqrt(gmm.covariances_[-1]))
        pdf_2nd = stats.norm.pdf(xvec, gmm.means_[-2][0], np.sqrt(gmm.covariances_[-2]))
        pdf_top = stats.norm.pdf(xvec, gmm.means_[-1][0], np.sqrt(gmm.covariances_[-1]))
        if pdf_2nd > pdf_top:
            self.background_component_ = self.n_components - 1
            self.signal_component_ = self.n_components - 2
        else:
            self.background_component_ = self.n_components - 2
            self.signal_component_ = self.n_components - 1

        var_sig = gmm.covariances_[self.signal_component_][0, 0]
        var_bg = gmm.covariances_[self.background_component_][0, 0]
        sign_flip = -1 if var_sig < var_bg else 1

        self.gmm_ = gmm
        self.rho_ = sign_flip * (min(var_sig, var_bg) / var_sig)
        self.moments_ = {
            "mean_signal": gmm.means_[self.signal_component_][0],
            "mean_background": gmm.means_[self.background_component_][0],
            "var_signal": var_sig,
            "var_background": var_bg,
            "weight_signal": gmm.weights_[self.signal_component_],
        }
        self.sample_ = sample  # kept for bic/log-likelihood in score()
        return self

    def transform(self, X: ImageLike) -> ImageLike:
        flat = np.asarray(_values(X)).flatten()
        probs = self.gmm_.predict_proba(flat.reshape(-1, 1))
        # gmm components were reordered in-place during fit, so columns already
        # align with signal_component_/background_component_
        m = self.moments_
        expected = (m["mean_signal"] - m["mean_background"]) + (1 - self.rho_) * (
            flat - m["mean_signal"]
        )
        adjusted = (expected * probs[:, self.signal_component_]).clip(min=0)

        self.labels_ = np.argmax(probs, axis=1).reshape(X.shape).astype(np.int32)
        self.probabilities_ = probs.reshape(X.shape + (probs.shape[1],)).astype(np.float32)

        # mainly for PostHocQuantile if used
        attrs = {
            **m,
            "rho": float(self.rho_),
            "signal_component": int(self.signal_component_),
            "background_component": int(self.background_component_),
            "n_components": self.n_components,
            "means_sorted": self.gmm_.means_.flatten().tolist(),
            "vars_sorted": self.gmm_.covariances_[:, 0, 0].tolist(),
            "weights_sorted": self.gmm_.weights_.tolist(),
        }
        data = adjusted.astype(X.dtype).reshape(X.shape)
        return _rewrap(X, data, attrs)

    def score(self, X: xr.DataArray | None = None) -> ScoreMetrics:
        m = self.moments_
        pooled_sd = np.sqrt((m["var_signal"] + m["var_background"]) / 2)
        cohens_d = (m["mean_signal"] - m["mean_background"]) / pooled_sd
        s = self.sample_.reshape(-1, 1)
        return {
            "cohens_d": float(cohens_d),
            "bic": float(self.gmm_.bic(s)),
            "log_likelihood": float(self.gmm_.score(s)),
            "signal_weight": float(m["weight_signal"]),
        }

class PostHocQuantile(BaseEstimator, TransformerMixin):
    """Normalise by the bgnorm-adjusted value of the top component's q-quantile.

    Reads the mean-sorted component params from the upstream bgnorm step's
    .attrs, so it stays a droppable pipeline step.
    """

    def __init__(self, q: float = 0.75) -> None:
        self.q = q

    def fit(self, X: xr.DataArray, y: None = None) -> PostHocQuantile:
        if not 0 < self.q < 1:
            raise ValueError("q must be in (0, 1)")
        return self

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        if not isinstance(X, xr.DataArray):
            raise TypeError(
                "PostHocQuantile reads GMM moments from DataArray .attrs, which a "
                "numpy/dask array can't carry between pipeline steps. Wrap the input "
                "as an xr.DataArray, or drop this step and rescale manually using the "
                "fitted BgNormChannel's .moments_ / .gmm_."
            )
        means = np.asarray(X.attrs["means_sorted"])
        varis = np.asarray(X.attrs["vars_sorted"])
        weights = np.asarray(X.attrs["weights_sorted"])
        mu3, mu2 = means[-1], means[-2]
        var3, var2 = varis[-1], varis[-2]

        # reference intensity at the q-quantile of the top component
        # qnorm = ppf (quantil function/percent point function)
        q3 = stats.norm.ppf(self.q, loc=mu3, scale=np.sqrt(var3))
        # posterior probability of the top component at q3 (across all components)
        # dens is vector for [p1, p2, p3]
        dens = weights * stats.norm.pdf(q3, loc=means, scale=np.sqrt(varis))
        p_total = dens.sum()
        p3 = dens[-1] / p_total
        # not sure if we need p1, p2
        var_factor = (var3 - min(var2, var3)) / var3
        adjq = p3 * ((mu3 - mu2) + (q3 - mu2) * var_factor)

        out = X.copy(data=X.values / adjq)
        out.attrs.update(X.attrs)
        out.attrs["posthoc_norm_factor"] = float(adjq)
        out.attrs["description"] = "BgNorm adjusted, post-hoc quantile normalised"
        return out

def default_pipeline(config: BgNormConfig | None = None, **overrides: object) -> Pipeline:
    """Standard bgnorm composition from a BgNormConfig dataclass."""
    cfg = _resolve_config(config, overrides)
    steps: list[tuple[str, BaseEstimator]] = [
        ("median", MedianFilter(cfg.median_filter_radius)),
        ("log2", Log2Transform(cfg.image_cofactor)),
        ("bgnorm", BgNormChannel(cfg.n_components, cfg.n_pixels_to_sample, cfg.pixel_sampling_seed)),
    ]
    if cfg.quantile_post_hoc_value is not None:
        steps.append(("posthoc", PostHocQuantile(cfg.quantile_post_hoc_value)))
    return Pipeline(steps)


def _coerce(image: ImageLike, channel_dim: str) -> tuple[xr.DataArray, bool, bool]:
    """Normalise any supported input to a (channel_dim, y, x) DataArray.

    Accepts xr.DataArray (with or without a channel dim) or an array-like
    (numpy/dask) of shape (y, x) or (c, y, x). Returns
    (DataArray, was_xarray, was_single_channel) so the caller can return output
    in the same flavour/rank the user gave.
    """
    if isinstance(image, xr.DataArray):
        if channel_dim in image.dims:
            return image, True, False
        return image.expand_dims(channel_dim), True, True  # 2D DataArray

    arr = image  # numpy or dask array
    if arr.ndim == 2:
        arr, single = arr[None], True # (y, x) -> (1, y, x); need c dim
    elif arr.ndim == 3:
        single = arr.shape[0] == 1
    else:
        raise ValueError(f"Expected 2D (y, x) or 3D (c, y, x) array, got ndim={arr.ndim}")
    da_img = xr.DataArray(
        arr,
        dims=(channel_dim, "y", "x"),
        coords={channel_dim: np.arange(arr.shape[0])},
    )
    return da_img, False, single


@dataclass
class BgNormArrays:
    """Plain-array result returned when the caller passes a numpy/dask image."""
    adjusted: np.ndarray
    labels: np.ndarray
    probs: np.ndarray
    summaries: pd.DataFrame

# function entrypoint
def bgnorm(
    image: ImageLike,
    pipeline: Pipeline | None = None,
    channel_dim: str = "c",
    config: BgNormConfig | None = None,
    **overrides: object,
) -> tuple[xr.Dataset, pd.DataFrame] | BgNormArrays:
    """Background-normalise a single- or multi-channel image.

    `image` may be an xr.DataArray ((c, y, x) or (y, x)) or a numpy/dask array
    of shape (y, x) or (c, y, x). Pass a custom `pipeline=` to run an experiment
    composition; otherwise a `default_pipeline` is built from `config=` / keyword
    overrides (validated via BgNormConfig).

    Returns (xr.Dataset, summaries) for DataArray input, or a BgNormArrays for
    array input (rank matched to what was passed in).
    """
    da_img, was_xarray, single = _coerce(image, channel_dim)
    cfg = _resolve_config(config, overrides)
    make_pipeline: PipelineFactory = (
        (lambda: pipeline) if pipeline is not None
        else (lambda: default_pipeline(cfg))
    )
    merged, summaries = _run_per_channel(da_img, make_pipeline, channel_dim)
    summaries = pd.DataFrame.from_dict(summaries, orient="index")

    if was_xarray:
        return (merged.isel({channel_dim: 0}) if single else merged), summaries

    adjusted = np.asarray(merged["adjusted_image"].data)
    labels = np.asarray(merged["labels"].data)
    probs = np.asarray(merged["probs"].data)
    if single:  # caller gave (y, x) -> hand back without the channel axis
        adjusted, labels, probs = adjusted[0], labels[0], probs[0]
    return BgNormArrays(adjusted, labels, probs, summaries)


def _run_per_channel(
    image: xr.DataArray,
    make_pipeline: PipelineFactory,
    channel_dim: str,
) -> tuple[xr.Dataset, dict[object, dict[str, float]]]:
    """Fit one (cloned) pipeline per channel and reassemble. This is the seam
    for dispatching C parallel jobs + opening/closing nested MLflow runs."""
    datasets, summaries = [], {}
    for ci, ch_name in enumerate(image[channel_dim].values):
        ch = image.isel({channel_dim: ci}) # (y, x) slice, retains scalar `c` coord
        ydim, xdim = ch.dims

        pipe = clone(make_pipeline())
        adjusted = pipe.fit_transform(ch) # DataArray (post-hoc applied if in pipe)
        bg = pipe.named_steps["bgnorm"]

        summaries[ch_name] = {**bg.moments_, **bg.score()}

        # probabilities_: (y, x, n_components) -> (n_components, y, x); carry the
        # channel scalar coord + spatial coords so xr.concat aligns per channel.
        probs = xr.DataArray(
            bg.probabilities_.transpose(2, 0, 1),
            dims=(f"{channel_dim}_probs", ydim, xdim),
            coords={
                ydim: ch.coords[ydim],
                xdim: ch.coords[xdim],
                channel_dim: ch.coords[channel_dim],
            },
        )
        datasets.append(
            xr.Dataset(
                {
                    "adjusted_image": adjusted,
                    "labels": ch.copy(data=bg.labels_),
                    "probs": probs,
                }
            )
        )

    merged = xr.concat(datasets, dim=channel_dim)
    return merged, summaries