from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, TypedDict

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

from .io import ImageLikeSchema, as_image_matrix
from .tracking import MLflowTracker, TrackingConfig

# type helpers
ImageLike = xr.DataArray | np.ndarray | da.Array
PipelineFactory = Callable[[], Pipeline]


class _Unset:
    """Sentinel for bgnorm() keyword params: distinguishes 'not passed' from an
    explicit value, so a base `config=` is only overridden by params the caller
    actually sets (defaults still come from BgNormConfig)."""
    def __repr__(self) -> str:
        return "DEFAULT"


_UNSET: Any = _Unset()


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
        0.75, gt=0, lt=1, description="If set, append PostHocQuantile(q); else omit it."
    )
    compute_bic_model_order: bool = Field(
        False, description="If True, also compute per-channel BIC model-order gains "
        "(k=1..n_components, reusing the n_components fit) during fit."
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
    """Per-channel goodness-of-fit + method-validity metrics.

    When `compute_bic_model_order=True`, score() also adds dynamic
    `bic_gain_{n_components}v{k}` keys (the per-point BIC gain of K=n_components over
    each smaller k), e.g. `bic_gain_3v1`, `bic_gain_3v2`.
    """
    cohens_d: float
    jsd: float           
    bic: float
    log_likelihood: float
    signal_weight: float          
    mean_gap: float           
    rho: float                
    conv_violated: float      
    signal_var_fraction: float 
    adjustment_scale: float  


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


def _gaussian_grid(
    mu1: float, var1: float, mu2: float, var2: float, n_grid: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integration grid + the two component pdfs"""
    sd1, sd2 = np.sqrt(var1), np.sqrt(var2)
    lo = min(mu1 - 6 * sd1, mu2 - 6 * sd2)
    hi = max(mu1 + 6 * sd1, mu2 + 6 * sd2)
    x = np.linspace(lo, hi, n_grid)
    return x, stats.norm.pdf(x, mu1, sd1), stats.norm.pdf(x, mu2, sd2)


def _jsd_from_pdfs(
    x: np.ndarray, p: np.ndarray, q: np.ndarray, w1: float = 0.5, w2: float = 0.5,
) -> float:
    """Jensen-Shannon divergence (base 2) of two pdfs sampled on grid `x`, mixed
    with weights (w1, w2). Equal weights for symmetric JSD in [0, 1]; 
    """
    w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
    mix = w1 * p + w2 * q

    def _kl(a: np.ndarray) -> float:
        terms = np.zeros_like(a)
        mask = a > 0
        terms[mask] = a[mask] * np.log2(a[mask] / mix[mask])
        return float(np.trapezoid(terms, x))

    return w1 * _kl(p) + w2 * _kl(q)


# def _gaussian_jsd(
#     mu1: float, var1: float, mu2: float, var2: float,
#     w1: float = 0.5, w2: float = 0.5, n_grid: int = 2000,
# ) -> float:
#     """Jensen-Shannon divergence (base 2) between two 1-D Gaussians (no closed form,
#     so integrated on a shared grid)."""
#     x, p, q = _gaussian_grid(mu1, var1, mu2, var2, n_grid)
#     return _jsd_from_pdfs(x, p, q, w1, w2)


def _bic_fitness_comparison(
    sample: np.ndarray,
    kmax: int,
    kmax_gmm: GaussianMixture,
    random_state: int = 42,
) -> dict[str, object]:
    """Per-data-point BIC model order over k = 1 .. kmax for a 1-D sample.

    Reuses the already-fitted `kmax_gmm` (the channel's n_components fit) for the kmax
    term — so the kmax model is NOT refit — and fits only k = 1 .. kmax-1. Returns the
    negated, per-point BIC (``nbic[k] = -GMM.bic(sample) / n``; larger = better) and
    the gain of kmax over each smaller k (``gains[k] = nbic[kmax] - nbic[k]``).

    A near-zero gain of kmax over K=1 flags a channel that doesn't support kmax
    components (a flat/dead channel), useful QC for the assumed component count.

    Returns ``{"n": int, "nbic": {k: float}, "gains": {k: float}}``.
    """
    s = np.asarray(sample, dtype=float).reshape(-1, 1)
    n = s.shape[0]
    kmax = int(kmax)
    nbic = {kmax: -float(kmax_gmm.bic(s)) / n}  # reuse the fitted kmax GMM (no refit)
    for k in range(1, kmax):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full",
            init_params="k-means++", random_state=random_state,
        )
        gmm.fit(s)
        nbic[k] = -float(gmm.bic(s)) / n  # negated + per-point: larger = better fit
    gains = {k: nbic[kmax] - nbic[k] for k in range(1, kmax)}
    return {"n": n, "nbic": nbic, "gains": gains}


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

class Log10Transform(BaseEstimator, TransformerMixin):
    """log10(X / cofactor + 1) scalar transform"""

    def __init__(self, cofactor: int = 150) -> None:
        self.cofactor = cofactor

    def fit(self, X: ImageLike, y: None = None) -> Log10Transform:
        if self.cofactor < 1:
            raise ValueError("Cofactor must be 1 or greater")
        return self

    def transform(self, X: ImageLike) -> ImageLike:
        return _rewrap(X, da.log10(_as_dask(X) / self.cofactor + 1))

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
    .score(X); goodness-of-fit + method-validity metrics
        {cohens_d, jsd, bic, log_likelihood, signal_weight,
         mean_gap, rho, conv_violated, signal_var_fraction, adjustment_scale}.
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
    bic_model_order_: dict  # set in fit() when compute_bic_model_order=True

    def __init__(
        self,
        n_components: int = 3, # NOTE: this might become static variable sicne bgnorm core assumption is ncomp=3
        n_pixels_to_sample: int = int(1e5),
        pixel_sampling_seed: int = 0,
        compute_bic_model_order: bool = False,
    ) -> None:
        self.n_components = n_components
        self.n_pixels_to_sample = n_pixels_to_sample
        self.pixel_sampling_seed = pixel_sampling_seed
        # if True, fit() also computes the BIC model-order gains over k=1..n_components,
        # reusing the n_components GMM for kmax (see bic_model_order()).
        self.compute_bic_model_order = compute_bic_model_order

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

        gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type="full",
            init_params="k-means++",
            random_state=42, 
        )
        gmm.fit(sample.reshape(-1, 1))

        # reorder components by mean (ascending). 
        order = np.argsort(gmm.means_.flatten())
        gmm.means_ = gmm.means_[order]
        gmm.covariances_ = gmm.covariances_[order]
        gmm.weights_ = gmm.weights_[order]
        gmm.precisions_cholesky_ = gmm.precisions_cholesky_[order]

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
        if self.compute_bic_model_order:
            # reuse the n_components GMM just fitted (gmm) for kmax — no refit
            self.bic_model_order_ = _bic_fitness_comparison(sample, self.n_components, gmm)
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

        # can be compressed to int8; 0 to 255 components
        self.labels_ = np.argmax(probs, axis=1).reshape(X.shape).astype(np.uint8)
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
        mean_gap = float(m["mean_signal"] - m["mean_background"])
        cohens_d = mean_gap / pooled_sd

        x, p, q = _gaussian_grid(
            m["mean_signal"], m["var_signal"], m["mean_background"], m["var_background"]
        )
        jsd = _jsd_from_pdfs(x, p, q)
        s = self.sample_.reshape(-1, 1)


        rho = float(self.rho_)
        # bic_gain_{kmax}v{k} = nbic[kmax] - nbic[k], kmax = n_components.
        bic_gains = {}
        if self.compute_bic_model_order:
            kmax = self.n_components
            bic_gains = {
                f"bic_gain_{kmax}v{k}": float(g)
                for k, g in self.bic_model_order()["gains"].items()
            }
        return {
            "cohens_d": float(cohens_d),
            "jsd": float(jsd),
            "bic": float(self.gmm_.bic(s)),
            "log_likelihood": float(self.gmm_.score(s)),
            "signal_weight": float(m["weight_signal"]),
            "mean_gap": mean_gap,
            "rho": rho,
            "conv_violated": float(rho < 0),
            "signal_var_fraction": float(1.0 - rho),
            "adjustment_scale": mean_gap * float(m["weight_signal"]),
            **bic_gains,
        }

    def bic_model_order(self) -> dict[str, object]:
        """Per-point BIC model order over k = 1 .. n_components, reusing the fitted
        n_components GMM for kmax (so kmax is never refit). Cached on
        `bic_model_order_`; computed eagerly in fit() when
        `compute_bic_model_order=True`, otherwise on first call here. A near-zero gain
        of K=n_components over K=1 flags a channel that doesn't support the assumed
        component count (e.g. a flat/dead channel)."""
        if getattr(self, "bic_model_order_", None) is None:
            self.bic_model_order_ = _bic_fitness_comparison(self.sample_, self.n_components, self.gmm_)
        return self.bic_model_order_

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
        ("bgnorm", BgNormChannel(
            cfg.n_components, cfg.n_pixels_to_sample, cfg.pixel_sampling_seed,
            compute_bic_model_order=cfg.compute_bic_model_order,
        )),
    ]
    if cfg.quantile_post_hoc_value is not None:
        steps.append(("posthoc", PostHocQuantile(cfg.quantile_post_hoc_value)))
    return Pipeline(steps)


@dataclass
class BgNormArrays:
    """Plain-array result returned when the caller passes a numpy/dask image.

    `adjusted` is the bgnorm output; `adjusted_post_hoc` is the quantile-normalised
    output (None unless a PostHocQuantile step ran).
    """
    adjusted: np.ndarray
    labels: np.ndarray
    probs: np.ndarray
    summaries: pd.DataFrame
    adjusted_post_hoc: np.ndarray | None = None

def _build_tracker(
    tracking: TrackingConfig,
    image: xr.DataArray,
    pipeline: Pipeline,
) -> MLflowTracker:
    """Resolve the experiment image name (explicit > DataArray.name > "image")
    and construct the MLflow tracker. `pipeline` is a representative (unfitted)
    composition; the tracker logs its steps/params as the parent parameter setting."""
    name = tracking.image_name
    if name is None and isinstance(image, xr.DataArray) and image.name is not None:
        name = str(image.name)
    return MLflowTracker(tracking, image_name=name or "image", pipeline=pipeline)


def bgnorm(
    image: ImageLike,
    pipeline: Pipeline | None = None,
    channel_dim: str = "c",
    config: BgNormConfig | None = None,
    *,
    median_filter_radius: int = _UNSET,
    image_cofactor: int = _UNSET,
    n_components: int = _UNSET,
    n_pixels_to_sample: int = _UNSET,
    pixel_sampling_seed: int = _UNSET,
    quantile_post_hoc_value: float | None = _UNSET,
    compute_bic_model_order: bool = _UNSET,
    tracking: TrackingConfig | None = None,
) -> tuple[xr.Dataset, pd.DataFrame] | BgNormArrays:
    """Background-normalise a single- or multi-channel image.

    `image` may be an xr.DataArray ((c, y, x) or (y, x)) or a numpy/dask array of
    shape (y, x) or (c, y, x); it is parsed/validated via `io.as_image_matrix`.

    The default pipeline is built from `config=` (a BgNormConfig) overlaid with any
    of the keyword params below that you pass explicitly — they mirror BgNormConfig,
    so they autocomplete and show up in help(). Pass a custom `pipeline=` to run your
    own composition (config / keyword params are then ignored).

    Parameters
    ----------
    image : xr.DataArray | np.ndarray | dask.array
        (c, y, x) or (y, x) input.
    pipeline : sklearn Pipeline, optional
        Custom composition; overrides the config-built default_pipeline.
    channel_dim : str
        Name of the channel axis (default "c").
    config : BgNormConfig, optional
        Base parameter set; any explicit keyword below overrides it.
    median_filter_radius : int
        Square median-filter radius, px (default 3).
    image_cofactor : int
        Divisor in log2(x / cofactor + 1) (default 150).
    n_components : int
        GMM components (default 3: background / tissue background / signal).
    n_pixels_to_sample : int
        Positive pixels sampled to fit the GMM (default 100000).
    pixel_sampling_seed : int
        Seed for the pixel sample (default 0).
    quantile_post_hoc_value : float | None
        If in (0, 1), append PostHocQuantile(q) (default 0.75); None disables it.
    compute_bic_model_order : bool
        Also compute per-channel BIC model-order gains during fit (default False).
    tracking : TrackingConfig, optional
        If given, log the run to MLflow (parent run per parameter setting, nested
        run per channel with ScoreMetrics + plots/grids).

    Returns
    -------
    (xr.Dataset, summaries) for DataArray input, or a BgNormArrays for array input
    (rank matched to the input).
    """
    # only the params the caller actually set become overrides on the base config
    overrides = {
        k: v for k, v in {
            "median_filter_radius": median_filter_radius,
            "image_cofactor": image_cofactor,
            "n_components": n_components,
            "n_pixels_to_sample": n_pixels_to_sample,
            "pixel_sampling_seed": pixel_sampling_seed,
            "quantile_post_hoc_value": quantile_post_hoc_value,
            "compute_bic_model_order": compute_bic_model_order,
        }.items() if v is not _UNSET
    }
    was_xarray = isinstance(image, xr.DataArray)
    name = tracking.image_name if tracking is not None else None
    da_img = as_image_matrix(image, channel_dim=channel_dim, name=name)
    single = int(da_img.sizes[channel_dim]) == 1
    cfg = _resolve_config(config, overrides)
    make_pipeline: PipelineFactory = (
        (lambda: pipeline) if pipeline is not None
        else (lambda: default_pipeline(cfg))
    )
    # Probe the composition once (unfitted) so the tracker can record which
    # transforms run; per-channel pipelines are still cloned fresh below.
    tracker = (
        _build_tracker(tracking, da_img, make_pipeline())
        if tracking is not None
        else None
    )
    with (tracker or nullcontext()):
        merged, summaries, transformed = _run_per_channel(
            da_img, make_pipeline, channel_dim, tracker
        )
        summaries = pd.DataFrame.from_dict(summaries, orient="index")
        if tracker is not None:
            tracker.log_summary(summaries)
            tracker.log_grids(da_img, transformed, merged, channel_dim)

    if was_xarray:
        return (merged.isel({channel_dim: 0}) if single else merged), summaries

    adjusted = np.asarray(merged["adjusted_image"].data)
    labels = np.asarray(merged["labels"].data)
    probs = np.asarray(merged["probs"].data)
    post_hoc = (
        np.asarray(merged["adjusted_image_post_hoc"].data)
        if "adjusted_image_post_hoc" in merged else None
    )
    if single:  # caller gave (y, x) -> hand back without the channel axis
        adjusted, labels, probs = adjusted[0], labels[0], probs[0]
        if post_hoc is not None:
            post_hoc = post_hoc[0]
    return BgNormArrays(adjusted, labels, probs, summaries, post_hoc)


def _pre_bgnorm_transform(pipe: Pipeline, bgnorm_step: BgNormChannel, X: xr.DataArray) -> xr.DataArray:
    """Replay the fitted pre-bgnorm transformers (e.g. median, log2) on X — i.e.
    the channel as the GMM actually saw it."""
    out = X
    for _, est in pipe.steps:
        if est is bgnorm_step:
            break
        out = est.transform(out)
    return out


def _run_per_channel(
    image: xr.DataArray,
    make_pipeline: PipelineFactory,
    channel_dim: str,
    tracker: MLflowTracker | None = None,
) -> tuple[xr.Dataset, dict[object, dict[str, float]], xr.DataArray | None]:
    """Fit one cloned pipeline per channel and reassemble. 
    Pipeline cloned for parallel dispatch (TODO)

    If mlflow tracker is provided, also collects the per-channel transformed image
    (post median+log2, pre bgnorm) into a (c, y, x) DataArray for the image grids;
    returns None otherwise"""
    # enforce the canonical (channel_dim, y, x) image matrix at the compute seam
    image = ImageLikeSchema.validate(image, channel_dim=channel_dim)
    datasets, summaries, transformed = [], {}, []
    signal_comps, background_comps = [], []  # per-channel role assignment (mean-sorted idx)
    for ci, ch_name in enumerate(image[channel_dim].values):
        ch = image.isel({channel_dim: ci}) # (y, x) slice, retains scalar `c` coord
        ydim, xdim = ch.dims

        with (tracker.channel(ch_name) if tracker is not None else nullcontext()):
            pipe = clone(make_pipeline())
            adjusted_final = pipe.fit_transform(ch)  # bgnorm, + posthoc quantile if in pipe
            bg = pipe.named_steps["bgnorm"]

            metrics = {**bg.moments_, **bg.score()}
            summaries[ch_name] = metrics
            signal_comps.append(int(bg.signal_component_))
            background_comps.append(int(bg.background_component_))

            has_posthoc = any(isinstance(est, PostHocQuantile) for _, est in pipe.steps)
            # the GMM input (median+log): needed for the grids, and to recover the
            # pre-quantile bgnorm image when a posthoc step is present.
            xformed = (
                _pre_bgnorm_transform(pipe, bg, ch)
                if (tracker is not None or has_posthoc) else None
            )
            if has_posthoc:
                adjusted_image = bg.transform(xformed)  # bgnorm (pre-quantile)
                adjusted_post_hoc = adjusted_final # bgnormQ
            else:
                adjusted_image = adjusted_final
                adjusted_post_hoc = None

            if tracker is not None:
                tracker.log_channel(
                    ch_name, ch, xformed, adjusted_image, adjusted_post_hoc, bg, metrics, pipe
                )
                transformed.append(xformed)

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
        data_vars = {
            "adjusted_image": adjusted_image,
            "labels": ch.copy(data=bg.labels_),
            "probs": probs,
        }
        if adjusted_post_hoc is not None:
            data_vars["adjusted_image_post_hoc"] = adjusted_post_hoc
        datasets.append(xr.Dataset(data_vars))

    merged = xr.concat(datasets, dim=channel_dim)
    # which mean-sorted GMM component is signal vs (tissue) background per channel
    # the assignment can flip per channel, so carry it for role-correct plotting.
    merged = merged.assign_coords(
        signal_component=(channel_dim, np.array(signal_comps, dtype=int)),
        background_component=(channel_dim, np.array(background_comps, dtype=int)),
    )
    transformed_da = xr.concat(transformed, dim=channel_dim) if transformed else None
    return merged, summaries, transformed_da