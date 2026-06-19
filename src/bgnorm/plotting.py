"""Per-run plot artifacts for the bgnorm MLflow integration.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import pandas as pd
    import xarray as xr
    from matplotlib.figure import Figure
    from sklearn.pipeline import Pipeline

    from .core import BgNormChannel


@dataclass
class ChannelPlotContext:
    """Everything produced for one channel, handed to each channel plot."""

    channel: str
    image: xr.DataArray
    transformed: xr.DataArray # post pre-bgnorm transforms, the GMM's input
    adjusted: xr.DataArray # bgnorm output (pre-quantile)
    model: BgNormChannel
    metrics: dict[str, float]
    pipeline: Pipeline
    adjusted_q: xr.DataArray | None = None # bgnormQ output (None without posthoc)
    transform_label: str = "transformed" # the actual pre-bgnorm transforms, e.g. "median + log10"


@dataclass
class ImageGridContext:
    """All-channel stacks for the parent-run image grids. Each image field is a
    (channel_dim, y, x) DataArray; `probs` is (..., channel_dim, ..., y, x) with a
    `{channel_dim}_probs` axis of length `n_components`."""

    raw: xr.DataArray # raw, untransformed input, (c, y, x)
    transformed: xr.DataArray | None # post pre-bgnorm transforms (None if not collected)
    bgnorm: xr.DataArray # bgnorm output (pre-quantile), (c, y, x)
    bgnorm_q: xr.DataArray | None # bgnormQ output (None without posthoc)
    labels: xr.DataArray # argmax mean-sorted component id, (c, y, x)
    probs: xr.DataArray # per-component posteriors, has a {c}_probs axis
    signal_components: object  # (c,) int array
    background_components: object  # (c,) int array
    channel_dim: str
    n_components: int
    max_px: int = 128 # per-channel thumbnail longest edge
    transform_label: str = "transformed" # the actual pre-bgnorm transforms, e.g. "median + log10"


ChannelPlot = Callable[["ChannelPlotContext"], "Figure"]
SummaryPlot = Callable[["pd.DataFrame"], "Figure"]
GridPlot = Callable[["ImageGridContext"], "Figure"]


ROLE_BACKGROUND, ROLE_TISSUE, ROLE_SIGNAL = 0, 1, 2
CLASS_ROLES = {ROLE_BACKGROUND: "background", ROLE_TISSUE: "tissue background", ROLE_SIGNAL: "signal"}
CLASS_COLORS = ["grey", "saddlebrown", "royalblue"]


def _plt():
    """matplotlib.pyplot on a headless (Agg) backend; imported lazily."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt

def gmm_fit(ctx: ChannelPlotContext) -> Figure:
    """Histogram of the sampled pixels with the fitted GMM components overlaid —
    reads the core.py `BgNormChannel` (its `.sample_` and `.gmm_`)."""
    import numpy as np
    from scipy import stats

    plt = _plt()
    bg = ctx.model
    sample = np.asarray(bg.sample_).ravel()
    means = bg.gmm_.means_.ravel()
    varis = bg.gmm_.covariances_[:, 0, 0]
    weights = bg.gmm_.weights_

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(sample, bins=100, density=True, color="0.85", label="sampled pixels")
    xs = np.linspace(sample.min(), sample.max(), 512)
    mixture = np.zeros_like(xs)
    for k, (m, v, w) in enumerate(zip(means, varis, weights)):
        comp = w * stats.norm.pdf(xs, m, np.sqrt(v))
        mixture += comp
        if k == bg.signal_component_:
            role = "signal"
        elif k == bg.background_component_:
            role = "background"
        else:
            role = f"comp{k}"
        ax.plot(xs, comp, label=f"{role} (w={w:.2f})")
    ax.plot(xs, mixture, "k--", lw=1, label="mixture")
    ax.set(title=f"{ctx.channel}: GMM fit", xlabel="transformed intensity", ylabel="density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def transformed_vs_adjusted(ctx: ChannelPlotContext) -> Figure:
    """Fair before/after for one channel: the GMM input (median+log,
    `ctx.transformed`) vs the bgnorm output, and — when a posthoc quantile step ran
    — the bgnormQ output too. bgnorm works in the transformed space, so comparing
    against the raw image would overstate the change."""
    import numpy as np

    plt = _plt()
    panels = [(ctx.transform_label, ctx.transformed), ("bgnorm", ctx.adjusted)]
    if ctx.adjusted_q is not None:
        panels.append(("bgnormQ", ctx.adjusted_q))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, da) in zip(axes, panels):
        im = ax.imshow(np.asarray(da.values), cmap="magma")
        ax.set_title(f"{ctx.channel}: {title}")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig

def _grid_shape(n: int) -> tuple[int, int]:
    import math

    ncols = math.ceil(math.sqrt(n))
    return math.ceil(n / ncols), ncols


def _montage(stack, channel_dim, names, *, title, cmap, norm=None, vmin=None,
             vmax=None, max_px=128, colorbar=True):
    """Lay every channel of `stack` (a (c, y, x) DataArray) out as a downsampled
    grid on a single shared colour scale, with one shared colorbar. When `vmin`/
    `vmax`/`norm` aren't given the scale is the robust 1-99th percentile across all
    channels (so one channel's outliers don't blow out the rest)."""
    import numpy as np

    plt = _plt()
    spatial = [d for d in stack.dims if d != channel_dim]
    sy = max(1, stack.sizes[spatial[0]] // max_px)
    sx = max(1, stack.sizes[spatial[1]] // max_px)
    thumbs = stack.isel({spatial[0]: slice(None, None, sy), spatial[1]: slice(None, None, sx)})
    arr = np.asarray(thumbs.values)  # (c, y', x') — downsampled, cheap to materialise once

    if norm is None and vmin is None and vmax is None:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vmin, vmax = float(np.percentile(finite, 1)), float(np.percentile(finite, 99))

    n = stack.sizes[channel_dim]
    nrows, ncols = _grid_shape(n)
    # constrained layout manages spacing for the shared colorbar + suptitle cleanly
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.7),
                             squeeze=False, layout="constrained")
    im = None
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i >= n:
            continue
        im = ax.imshow(arr[i], cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(str(names[i]), fontsize=5)
    fig.suptitle(title, fontsize=11)
    if colorbar and im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01)
    return fig, fig.axes


def _names(ctx: ImageGridContext) -> list:
    return list(ctx.raw.coords[ctx.channel_dim].values)


def _role_labels(ctx: ImageGridContext):
    """Remap each channel's mean-sorted component labels into role space
    (0 background, 1 tissue background, 2 signal), using that channel's own
    signal/background component assignment."""
    import numpy as np
    import xarray as xr

    cd = ctx.channel_dim
    sig = xr.DataArray(np.asarray(ctx.signal_components), dims=cd)
    bg = xr.DataArray(np.asarray(ctx.background_components), dims=cd)
    return xr.where(
        ctx.labels == sig, ROLE_SIGNAL, xr.where(ctx.labels == bg, ROLE_TISSUE, ROLE_BACKGROUND)
    )


def _role_prob(ctx: ImageGridContext, role: int):
    """Per-channel posterior for a role, in [0, 1]. Signal/tissue use that channel's
    assigned component column; background is the remaining mass (1 - signal - tissue)."""
    import numpy as np
    import xarray as xr

    cd = ctx.channel_dim
    pk = f"{cd}_probs"
    p_sig = ctx.probs.isel({pk: xr.DataArray(np.asarray(ctx.signal_components), dims=cd)})
    p_bg = ctx.probs.isel({pk: xr.DataArray(np.asarray(ctx.background_components), dims=cd)})
    if role == ROLE_SIGNAL:
        return p_sig
    if role == ROLE_TISSUE:
        return p_bg
    return (1.0 - p_sig - p_bg).clip(min=0.0)


def raw_grid(ctx: ImageGridContext) -> Figure:
    """Raw, untransformed input images, one cell per channel (shared scale + cbar)."""
    fig, _ = _montage(ctx.raw, ctx.channel_dim, _names(ctx),
                      title="raw (untransformed input)", cmap="magma", max_px=ctx.max_px)
    return fig


def transformed_grid(ctx: ImageGridContext) -> Figure:
    """The image the GMM was fit on: the actual pre-bgnorm transforms applied
    (e.g. median + log10), labelled from the pipeline composition."""
    fig, _ = _montage(ctx.transformed, ctx.channel_dim, _names(ctx),
                      title=ctx.transform_label, cmap="magma", max_px=ctx.max_px)
    return fig


def bgnorm_grid(ctx: ImageGridContext) -> Figure:
    """bgnorm-adjusted images (pre quantile), one cell per channel."""
    fig, _ = _montage(ctx.bgnorm, ctx.channel_dim, _names(ctx),
                      title="bgnorm", cmap="magma", max_px=ctx.max_px)
    return fig


def bgnormq_grid(ctx: ImageGridContext) -> Figure:
    """Quantile-normalised bgnorm images (bgnormQ); only when a posthoc step ran."""
    fig, _ = _montage(ctx.bgnorm_q, ctx.channel_dim, _names(ctx),
                      title="bgnormQ", cmap="magma", max_px=ctx.max_px)
    return fig


def labels_grid(ctx: ImageGridContext) -> Figure:
    """Predicted roles as a semantic mask, remapped per channel: background (grey),
    tissue background (brown), signal (blue). Discrete role legend, no colorbar."""
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS) + 0.5), cmap.N)
    fig, _ = _montage(_role_labels(ctx), ctx.channel_dim, _names(ctx), title="predicted roles",
                      cmap=cmap, norm=norm, max_px=ctx.max_px, colorbar=False)
    handles = [Patch(color=CLASS_COLORS[k], label=f"{k}: {CLASS_ROLES[k]}")
               for k in (ROLE_BACKGROUND, ROLE_TISSUE, ROLE_SIGNAL)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    return fig


def prob_grid(ctx: ImageGridContext, role: int) -> Figure:
    """Posterior probability of a role (signal / tissue background / background)
    across channels, on a fixed [0, 1] scale. The component is selected per channel
    from that channel's signal/background assignment."""
    fig, _ = _montage(_role_prob(ctx, role), ctx.channel_dim, _names(ctx),
                      title=f"P({CLASS_ROLES[role]})", cmap="viridis",
                      vmin=0.0, vmax=1.0, max_px=ctx.max_px)
    return fig


CHANNEL_PLOTS: dict[str, ChannelPlot] = {
    "gmm_fit": gmm_fit,
    "transformed_vs_adjusted": transformed_vs_adjusted,
}

SUMMARY_PLOTS: dict[str, SummaryPlot] = {}

GRID_PLOTS: dict[str, GridPlot] = {
    "raw": raw_grid,
    "transformed": transformed_grid,
    "bgnorm": bgnorm_grid,
    "labels": labels_grid,
}


def _fig_to_rgb(fig):
    """Rasterise an Agg figure to an (H, W, 3) uint8 array for `mlflow.log_image`."""
    import numpy as np

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def _log_figures(mlflow, items: dict, arg, kind: str, subdir: str = "plots") -> None:
    """Render each registered plot and log it as a keyed image under `{subdir}/{stem}`.

    Uses `mlflow.log_image(..., key=...)` rather than `log_figure` so the result is a
    *logged image* (with step/timestamp metadata), which is what the MLflow UI needs
    to make it selectable in the Image Grid chart — plain artifact PNGs from
    `log_figure` only show in the Artifacts browser (see mlflow/mlflow#15760). Falls
    back to `log_figure` on older MLflow that lacks keyed image logging. A failing
    plot is warned about and skipped so it never aborts the run."""
    plt = _plt()
    for stem, fn in items.items():
        try:
            fig = fn(arg)
        except Exception as exc:  # noqa: BLE001 - one bad plot shouldn't kill the run
            warnings.warn(f"bgnorm {kind} plot '{stem}' failed: {exc}", stacklevel=2)
            continue
        try:
            try:
                mlflow.log_image(_fig_to_rgb(fig), key=f"{subdir}/{stem}")
            except TypeError:  # MLflow < 2.12: no keyed image logging
                mlflow.log_figure(fig, f"{subdir}/{stem}.png")
        finally:
            plt.close(fig)


def log_channel_artifacts(
    mlflow, ctx: ChannelPlotContext, names: list[str] | None = None
) -> None:
    """Log every CHANNEL_PLOTS figure (or the `names` subset) for the active run."""
    items = CHANNEL_PLOTS if names is None else {
        k: CHANNEL_PLOTS[k] for k in names if k in CHANNEL_PLOTS
    }
    _log_figures(mlflow, items, ctx, kind="channel")


def log_summary_artifacts(
    mlflow, summaries: pd.DataFrame, names: list[str] | None = None
) -> None:
    """Log every SUMMARY_PLOTS figure (or the `names` subset) for the parent run."""
    items = SUMMARY_PLOTS if names is None else {
        k: SUMMARY_PLOTS[k] for k in names if k in SUMMARY_PLOTS
    }
    _log_figures(mlflow, items, summaries, kind="summary")


def log_grid_artifacts(mlflow, ctx: ImageGridContext) -> None:
    """Render and upload the all-channel image grids under `grids/` on the parent
    run: the fixed GRID_PLOTS (raw, transformed, bgnorm, labels), the bgnormQ grid
    when a posthoc quantile step ran, and one probability grid per role
    (signal / tissue background / background). The transformed grid is skipped
    when `ctx.transformed` is None."""
    items = {k: fn for k, fn in GRID_PLOTS.items() if not (k == "transformed" and ctx.transformed is None)}
    if ctx.bgnorm_q is not None:
        items["bgnormQ"] = bgnormq_grid
    # role-based probability grids (per-channel component selection); the background
    # role only exists when there's a component below the top-two (n_components > 2)
    roles = {"prob_signal": ROLE_SIGNAL, "prob_tissue_background": ROLE_TISSUE}
    if ctx.n_components > 2:
        roles["prob_background"] = ROLE_BACKGROUND
    items.update({stem: (lambda c, r=r: prob_grid(c, r)) for stem, r in roles.items()})
    _log_figures(mlflow, items, ctx, kind="grid", subdir="grids")
