"""Optional MLflow tracking integration for the bgnorm pipeline.

For now layout produced in the tracking server:
experiment  "bgnorm/{image_name}"
└── parent run : this represnets one "parameter setting" (a BgNormConfig)
    ├── nested run "DAPI"
    ├── nested run "CD4"
    └── ... (for C channels in the image)
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


class TrackingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_name: str | None = None
    """Name the experiment is built around. Falls back to the DataArray's `.name`,
    then to "image". Used as `bgnorm/{image_name}` unless `experiment_name` is set."""

    experiment_name: str | None = None
    """Override the whole experiment name. Default: `bgnorm/{image_name}`."""

    run_name: str | None = None
    """Parent run name. Default: a compact string derived from the BgNormConfig
    (the "parameter setting")."""

    tracking_uri: str | None = None
    """MLflow tracking server URI. If none, returns local mlruns/runs."""

    autolog: bool = False
    """Enable `mlflow.sklearn.autolog` so per-channel pipeline params (and, with
    `log_models`, the fitted estimator) are captured automatically. ScoreMetrics
    are logged by bgnorm regardless of this flag."""

    log_models: bool = False
    """Only meaningful with `autolog`: log the fitted sklearn pipeline as an
    artifact per channel.`"""

    log_plots: bool = True
    """Render and upload the plots registered in `bgnorm.plotting` as run
    artifacts: CHANNEL_PLOTS on each nested run, SUMMARY_PLOTS on the parent."""

    channel_plot_names: list[str] | None = None
    """Subset of `plotting.CHANNEL_PLOTS` to render (by stem). None = all."""

    summary_plot_names: list[str] | None = None
    """Subset of `plotting.SUMMARY_PLOTS` to render (by stem). None = all."""

    log_grids: bool = True
    """Upload the downsampled all-channel image grids (raw / transformed / adjusted
    / labels / per-class probabilities) as artifacts on the parent run."""

    grid_max_px: int = 128
    """Longest-edge size each channel thumbnail is downsampled to in the grids."""


# Salient param folded into the run-name tag, keyed by step class name so this
# module never has to import (and risk a cycle with) bgnorm.core. Steps not listed
# contribute just their pipeline name, so custom/omitted steps still read clearly.
_STEP_TAG = {
    "MedianFilter": lambda s: f"med{s.radius}",
    "Log2Transform": lambda s: f"log2c{s.cofactor}",
    "Log10Transform": lambda s: f"log10c{s.cofactor}",
    "ArcsinhTransform": lambda s: f"asinhc{s.cofactor}",
    "BgNormChannel": lambda s: f"bgk{s.n_components}s{s.pixel_sampling_seed}",
    "PostHocQuantile": lambda s: f"q{s.q}",
}


def _step_tag(name: str, step: object) -> str:
    fn = _STEP_TAG.get(type(step).__name__)
    return fn(step) if fn else name


# Human-readable label per pre-bgnorm transform class, for the image-grid titles.
_TRANSFORM_LABEL = {
    "MedianFilter": "median",
    "Log2Transform": "log2",
    "Log10Transform": "log10",
    "ArcsinhTransform": "arcsinh",
}


def grid_labels(pipeline: Pipeline) -> tuple[str, str]:
    """Derive the image-grid titles from the actual pipeline composition:
    (transform_label, adjusted_label) — e.g. ("median + log10", "bgnormQ").

    The transform label lists the transforms applied *before* bgnorm (what the GMM
    sees); the adjusted label is "bgnormQ" if a PostHocQuantile step runs, else
    "bgnorm"."""
    pre, seen_bgnorm, has_quantile = [], False, False
    for _, step in pipeline.steps:
        cls = type(step).__name__
        if cls == "BgNormChannel":
            seen_bgnorm = True
        elif cls == "PostHocQuantile":
            has_quantile = True
        elif not seen_bgnorm and cls in _TRANSFORM_LABEL:
            pre.append(_TRANSFORM_LABEL[cls])
    transform_label = " + ".join(pre) if pre else "transformed"
    return transform_label, ("bgnormQ" if has_quantile else "bgnorm")


def default_run_name(pipeline: Pipeline) -> str:
    """Compact name encoding the actual pipeline composition + each step's salient
    param, so runs that drop `median`/`log2` (or any step) stay visibly distinct.
    e.g. ``med3-log2c150-bgk3s0`` vs ``bgk3s0`` (transforms omitted)."""
    return "-".join(_step_tag(n, s) for n, s in pipeline.named_steps.items())


def pipeline_steps(pipeline: Pipeline) -> str:
    """Comma-joined step names — the composition, logged as a top-level param so the
    set of applied transforms is a first-class comparison axis in the MLflow UI."""
    return ",".join(pipeline.named_steps)


def _flat_params(pipe: Pipeline) -> dict[str, object]:
    """Scalar-valued pipeline params, safe to hand to `mlflow.log_params`. Params of
    an omitted step are simply absent (no `log2__cofactor` once `log2` is dropped)."""
    return {
        k: v
        for k, v in pipe.get_params(deep=True).items()
        if v is None or isinstance(v, (int, float, str, bool))
    }


class MLflowTracker:
    """Owns the experiment + parent run; hands out one nested run per channel.

    The pipeline is the source of truth for the "parameter setting": its
    composition + step params are logged on the parent run (so omitting `log2`/
    `median` is recorded faithfully), and the nested runs carry only per-channel
    metrics. Use as a context manager around the per-channel loop:

    with MLflowTracker(tcfg, image_name="core_R0_C0", pipeline=pipe) as tracker:
        for ch_name in channels:
            with tracker.channel(ch_name):
                ...fit...
                tracker.log_channel(metrics)
    """
    def __init__(
        self,
        tcfg: TrackingConfig,
        *,
        image_name: str,
        pipeline: Pipeline,
    ) -> None:
        import mlflow  # local: keeps mlflow an optional dependency

        self._mlflow = mlflow
        self.tcfg = tcfg
        self.image_name = image_name
        self.pipeline = pipeline
        self.experiment_name = tcfg.experiment_name or f"bgnorm/{image_name}"
        self.run_name = tcfg.run_name or default_run_name(pipeline)
        self.parent_run_id: str | None = None  # set when the parent run starts

        if tcfg.tracking_uri:
            mlflow.set_tracking_uri(tcfg.tracking_uri)
        else:
            if mlflow.is_tracking_uri_set():
                print(f"Defaulting to already set uri: {mlflow.get_tracking_uri()}")
        mlflow.set_experiment(self.experiment_name)
        if tcfg.autolog:
            import mlflow.sklearn

            mlflow.sklearn.autolog(log_models=tcfg.log_models, silent=True)

    # context amanger methods
    def __enter__(self) -> MLflowTracker:
        parent = self._mlflow.start_run(run_name=self.run_name)
        self.parent_run_id = parent.info.run_id
        # bgnorm.parent / bgnorm.parent_run_id let you scope a metric comparison to
        # one parameter setting's channels in the MLflow UI (filter on the tag).
        self._mlflow.set_tags(
            {
                "bgnorm.image": self.image_name,
                "bgnorm.steps": pipeline_steps(self.pipeline),
                "bgnorm.parent": self.run_name,
                "bgnorm.parent_run_id": self.parent_run_id,
            }
        )
        self._mlflow.log_param("steps", pipeline_steps(self.pipeline))
        self._mlflow.log_params(_flat_params(self.pipeline))
        return self

    def __exit__(self, *exc: object) -> bool:
        self._mlflow.end_run()
        return False  # never swallow exceptions

    def mlflow_health_check(self):
        import requests
        import mlflow
        from urllib.parse import urlparse, urljoin

        def is_http_uri(uri: str):
            try:
                parsed = urlparse(uri)
                # Checks if the protocol is exactly 'http' or 'https'
                return parsed.scheme in ('http', 'https')
            except ValueError:
                return False
        
        if (
            mlflow.is_tracking_uri_set() and 
            is_http_uri(mlflow.get_tracking_uri())
        ):
            uri = mlflow.get_tracking_uri()
            uri_health_endpoint = urljoin(uri, "health")
            try:
                resp = requests.get(uri_health_endpoint)
            except requests.exceptions.ConnectionError as e:
                print(e)
            
            assert resp.status_code == 200
        else:
            print("Tracking locally.")

    @contextmanager
    def channel(self, ch_name: object) -> Iterator[None]:
        """Nested run for a single channel, named after the channel. Carries the
        parent tags so a metric comparison can be scoped to this parent's channels."""
        with self._mlflow.start_run(run_name=str(ch_name), nested=True):
            self._mlflow.set_tags(
                {
                    "bgnorm.channel": str(ch_name),
                    "bgnorm.parent": self.run_name,
                    "bgnorm.parent_run_id": self.parent_run_id,
                }
            )
            yield

    def log_channel(
        self,
        channel: object,
        image: object,
        transformed: object,
        adjusted: object,
        adjusted_q: object,
        model: object,
        metrics: dict[str, float],
        pipeline: Pipeline,
    ) -> None:
        """Log the moments + ScoreMetrics for the active nested run (step params
        live on the parent, being identical across channels), then render and
        upload the registered per-channel plots as artifacts. `adjusted` is the
        bgnorm image; `adjusted_q` is the bgnormQ image (None without posthoc)."""
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if self.tcfg.log_plots:
            from . import plotting  # lazy: keeps matplotlib an optional dependency

            transform_label, _ = grid_labels(pipeline)
            ctx = plotting.ChannelPlotContext(
                channel=str(channel),
                image=image,
                transformed=transformed,
                adjusted=adjusted,
                adjusted_q=adjusted_q,
                model=model,
                metrics=metrics,
                pipeline=pipeline,
                transform_label=transform_label,
            )
            plotting.log_channel_artifacts(self._mlflow, ctx, self.tcfg.channel_plot_names)

    def log_summary(self, summaries: object) -> None:
        """Render and upload the SUMMARY_PLOTS as artifacts on the parent run."""
        if self.tcfg.log_plots:
            from . import plotting

            plotting.log_summary_artifacts(
                self._mlflow, summaries, self.tcfg.summary_plot_names
            )

    def log_grids(
        self,
        raw: object,
        transformed: object,
        merged: object,
        channel_dim: str,
    ) -> None:
        """Render and upload the downsampled all-channel image grids (raw,
        transformed, bgnorm, bgnormQ, labels, per-class probabilities) on the parent
        run. `transformed` may be None if it wasn't collected — that grid is skipped;
        the bgnormQ grid only appears when a posthoc quantile step ran."""
        if not (self.tcfg.log_plots and self.tcfg.log_grids):
            return
        from . import plotting

        probs = merged["probs"]
        transform_label, _ = grid_labels(self.pipeline)
        ctx = plotting.ImageGridContext(
            raw=raw,
            transformed=transformed,
            bgnorm=merged["adjusted_image"],
            bgnorm_q=merged["adjusted_image_post_hoc"] if "adjusted_image_post_hoc" in merged else None,
            labels=merged["labels"],
            probs=probs,
            signal_components=merged["signal_component"].values,
            background_components=merged["background_component"].values,
            channel_dim=channel_dim,
            n_components=int(probs.sizes[f"{channel_dim}_probs"]),
            max_px=self.tcfg.grid_max_px,
            transform_label=transform_label,
        )
        plotting.log_grid_artifacts(self._mlflow, ctx)
