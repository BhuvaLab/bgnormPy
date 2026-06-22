"""Builders + validation for the bgnorm image matrix.
NOTE: originally designed for Image2DModels parsed for SpatialData objects,
this provides a more general entry point compatible with more input formats
"""
from __future__ import annotations

import os
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, model_validator

ArrayLike = np.ndarray | da.Array


class ImageLikeSchema(BaseModel):
    """ Compatible format for Bgnorm inputs
    
    Required attrs/fields enforced:
      - xr.DataArray
      - dim order (channel_dim, y_dim, x_dim)
      - a channel coordinate on ``channel_dim``
      - numeric dtype
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    image: xr.DataArray
    # usually below validated with Image2DModel .parse
    channel_dim: str = "c"
    y_dim: str = "y"
    x_dim: str = "x"

    @model_validator(mode="after")
    def _validate(self) -> ImageLikeSchema:
        """validate """
        img = self.image
        if not isinstance(img, xr.DataArray):  # pragma: no cover - pydantic guards too
            raise TypeError(f"image must be an xr.DataArray, got {type(img).__name__}")

        expected = (self.channel_dim, self.y_dim, self.x_dim)
        if tuple(img.dims) != expected:
            raise ValueError(f"image dims must be {expected}, got {tuple(img.dims)}")

        if self.channel_dim not in img.coords:
            raise ValueError(
                f"missing channel coordinate {self.channel_dim!r}; channels must be "
                "labelled (e.g. ['DAPI', 'CD4-ATTO550', ...])"
            )
        labels = [str(c) for c in img.coords[self.channel_dim].values]
        if len(set(labels)) != len(labels):
            raise ValueError(f"channel names must be unique, got {labels}")

        if not np.issubdtype(img.dtype, np.number) or np.issubdtype(
            img.dtype, np.complexfloating
        ):
            raise ValueError(f"image dtype must be real numeric, got {img.dtype}")

        return self

    @property
    def channel_names(self) -> list[str]:
        return [str(c) for c in self.image.coords[self.channel_dim].values]

    @classmethod
    def validate(cls, image: xr.DataArray, *, channel_dim: str = "c") -> xr.DataArray:
        """Validate and return the image matrix (raises on any violation)."""
        return cls(image=image, channel_dim=channel_dim).image


def as_image_matrix(
    image,
    *,
    channel_dim: str = "c",
    name: str | None = None,
    channel_names: list[str] | None = None,
) -> xr.DataArray:
    """Coerce any supported input into a validated bgnorm image matrix.
    """
    if isinstance(image, (str, os.PathLike)):
        return _from_path(
            image, channel_dim=channel_dim, name=name, channel_names=channel_names
        )

    if isinstance(image, xr.DataArray):
        resolved = name or (str(image.name) if image.name else "image")
        matrix = _to_cyx(image, channel_dim=channel_dim, name=resolved)
        if channel_names is not None:
            matrix = matrix.assign_coords({channel_dim: [str(c) for c in channel_names]})
        return ImageLikeSchema.validate(matrix, channel_dim=channel_dim)

    arr = image if isinstance(image, da.Array) else np.asarray(image)
    if arr.ndim == 3 and channel_names is None:  # auto-label multichannel arrays
        channel_names = [f"channel_{i}" for i in range(arr.shape[0])]
    return from_numpy(arr, name or "image", channel_names=channel_names, channel_dim=channel_dim)


def _from_path(
    path: os.PathLike | str,
    *,
    channel_dim: str = "c",
    name: str | None = None,
    channel_names: list[str] | None = None,
) -> xr.DataArray:
    """Read an image file into a validated image matrix, dispatched by extension.

    Currently supports .png, .qptiff. .tif/.tiff

    `channel_names` is forwarded to the PNG/TIFF readers; QPTIFF can derive its own
    channel labels from metadata. Raises on a missing file or unsupported extension.
    """
    p = os.fspath(path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"image path does not exist: {p}")
    low = p.lower()
    if low.endswith(".png"):
        return from_png(path, name, channel_names=channel_names, channel_dim=channel_dim)
    if low.endswith(".qptiff"):
        return from_qptiff(path, channel_dim=channel_dim, name=name)
    if low.endswith((".tif", ".tiff")):
        return from_tiff(path, name, channel_names=channel_names, channel_dim=channel_dim)
    raise ValueError(
        f"unsupported image extension for {p!r}; supported: .png, .tif/.tiff, .qptiff"
    )


def _channel_labels(
    n: int, channel_name: str | None, channel_names: list[str] | None
) -> list[str]:
    """Resolve channel coordinate labels for an array with ``n`` channels."""
    if channel_names is not None:
        if len(channel_names) != n:
            raise ValueError(f"channel_names has {len(channel_names)} entries, need {n}")
        return [str(c) for c in channel_names]
    if n == 1:
        return [str(channel_name) if channel_name is not None else "channel_0"]
    raise ValueError(f"channel_names is required for a {n}-channel array")


def _to_cyx(image: xr.DataArray, *, channel_dim: str, name: str) -> xr.DataArray:
    """Normalise an arbitrary DataArray (e.g. bioio 'TCZYX', spatialdata 'c,y,x')
    into a ``(channel_dim, y, x)`` matrix: rename dims case-insensitively, squeeze
    singleton extras, label channels if needed, set the name."""
    rename = {}
    for d in image.dims:
        low = str(d).lower()
        if low in ("c", "channel", "channels"):
            rename[d] = channel_dim
        elif low == "y":
            rename[d] = "y"
        elif low == "x":
            rename[d] = "x"
    image = image.rename(rename)

    core = {channel_dim, "y", "x"}
    for d in [d for d in image.dims if d not in core]:
        if image.sizes[d] == 1:
            image = image.squeeze(d, drop=True)
        else:
            raise ValueError(
                f"image has a non-singleton dim {d!r} (size {image.sizes[d]}); select a "
                "single index (timepoint / z-plane / scene) before passing to bgnorm"
            )

    if channel_dim not in image.dims:  # 2-D single-channel input
        image = image.expand_dims(channel_dim)
    image = image.transpose(channel_dim, "y", "x")

    if channel_dim not in image.coords:
        n = image.sizes[channel_dim]
        labels = [name] if n == 1 else [f"channel_{i}" for i in range(n)]
        image = image.assign_coords({channel_dim: labels})
    if not image.name:
        image = image.rename(name)
    return image


def from_numpy(
    array: ArrayLike,
    name: str,
    channel_name: str | None = None,
    channel_names: list[str] | None = None,
    *,
    channel_dim: str = "c",
) -> xr.DataArray:
    """Build a validated image matrix from a ``(y, x)`` or ``(c, y, x)`` array.

    A 2-D array becomes a single channel labelled ``channel_name`` (default
    ``"channel_0"``); a 3D array (multichannel) requires ``channel_names`` of matching length.
    """
    arr = array if isinstance(array, da.Array) else np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None]  # (y, x) -> (1, y, x)
    elif arr.ndim != 3:
        raise ValueError(f"expected a (y, x) or (c, y, x) array, got ndim={arr.ndim}")

    labels = _channel_labels(arr.shape[0], channel_name, channel_names)
    img = xr.DataArray(
        arr, dims=(channel_dim, "y", "x"), coords={channel_dim: labels}, name=name
    )
    return ImageLikeSchema.validate(img, channel_dim=channel_dim)


def from_png(
    png_file_path: os.PathLike | str,
    name: str | None = None,
    channel_name: str | None = None,
    channel_names: list[str] | None = None,
    *,
    channel_dim: str = "c",
) -> xr.DataArray:
    from PIL import Image
    arr = np.asarray(Image.open(os.fspath(png_file_path)))
    name = name or Path(png_file_path).stem
    if arr.ndim == 3:  # (y, x, channels) -> (channels, y, x)
        arr = np.moveaxis(arr, -1, 0)
        if channel_names is None:
            channel_names = ["r", "g", "b", "a"][: arr.shape[0]]
    return from_numpy(
        arr, name, channel_name=channel_name, channel_names=channel_names,
        channel_dim=channel_dim,
    )

def from_tiff(
    tiff_file_path: os.PathLike | str,
    name: str | None = None,
    channel_name: str | None = None,
    channel_names: list[str] | None = None,
    *,
    channel_dim: str = "c",
) -> xr.DataArray:
    """Build a validated image matrix from a plain (unannotated) TIFF. A 3-D TIFF
    is assumed channels-first ``(c, y, x)`` and needs ``channel_names``; for rich
    QPTIFF metadata use :func:`from_qptiff` instead."""
    import tifffile
    arr = tifffile.imread(os.fspath(tiff_file_path))
    name = name or Path(tiff_file_path).stem
    return from_numpy(
        arr, name, channel_name=channel_name, channel_names=channel_names,
        channel_dim=channel_dim,
    )

def from_spatialdata(
    sdata,
    image_key: str,
    *,
    scale: str = "scale0", # NOTE: for now enforce bgnorm on full scale images, not tested for lower pyramids
    channel_dim: str = "c",
    name: str | None = None,
) -> xr.DataArray:
    """Build a validated image matrix from a SpatialData object's image element.
    Handles both multiscale (DataTree) and single-scale (DataArray) elements."""
    try:
        import spatialdata as sd  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "SpatialData not installed. Install with extras: "
            "`pip install bgnorm[spatialdata]` or `uv add bgnorm[spatialdata]`"
        ) from e

    element = sdata[image_key]
    try:  # multiscale_spatial_image: indexable by scale, image lives in `.image`
        image = element[scale].image
    except (KeyError, TypeError, AttributeError):
        image = element  # single-scale Image2DModel is already a DataArray
    if not isinstance(image, xr.DataArray):
        raise TypeError(f"sdata[{image_key!r}] did not resolve to an xr.DataArray")

    matrix = _to_cyx(image, channel_dim=channel_dim, name=name or image_key)
    return ImageLikeSchema.validate(matrix, channel_dim=channel_dim)


# below is our custom qptiff reader fork from bioio, enforce this branch to be installed
_QPTIFF_FORK_MARKERS = ("QptiffMetadata", "ome_metadata_from_qptiff", "squeeze_to_cyx")
_QPTIFF_INSTALL_HINT = (
    "QPTIFF support needs the forked bioio-tifffile reader. Install it with:\n"
    '  pip install "bgnorm[qptiff]"\n'
    "or directly:\n"
    "  pip install bioio "
    '"bioio-tifffile @ git+https://github.com/rtubelleza/bioio-tifffile.git'
    '@feature/read-qptiffs-rich-reduce-ome"'
)


def _require_qptiff_fork():
    """Import bioio + the forked reader, asserting it is the qptiff fork branch."""
    try:
        from bioio import BioImage
        import bioio_tifffile
    except ImportError as e:
        raise ImportError(_QPTIFF_INSTALL_HINT) from e

    missing = [m for m in _QPTIFF_FORK_MARKERS if not hasattr(bioio_tifffile, m)]
    if missing:
        raise ImportError(
            "the installed bioio-tifffile is not the qptiff fork bgnorm requires "
            f"(missing {missing}).\n{_QPTIFF_INSTALL_HINT}"
        )
    return BioImage, bioio_tifffile


def from_qptiff(
    qptiff_path: os.PathLike | str,
    *,
    scene_idx: int | str = 0,
    scene_name: str | None = "FullResolution", # default for the main image in qptiffs
    channel_dim: str = "c",
    name: str | None = None,
) -> xr.DataArray:
    """Build a validated image matrix from QPTIFF via our custom forked
    Bioio reader. Reads the chosen scene, normalises bioio's dims to ``(c, y, x)``,
    and carries the channel-name coordinate the reader derives from QPI/OME metadata.
    """
    BioImage, bioio_tifffile = _require_qptiff_fork()

    img = BioImage(os.fspath(qptiff_path), reader=bioio_tifffile.Reader)
    scenes = list(img.scenes)
    img.set_scene(scenes[scene_idx] if isinstance(scene_idx, int) else scene_idx)

    matrix = _to_cyx(
        img.xarray_dask_data, channel_dim=channel_dim, name=name or Path(qptiff_path).stem
    )
    return ImageLikeSchema.validate(matrix, channel_dim=channel_dim)
