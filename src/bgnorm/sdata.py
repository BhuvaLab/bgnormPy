"""Entrypoint to run BgNorm on SpatialData objects (Image2DModels only)."""

from spatialdata.models import Image2DModel, TableModel
import pandas as pd
import anndata as ad
from bgnorm.src.core import (
    bgnorm,
    Pipeline
)

def _bgnorm_sdata_shapeless():
    """bgnorms entirety of the selected image in the given SpatialData object."""
    pass

def _bgnorm_sdata_by_shape(
    sdata,
    image_key: str,
    scale: str,
    channel_dim: str,
    shape_key: str,
    shape_col: str = "geometry", # geomtry
):
    """bgnorms sections defined by shapes key of the selected image in the given
    SpatialData object."""
    pass

# def bgnorm_sdata(
#     sdata,
#     image_key="image",
#     scale="scale0",
#     channel_dim: str = "c",
#     tile_gdf_key: str | None = None,
#     tile_gdf_mark_col: str | None = None,
#     tmp_cache: bool = True,
#     tmp_cache_prefix: str = "_bgnorm",
#     tmp_clean_up: bool = True,
#     output_suffix="_bgnorm",
#     scale_factors: None | list[int] = None,
#     **bgnorm_kwargs
# ):
#     multichannel_image = sdata[image_key][scale].image

#     tile_gdf = (
#         sdata.shapes[tile_gdf_key] if tile_gdf_key in sdata.shapes else None
#     )

#     merged, bgnorm_channel_summaries = bgnorm(
#         multichannel_image,
#         channel_dim=channel_dim,
#         median_filter_radius=median_filter_radius,
#         image_cofactor=image_cofactor,
#         n_pixels_to_sample=n_pixels_to_sample,
#         pixel_sampling_seed=pixel_sampling_seed,
#         n_components=n_components,
#         tile_gdf=tile_gdf, 
#         tile_gdf_mark_col=tile_gdf_mark_col,
#         quantile_post_hoc_value=quantile_post_hoc_value,
#         tmp_cache=tmp_cache,
#         tmp_cache_prefix=tmp_cache_prefix,
#         tmp_clean_up=tmp_clean_up
#     )

#     adjusted_image_post_hoc = None
#     if "adjusted_image_post_hoc" in merged:
#         adjusted_image_post_hoc = merged["adjusted_image_post_hoc"]
#         adjusted_image_ph_key = f"{image_key}{output_suffix}_post_hoc"
#         sdata[adjusted_image_ph_key] = Image2DModel.parse(
#             adjusted_image_post_hoc,
#             dims=adjusted_image_post_hoc.dims,
#             c_coords=adjusted_image_post_hoc.coords[channel_dim],
#             scale_factors=scale_factors,
#         )
#         sdata.write_element(adjusted_image_ph_key)

#     # Save to sdata
#     adjusted_image = merged["adjusted_image"]
#     adjusted_image_key = f"{image_key}{output_suffix}"
#     sdata[adjusted_image_key] = Image2DModel.parse(
#         adjusted_image,
#         dims=adjusted_image.dims,
#         c_coords=adjusted_image.coords[channel_dim],
#         scale_factors=scale_factors,
#     )
#     sdata.write_element(adjusted_image_key)

#     # component labels. Image2DModel as it is per channel
#     component_labels = merged["labels"]
#     component_labels_key = f"{image_key}{output_suffix}_component_labels"
#     sdata[component_labels_key] = Image2DModel.parse(
#         component_labels,
#         dims=component_labels.dims,
#         c_coords=component_labels.coords[channel_dim],
#         scale_factors=scale_factors,
#     )
#     sdata.write_element(component_labels_key)

#     # Class probabiltiies
#     component_probs = merged["probs"]
#     for component in component_probs.coords[f"{channel_dim}_probs"].values:
#         prob_subset = component_probs.sel(c_probs=component)
#         c_prob_key = f"{image_key}{output_suffix}_probs_comp{component}"
#         sdata[c_prob_key] = Image2DModel.parse(
#             prob_subset,
#             dims=prob_subset.dims,
#             c_coords=prob_subset.coords[channel_dim],
#             scale_factors=scale_factors,
#         )
#         sdata.write_element(c_prob_key)

#     # Channel Summaries
#     channel_summary_key = "bgnorm_channel_summaries"
#     var_only = ad.AnnData(
#         var=pd.DataFrame(index=bgnorm_channel_summaries.index),
#         varm={
#             channel_summary_key: bgnorm_channel_summaries
#         }
#     )
#     sdata[channel_summary_key] = TableModel.parse(
#         var_only
#     )
#     sdata.write_element(channel_summary_key)
#     return sdata