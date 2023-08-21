from collections.abc import Iterator
import itertools
from typing import Any

import numpy

from variants.vars_io import GT_ARRAY_ID
from variants.iterators import ArraysChunk, resize_chunks, run_pipeline
from variants.pop_stats import (
    _calc_gt_is_missing,
    _calc_missing_rate_per_var,
    _calc_obs_het_rate_per_var,
)


def _update_remove_mask(remove_mask, new_remove_mask, filtering_info, filter_name):
    if remove_mask is None:
        remove_mask = new_remove_mask
    else:
        remove_mask = numpy.logical_or(remove_mask, new_remove_mask)

    num_vars_to_remove = numpy.sum(remove_mask)
    previous_num_vars_removed = filtering_info.get("vars_removed", 0)
    num_vars_removed_by_this_filter = num_vars_to_remove - previous_num_vars_removed

    try:
        num_vars_removed_per_filter = filtering_info["num_vars_removed_per_filter"]
    except KeyError:
        num_vars_removed_per_filter = {}
        filtering_info["num_vars_removed_per_filter"] = num_vars_removed_per_filter

    num_vars_removed_per_filter[filter_name] = num_vars_removed_by_this_filter

    return remove_mask


def _flt_chunk(chunk, max_var_obs_het, max_missing_rate):
    gts = chunk[GT_ARRAY_ID]

    remove_mask = numpy.zeros((chunk.num_rows), dtype=bool)
    filtering_info = {"num_vars_removed_per_filter": {}}

    gt_is_missing = None
    if max_missing_rate < 1:
        gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)
        missing_rate = _calc_missing_rate_per_var(gts, gt_is_missing=gt_is_missing)
        this_remove_mask = missing_rate > max_missing_rate
        remove_mask = _update_remove_mask(
            remove_mask, this_remove_mask, filtering_info, "missing_rate"
        )
    if max_var_obs_het < 1:
        gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)
        obs_het_rate = _calc_obs_het_rate_per_var(gts, gt_is_missing=gt_is_missing)
        this_remove_mask = obs_het_rate > max_var_obs_het
        remove_mask = _update_remove_mask(
            remove_mask, this_remove_mask, filtering_info, "obs_het"
        )

    flt_chunk = chunk.apply_mask(numpy.logical_not(remove_mask))

    return {"chunk": flt_chunk, "filtering_info": filtering_info}


class _ChunkFilterer:
    def __init__(self, max_var_obs_het, max_missing_rate):
        self.max_var_obs_het = max_var_obs_het
        self.max_missing_rate = max_missing_rate
        self.stats = None

    def __call__(self, chunk):
        flt_chunk_res = _flt_chunk(
            chunk,
            max_var_obs_het=self.max_var_obs_het,
            max_missing_rate=self.max_missing_rate,
        )

        if self.stats is None:
            stats = {"num_vars_removed_per_filter": {}}
            self.stats = stats
        else:
            stats = self.stats

        for filter_name, vars_removed in flt_chunk_res["filtering_info"][
            "num_vars_removed_per_filter"
        ].items():
            num_vars_previously_removed = stats["num_vars_removed_per_filter"].get(
                filter_name, 0
            )
            stats["num_vars_removed_per_filter"][filter_name] = (
                num_vars_previously_removed + vars_removed
            )

        return flt_chunk_res["chunk"]


class VariantFilterer:
    def __init__(
        self,
        max_var_obs_het: float = 1.0,
        max_missing_rate: float = 1.0,
        num_variants_per_result_chunk: int | None = None,
    ):
        self._filter_chunk = _ChunkFilterer(
            max_var_obs_het=max_var_obs_het,
            max_missing_rate=max_missing_rate,
        )

        self.num_variants_per_result_chunk = num_variants_per_result_chunk

    def __call__(self, variants: Iterator[ArraysChunk]) -> Iterator[ArraysChunk]:
        num_variants_per_result_chunk = self.num_variants_per_result_chunk
        if num_variants_per_result_chunk is None:
            try:
                chunk = next(variants)
                variants = itertools.chain([chunk], variants)
            except StopIteration:
                raise ValueError("No variants to filter")
            num_variants_per_result_chunk = chunk.num_rows

        flt_variants = run_pipeline(
            variants,
            map_functs=[self._filter_chunk],
        )
        uniform_flt_variants = resize_chunks(
            flt_variants, num_variants_per_result_chunk
        )

        return uniform_flt_variants

    @property
    def stats(self):
        return self._filter_chunk.stats
