from collections.abc import Iterator
import itertools

import numpy

from variants.vars_io import GT_ARRAY_ID, VARIANTS_ARRAY_ID
from variants.iterators import ArraysChunk, resize_chunks, run_pipeline
from variants.regions import GenomeLocation
from variants.pop_stats import (
    _calc_gt_is_missing,
    _calc_missing_rate_per_var,
    _calc_obs_het_rate_per_var,
    _calc_maf_per_var,
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


def _check_vars_in_regions(chroms, poss, regions_to_keep):
    in_any_region = None
    for region in regions_to_keep:
        region_start, region_end = region
        in_this_region = numpy.logical_and(
            chroms == region_start.chrom,
            numpy.logical_and(poss >= region_start.pos, poss <= region_end.pos),
        )
        if in_any_region is None:
            in_any_region = in_this_region
        else:
            in_any_region = numpy.logical_or(in_this_region, in_any_region)

    return in_any_region


def _flt_chunk(chunk, max_var_obs_het, max_missing_rate, max_maf, regions_to_keep):
    gts = chunk[GT_ARRAY_ID]

    remove_mask = numpy.zeros((chunk.num_rows), dtype=bool)
    filtering_info = {"num_vars_removed_per_filter": {}}

    gt_is_missing = None
    if regions_to_keep:
        variants_info = chunk[VARIANTS_ARRAY_ID]
        chroms = variants_info["chrom"].values
        poss = variants_info["pos"].values
        in_any_region = _check_vars_in_regions(chroms, poss, regions_to_keep)
        this_remove_mask = numpy.logical_not(in_any_region)
        remove_mask = _update_remove_mask(
            remove_mask, this_remove_mask, filtering_info, "desired_region"
        )
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
    if max_maf < 1:
        gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)
        mafs = _calc_maf_per_var(gts)
        this_remove_mask = mafs > max_maf
        remove_mask = _update_remove_mask(
            remove_mask, this_remove_mask, filtering_info, "maf"
        )

    flt_chunk = chunk.apply_mask(numpy.logical_not(remove_mask))

    return {"chunk": flt_chunk, "filtering_info": filtering_info}


class _ChunkFilterer:
    def __init__(self, max_var_obs_het, max_missing_rate, max_maf, regions_to_keep):
        self.max_var_obs_het = max_var_obs_het
        self.max_missing_rate = max_missing_rate
        self.max_maf = max_maf

        if regions_to_keep:
            regions_to_keep = [
                (GenomeLocation(chrom, start), GenomeLocation(chrom, end))
                for chrom, start, end in regions_to_keep
            ]
        self.regions_to_keep = regions_to_keep

        self.stats = None

    def __call__(self, chunk):
        flt_chunk_res = _flt_chunk(
            chunk,
            max_var_obs_het=self.max_var_obs_het,
            max_missing_rate=self.max_missing_rate,
            max_maf=self.max_maf,
            regions_to_keep=self.regions_to_keep,
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


def _add_samples_to_source_metadata(chunk, samples):
    source_metadata = chunk.source_metadata
    if source_metadata is None:
        source_metadata = {}
    source_metadata["samples"] = samples
    chunk.source_metadata = source_metadata
    return chunk


class VariantFilterer:
    def __init__(
        self,
        max_var_obs_het: float = 1.0,
        max_missing_rate: float = 1.0,
        max_maf: float = 1.0,
        num_variants_per_result_chunk: int | None = None,
        regions_to_keep: list[tuple[str, int, int]] | None = None,
        desired_arrays: list[str] | None = None,
    ):
        self._filter_chunk = _ChunkFilterer(
            max_var_obs_het=max_var_obs_het,
            max_missing_rate=max_missing_rate,
            max_maf=max_maf,
            regions_to_keep=regions_to_keep,
        )

        self.num_variants_per_result_chunk = num_variants_per_result_chunk
        self.desired_arrays = desired_arrays

    def __call__(self, variants: Iterator[ArraysChunk]) -> Iterator[ArraysChunk]:
        num_variants_per_result_chunk = self.num_variants_per_result_chunk
        if num_variants_per_result_chunk is None:
            try:
                chunk = next(variants)
                variants = itertools.chain([chunk], variants)
            except StopIteration:
                raise ValueError("No variants to filter")
            num_variants_per_result_chunk = chunk.num_rows
            samples = chunk.source_metadata["samples"]

        flt_variants = run_pipeline(
            variants,
            map_functs=[self._filter_chunk],
        )
        uniform_flt_variants = resize_chunks(
            flt_variants, num_variants_per_result_chunk
        )

        if self.desired_arrays:
            vars = (
                chunk.copy(desired_arrays=self.desired_arrays)
                for chunk in uniform_flt_variants
            )
        else:
            vars = uniform_flt_variants

        # resize chunks removes the source_metadata
        vars_with_samples = (
            _add_samples_to_source_metadata(chunk, samples) for chunk in vars
        )

        return vars_with_samples

    @property
    def stats(self):
        return self._filter_chunk.stats
