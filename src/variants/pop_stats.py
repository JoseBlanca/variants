from functools import partial

import numpy
import pandas

from variants.iterators import VariantsIterator, ArrayChunkIterator, run_pipeline
from variants.vars_io import GT_ARRAY_ID, MISSING_INT

from enum import Enum

LINEAL = "lineal"
LOGARITHMIC = "logarithmic"
BinType = Enum("BinType", [LINEAL, LOGARITHMIC])


def _calc_pops_idxs(pops: dict[list[str]] | None, samples):
    if pops is None:
        pops_idxs = {0: slice(None, None)}
    else:
        samples_idx = {sample: idx for idx, sample in enumerate(samples)}
        pops_idxs = {}
        for pop_id, pop_samples in pops.items():
            pop_idxs = [samples_idx[sample] for sample in pop_samples]
            pops_idxs[pop_id] = pop_idxs
    return pops_idxs


def _calc_gt_is_missing(gts, gt_is_missing=None):
    if gt_is_missing is not None:
        return gt_is_missing
    allele_is_missing = gts == MISSING_INT
    allele_is_missing = numpy.any(allele_is_missing, axis=2)
    return allele_is_missing


def _calc_gts_is_het(gts, gt_is_missing=None):
    gt_is_het = numpy.logical_not(
        numpy.all(gts == gts[:, :, 0][:, :, numpy.newaxis], axis=2)
    )
    gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)
    gt_is_het = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))
    return gt_is_het


def _calc_obs_het_per_var_for_chunk(chunk, pops):
    gts = chunk[GT_ARRAY_ID].array
    gt_is_missing = _calc_gt_is_missing(gts)
    gt_is_het = _calc_gts_is_het(gts, gt_is_missing=gt_is_missing)
    gt_is_het = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))

    obs_het_per_var = {}
    for pop_id, pop_slice in pops.items():
        num_vars_het_per_var = numpy.sum(gt_is_het[:, pop_slice], axis=1)
        num_non_missing_per_var = gts.shape[1] - numpy.sum(
            gt_is_missing[:, pop_slice], axis=1
        )
        with numpy.errstate(invalid="ignore"):
            obs_het_per_var[pop_id] = num_vars_het_per_var / num_non_missing_per_var

    obs_het_per_var = pandas.DataFrame(obs_het_per_var)
    return {"obs_het_per_var": obs_het_per_var}


def _prepare_bins(
    hist_kwargs: dict,
    range=tuple[int, int],
    default_num_bins=40,
    default_bin_type: BinType = LINEAL,
):
    num_bins = hist_kwargs.get("num_bins", default_num_bins)
    bin_type = hist_kwargs.get("bin_type", default_bin_type)

    if bin_type == LINEAL:
        bins = numpy.linspace(range[0], range[1], num_bins + 1)
    elif bin_type == LOGARITHMIC:
        if range[0] == 0:
            raise ValueError("range[0] cannot be zero for logarithmic bins")
        bins = numpy.logspace(range[0], range[1], num_bins + 1)
    return bins


def _collect_stats_from_pop_dframes(
    accumulated_result, next_result: pandas.DataFrame, hist_bins_edges: numpy.array
):
    if accumulated_result is None:
        accumulated_result = {
            "sum_per_pop": pandas.Series(
                numpy.zeros((next_result.shape[1]), dtype=int),
                index=next_result.columns,
            ),
            "total_num_rows": pandas.Series(
                numpy.zeros((next_result.shape[1]), dtype=int),
                index=next_result.columns,
            ),
            "hist_counts": None,
        }

    accumulated_result["sum_per_pop"] += next_result.sum(axis=0)
    accumulated_result["total_num_rows"] += next_result.shape[
        0
    ] - next_result.isna().sum(axis=0)

    this_counts = {}
    for pop, pop_stats in next_result.items():
        this_counts[pop] = numpy.histogram(pop_stats, bins=hist_bins_edges)[0]
    this_counts = pandas.DataFrame(this_counts)

    if accumulated_result["hist_counts"] is None:
        accumulated_result["hist_counts"] = this_counts
    else:
        accumulated_result["hist_counts"] += this_counts

    return accumulated_result


def calc_obs_het_stats_per_var(
    variants: VariantsIterator,
    pops: list[str] | None = None,
    hist_kwargs=None,
) -> ArrayChunkIterator:
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_bins_edges = _prepare_bins(hist_kwargs, range=hist_kwargs.get("range", (0, 1)))

    pops = _calc_pops_idxs(pops, variants.samples)

    calc_obs_het_per_var_for_chunk = partial(_calc_obs_het_per_var_for_chunk, pops=pops)

    collect_stats_from_pop_dframes = partial(
        _collect_stats_from_pop_dframes, hist_bins_edges=hist_bins_edges
    )

    accumulated_result = run_pipeline(
        variants,
        map_functs=[calc_obs_het_per_var_for_chunk, lambda x: x["obs_het_per_var"]],
        reduce_funct=collect_stats_from_pop_dframes,
        reduce_initialializer=None,
    )

    mean = accumulated_result["sum_per_pop"] / accumulated_result["total_num_rows"]
    return {
        "mean": mean,
        "hist_bin_edges": hist_bins_edges,
        "hist_counts": accumulated_result["hist_counts"],
    }
