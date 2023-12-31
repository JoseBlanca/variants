from collections.abc import Iterator
from functools import partial
import itertools

import numpy
import pandas

from variants.iterators import (
    run_pipeline,
    ArraysChunk,
    get_samples_from_chunk,
    _peek_vars_iter,
)
from variants.globals import (
    GT_ARRAY_ID,
    MISSING_INT,
    VARIANTS_ARRAY_ID,
    CHROM_VARIANTS_COL,
    POS_VARIANTS_COL,
    MIN_NUM_GENOTYPES_FOR_POP_STAT,
)
from variants.variants import Variants

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


def _calc_missing_rate_per_var(gts, gt_is_missing=None):
    gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)

    missing_rate = gt_is_missing.sum(axis=1) / gt_is_missing.shape[1]
    return missing_rate


def _calc_gts_is_het(gts, gt_is_missing=None):
    gt_is_het = numpy.logical_not(
        numpy.all(gts == gts[:, :, 0][:, :, numpy.newaxis], axis=2)
    )
    gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)
    gt_is_het = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))
    return gt_is_het


def _calc_obs_het_rate_per_var(gts, gt_is_missing=None):
    gt_is_het = _calc_gts_is_het(gts, gt_is_missing=gt_is_missing)
    obs_het_rate = gt_is_het.sum(axis=1) / gt_is_missing.shape[1]
    return obs_het_rate


def _calc_obs_het_per_var_for_chunk(chunk, pops):
    gts = chunk.gt_array
    return _calc_obs_het_per_var_for_gts(gts, pops)


def _calc_obs_het_per_var_for_gts(gts, pops):
    gt_is_missing = _calc_gt_is_missing(gts)
    gt_is_het = _calc_gts_is_het(gts, gt_is_missing=gt_is_missing)
    gt_is_het = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))

    obs_het_per_var = {}
    called_gts_per_var = {}
    for pop_id, pop_slice in pops.items():
        num_vars_het_per_var = numpy.sum(gt_is_het[:, pop_slice], axis=1)
        num_samples = len(pop_slice) if isinstance(pop_slice, list) else gts.shape[1]
        num_non_missing_per_var = num_samples - numpy.sum(
            gt_is_missing[:, pop_slice], axis=1
        )
        with numpy.errstate(invalid="ignore"):
            obs_het_per_var[pop_id] = num_vars_het_per_var / num_non_missing_per_var
        called_gts_per_var[pop_id] = num_non_missing_per_var

    obs_het_per_var = pandas.DataFrame(obs_het_per_var)
    return {
        "obs_het_per_var": obs_het_per_var,
        "called_gts_per_var": called_gts_per_var,
    }


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


def get_different_alleles(
    vars_iter: Iterator[Variants], samples_to_consider: list[str] | None = None
):
    def accumulate_alleles(accumulated_alleles, new_alleles):
        accumulated_alleles.update(new_alleles)
        return accumulated_alleles

    if samples_to_consider:
        chunk, vars_iter = _peek_vars_iter(vars_iter)
        samples_slice = _calc_pops_idxs(
            pops={0: samples_to_consider}, samples=chunk.samples
        )[0]
    else:
        samples_slice = slice(None, None)

    result = run_pipeline(
        vars_iter,
        map_functs=[lambda chunk: numpy.unique(chunk.gt_array[:, samples_slice, :])],
        reduce_funct=accumulate_alleles,
        reduce_initialializer=set(),
    )
    result = sorted(result.difference([MISSING_INT]))
    return result


def _count_alleles_per_var(
    gts,
    pops: dict[str, list[int]],
    calc_freqs: bool,
    alleles=None,
    missing_gt=MISSING_INT,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
):
    alleles_in_chunk = set(numpy.unique(gts)).difference([missing_gt])
    ploidy = gts.shape[2]

    if alleles is not None:
        if alleles_in_chunk.difference(alleles):
            raise RuntimeError(
                f"These gts have alleles ({alleles_in_chunk}) not present in the given ones ({alleles})"
            )
    else:
        alleles = sorted(alleles_in_chunk)

    result = {}
    for pop_id, pop_slice in pops.items():
        pop_gts = gts[:, pop_slice, :]
        allele_counts = numpy.empty(
            shape=(pop_gts.shape[0], len(alleles)), dtype=numpy.int64
        )
        missing_counts = None
        for idx, allele in enumerate([missing_gt] + alleles):
            allele_counts_per_row = numpy.sum(pop_gts == allele, axis=(1, 2))
            if idx == 0:
                missing_counts = allele_counts_per_row
            else:
                allele_counts[:, idx - 1] = allele_counts_per_row
        allele_counts = pandas.DataFrame(allele_counts, columns=alleles)

        result[pop_id] = {
            "allele_counts": allele_counts,
            "missing_gts_per_var": missing_counts,
        }

        if calc_freqs:
            expected_num_allelic_gts_in_snp = pop_gts.shape[1] * pop_gts.shape[2]
            num_allelic_gts_per_snp = expected_num_allelic_gts_in_snp - missing_counts
            num_allelic_gts_per_snp = num_allelic_gts_per_snp.reshape(
                (num_allelic_gts_per_snp.shape[0], 1)
            )
            allelic_freqs_per_snp = allele_counts / num_allelic_gts_per_snp
            num_gts_per_snp = (
                num_allelic_gts_per_snp.reshape((num_allelic_gts_per_snp.size,))
                / ploidy
            )
            not_enough_data = num_gts_per_snp < min_num_genotypes
            allelic_freqs_per_snp[not_enough_data] = numpy.nan

            result[pop_id]["allelic_freqs"] = allelic_freqs_per_snp

    return {"counts": result, "alleles": alleles_in_chunk}


def _calc_maf_per_var(gts, missing_gt=MISSING_INT):
    allelic_freqs = _count_alleles_per_var(
        gts,
        pops={0: slice(None, None)},
        alleles=None,
        missing_gt=missing_gt,
        calc_freqs=True,
    )["counts"][0]["allelic_freqs"]

    mafs = allelic_freqs.max(axis=1).values
    return mafs


def _calc_maf_per_var_for_chunk(
    chunk,
    pops,
    missing_gt=MISSING_INT,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
):
    res = _count_alleles_per_var(
        chunk[GT_ARRAY_ID].values,
        pops,
        alleles=None,
        missing_gt=missing_gt,
        calc_freqs=True,
        min_num_genotypes=min_num_genotypes,
    )
    major_allele_freqs = {}
    for pop, pop_res in res["counts"].items():
        pop_allelic_freqs = pop_res["allelic_freqs"]
        major_allele_freqs[pop] = pop_allelic_freqs.max(axis=1)
    major_allele_freqs = pandas.DataFrame(major_allele_freqs)
    return {"major_allele_freqs_per_var": major_allele_freqs}


def _get_metadata_from_variants(variants):
    try:
        chunk = next(variants)
    except StopIteration:
        raise ValueError("No variants")
    variants = itertools.chain([chunk], variants)
    samples = get_samples_from_chunk(chunk)
    ploidy = chunk.ploidy
    return {"samples": samples, "ploidy": ploidy}, variants


def _get_samples_from_variants(variants):
    metadata, variants = _get_metadata_from_variants(variants)
    return metadata["samples"], variants


def calc_major_allele_stats_per_var(
    vars_iter: Iterator[ArraysChunk],
    pops: list[str] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    hist_kwargs=None,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["range"] = hist_kwargs.get("range", (0, 1))

    samples, variants = _get_samples_from_variants(vars_iter)
    pops = _calc_pops_idxs(pops, samples)

    return _calc_stats_per_var(
        variants=variants,
        calc_stats_for_chunk=partial(
            _calc_maf_per_var_for_chunk, pops=pops, min_num_genotypes=min_num_genotypes
        ),
        get_stats_for_chunk_result=lambda x: x["major_allele_freqs_per_var"],
        hist_kwargs=hist_kwargs,
    )


def calc_qual_per_var_for_chunk(chunk):
    quals = pandas.DataFrame({0: chunk.arrays[VARIANTS_ARRAY_ID]["qual"]})
    return {"quals": quals}


def _calc_stats_per_var(
    variants: Iterator[ArraysChunk],
    calc_stats_for_chunk,
    get_stats_for_chunk_result,
    hist_kwargs=None,
) -> Iterator[ArraysChunk]:
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_bins_edges = _prepare_bins(hist_kwargs, range=hist_kwargs["range"])

    collect_stats_from_pop_dframes = partial(
        _collect_stats_from_pop_dframes, hist_bins_edges=hist_bins_edges
    )

    accumulated_result = run_pipeline(
        variants,
        map_functs=[
            calc_stats_for_chunk,
            get_stats_for_chunk_result,
        ],
        reduce_funct=collect_stats_from_pop_dframes,
        reduce_initialializer=None,
    )

    mean = accumulated_result["sum_per_pop"] / accumulated_result["total_num_rows"]
    return {
        "mean": mean,
        "hist_bin_edges": hist_bins_edges,
        "hist_counts": accumulated_result["hist_counts"],
    }


def calc_qual_stats_per_var(
    vars_iter: Iterator[ArraysChunk],
    hist_kwargs=None,
) -> Iterator[ArraysChunk]:
    if hist_kwargs is None:
        raise ValueError(
            "hist_kwargs should be a dict and at least has to provide the range"
        )
    if "range" not in hist_kwargs:
        raise ValueError("You should provide a range for the histogram in hist_kwargs")

    return _calc_stats_per_var(
        variants=vars_iter,
        calc_stats_for_chunk=calc_qual_per_var_for_chunk,
        get_stats_for_chunk_result=lambda x: x["quals"],
        hist_kwargs=hist_kwargs,
    )


def calc_obs_het_stats_per_var(
    vars_iter: Iterator[ArraysChunk],
    pops: list[str] | None = None,
    hist_kwargs=None,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["range"] = hist_kwargs.get("range", (0, 1))

    samples, variants = _get_samples_from_variants(vars_iter)
    pops = _calc_pops_idxs(pops, samples)

    return _calc_stats_per_var(
        variants=variants,
        calc_stats_for_chunk=partial(_calc_obs_het_per_var_for_chunk, pops=pops),
        get_stats_for_chunk_result=lambda x: x["obs_het_per_var"],
        hist_kwargs=hist_kwargs,
    )


def _calc_expected_het_per_snp(
    gts, pops, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT, ploidy=None
):
    if ploidy is None:
        ploidy = gts.shape[2]

    res = _count_alleles_per_var(
        gts,
        pops=pops,
        calc_freqs=True,
        min_num_genotypes=min_num_genotypes,
    )

    sorted_pops = sorted(pops.keys())

    missing_allelic_gts = {
        pop_id: res["counts"][pop_id]["missing_gts_per_var"] for pop_id in sorted_pops
    }
    missing_allelic_gts = pandas.DataFrame(missing_allelic_gts, columns=sorted_pops)

    exp_het = {}
    for pop_id in sorted_pops:
        allele_freqs = res["counts"][pop_id]["allelic_freqs"].values
        exp_het[pop_id] = 1 - numpy.sum(allele_freqs**ploidy, axis=1)
    exp_het = pandas.DataFrame(exp_het, columns=sorted_pops)

    return {"exp_het": exp_het, "missing_allelic_gts": missing_allelic_gts}


def _calc_unbiased_exp_het_per_snp(
    gts, pops, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT, ploidy=None
):
    "Calculated using Unbiased Heterozygosity (Codom Data) Genalex formula"
    if ploidy is None:
        ploidy = gts.shape[2]

    res = _calc_expected_het_per_snp(
        gts,
        pops=pops,
        min_num_genotypes=min_num_genotypes,
        ploidy=ploidy,
    )
    exp_het = res["exp_het"]

    missing_allelic_gts = res["missing_allelic_gts"]

    num_allelic_gtss = []
    for pop in missing_allelic_gts.columns:
        pop_slice = pops[pop]
        if (
            isinstance(pop_slice, slice)
            and pop_slice.start is None
            and pop_slice.stop is None
            and pop_slice.step is None
        ):
            num_allelic_gts = gts.shape[1] * ploidy
        else:
            num_allelic_gts = len(pops[pop]) * ploidy
        num_allelic_gtss.append(num_allelic_gts)
    num_exp_allelic_gts_per_pop = numpy.array(num_allelic_gtss)
    num_called_allelic_gts_per_snp = (
        num_exp_allelic_gts_per_pop[numpy.newaxis, :] - missing_allelic_gts
    )
    num_samples = num_called_allelic_gts_per_snp / ploidy

    unbiased_exp_het = (2 * num_samples / (2 * num_samples - 1)) * exp_het
    return unbiased_exp_het


def _calc_exp_het_per_var_for_chunk(
    chunk, pops, min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT, ploidy=None
):
    gts = chunk.gt_array
    return {
        "unbiased_exp_het_per_var": _calc_unbiased_exp_het_per_snp(
            gts, pops, min_num_genotypes=min_num_genotypes, ploidy=ploidy
        )
    }


def calc_exp_het_stats_per_var(
    vars_iter: Iterator[ArraysChunk],
    pops: dict[str, list[str]] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    ploidy=None,
    hist_kwargs=None,
):
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["range"] = hist_kwargs.get("range", (0, 1))

    samples, variants = _get_samples_from_variants(vars_iter)
    pops = _calc_pops_idxs(pops, samples)

    return _calc_stats_per_var(
        variants=variants,
        calc_stats_for_chunk=partial(
            _calc_exp_het_per_var_for_chunk,
            pops=pops,
            min_num_genotypes=min_num_genotypes,
            ploidy=ploidy,
        ),
        get_stats_for_chunk_result=lambda x: x["unbiased_exp_het_per_var"],
        hist_kwargs=hist_kwargs,
    )


def create_chrom_pos_pandas_index_from_vars(vars: Variants):
    if VARIANTS_ARRAY_ID in vars.arrays:
        variants_info = vars.arrays[VARIANTS_ARRAY_ID]
        chroms = variants_info[CHROM_VARIANTS_COL]
        poss = variants_info[POS_VARIANTS_COL]
        index = pandas.MultiIndex.from_arrays(
            [chroms, poss], names=[CHROM_VARIANTS_COL, POS_VARIANTS_COL]
        )
    else:
        index = pandas.RangeIndex(vars.num_rows)
    return index


def calc_exp_het_per_var(
    vars_iter: Iterator[ArraysChunk],
    pops: dict[str, list[str]] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
    ploidy=None,
):
    samples, vars_iter = _get_samples_from_variants(vars_iter)
    pops = _calc_pops_idxs(pops, samples)

    for vars in vars_iter:
        exp_het = _calc_exp_het_per_var_for_chunk(
            vars,
            pops=pops,
            min_num_genotypes=min_num_genotypes,
            ploidy=ploidy,
        )["unbiased_exp_het_per_var"]
        exp_het.index = create_chrom_pos_pandas_index_from_vars(vars)
        yield exp_het
