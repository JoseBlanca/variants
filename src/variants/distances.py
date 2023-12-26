from collections.abc import Iterator
import itertools

import numpy
import pandas
import scipy

from variants.variants import Variants
from variants.iterators import run_pipeline, _peek_vars_iter, ArraysChunk
from variants.globals import MISSING_INT, MIN_NUM_GENOTYPES_FOR_POP_STAT
from variants.pop_stats import (
    _calc_pops_idxs,
    _calc_obs_het_per_var_for_gts,
    _count_alleles_per_var,
    _get_samples_from_variants,
)


class Distances:
    def __init__(self, dist_vector, names):
        self.names = numpy.array(names)
        self._dist_vector = dist_vector

    @property
    def dist_vector(self):
        return self._dist_vector

    @property
    def square_dists(self):
        dists = scipy.spatial.distance.squareform(self.dist_vector)
        dists = pandas.DataFrame(dists, index=self.names, columns=self.names)
        return dists

    @property
    def triang_list_of_lists(self):
        dist_vector = iter(self.dist_vector)
        length = 0
        dists = []
        while True:
            dist_row = list(itertools.islice(dist_vector, length))
            if length and not dist_row:
                break
            dist_row.append(0)
            dists.append(dist_row)
            length += 1
        return dists


def _is_missing(matrix, axis=1):
    if axis is None:
        return matrix == MISSING_INT
    else:
        return numpy.any(matrix == MISSING_INT, axis=axis)


def _get_sample_gts(gts, sample_i, sample_j, indi_cache):
    if sample_i in indi_cache:
        indi1, is_missing_1 = indi_cache[sample_i]
    else:
        indi1 = gts[:, sample_i]
        is_missing_1 = _is_missing(indi1)
        indi_cache[sample_i] = indi1, is_missing_1

    if sample_j in indi_cache:
        indi2, is_missing_2 = indi_cache[sample_j]
    else:
        indi2 = gts[:, sample_j]
        is_missing_2 = _is_missing(indi2)
        indi_cache[sample_j] = indi2, is_missing_2

    is_called = numpy.logical_not(numpy.logical_or(is_missing_1, is_missing_2))

    indi1 = indi1[is_called]
    indi2 = indi2[is_called]

    assert issubclass(indi1.dtype.type, numpy.integer)
    assert issubclass(indi2.dtype.type, numpy.integer)

    return indi1, indi2


def _kosman(gts, sample_i, sample_j, indi_cache):
    """It calculates the distance between two individuals using the Kosman dist

    The Kosman distance is explain in DOI: 10.1111/j.1365-294X.2005.02416.x
    """

    indi1, indi2 = _get_sample_gts(gts, sample_i, sample_j, indi_cache)

    if indi1.shape[1] != 2:
        raise ValueError("Only diploid are allowed")

    alleles_comparison1 = indi1 == indi2.transpose()[:, :, None]
    alleles_comparison2 = indi2 == indi1.transpose()[:, :, None]

    result = numpy.add(
        numpy.any(alleles_comparison2, axis=2).sum(axis=0),
        numpy.any(alleles_comparison1, axis=2).sum(axis=0),
    )

    result2 = numpy.full(result.shape, fill_value=0.5)
    result2[result == 0] = 1
    result2[result == 4] = 0
    return result2.sum(), result2.shape[0]


_PAIRWISE_DISTANCES = {"kosman": _kosman}


class _IndiPairwiseCalculator:
    def __init__(self):
        self._pairwise_dist_cache = {}
        self._indi_cache = {}

    def calc_dist(
        self, variations, method="kosman", pop1_samples=None, pop2_samples=None
    ):
        gts = variations.gt_array
        dist_cache = self._pairwise_dist_cache
        indi_cache = self._indi_cache

        identical_indis = numpy.unique(gts, axis=1, return_inverse=True)[1]

        if pop1_samples is None:
            n_samples = gts.shape[1]
            num_dists_to_calculate = int((n_samples**2 - n_samples) / 2)
            dists = numpy.zeros(num_dists_to_calculate)
            n_snps_matrix = numpy.zeros(num_dists_to_calculate)
        else:
            shape = (len(pop1_samples), len(pop2_samples))
            dists = numpy.zeros(shape)
            n_snps_matrix = numpy.zeros(shape)

        index = 0
        dist_funct = _PAIRWISE_DISTANCES[method]

        if pop1_samples is None:
            sample_combinations = itertools.combinations(range(n_samples), 2)
        else:
            pop1_sample_idxs = [
                idx
                for idx, sample in enumerate(variations.samples)
                if sample in pop1_samples
            ]
            pop2_sample_idxs = [
                idx
                for idx, sample in enumerate(variations.samples)
                if sample in pop2_samples
            ]
            sample_combinations = itertools.product(pop1_sample_idxs, pop2_sample_idxs)
        for sample_i, sample_j in sample_combinations:
            indentical_type_for_sample_i = identical_indis[sample_i]
            indentical_type_for_sample_j = identical_indis[sample_j]
            key = tuple(
                sorted((indentical_type_for_sample_i, indentical_type_for_sample_j))
            )
            try:
                dist, n_snps = dist_cache[key]
            except KeyError:
                dist, n_snps = dist_funct(gts, sample_i, sample_j, indi_cache)
                dist_cache[key] = dist, n_snps

            if pop1_samples is None:
                dists[index] = dist
                n_snps_matrix[index] = n_snps
                index += 1
            else:
                dists_samplei_idx = pop1_sample_idxs.index(sample_i)
                dists_samplej_idx = pop2_sample_idxs.index(sample_j)
                dists[dists_samplei_idx, dists_samplej_idx] = dist
                n_snps_matrix[dists_samplei_idx, dists_samplej_idx] = n_snps
        return dists, n_snps_matrix


def _calc_kosman_dist_for_chunk(chunk: Variants):
    pairwise_dist_calculator = _IndiPairwiseCalculator()
    return pairwise_dist_calculator.calc_dist(chunk, method="kosman")


def _reduce_kosman_dists(acummulated_dists_and_snps, new_dists_and_snps):
    new_dists, new_n_snps = new_dists_and_snps
    if acummulated_dists_and_snps is None:
        abs_distances = new_dists_and_snps[0].copy()
        n_snps_matrix = new_dists_and_snps[1]
    else:
        abs_distances, n_snps_matrix = acummulated_dists_and_snps
        abs_distances = numpy.add(abs_distances, new_dists)
        n_snps_matrix = numpy.add(n_snps_matrix, new_n_snps)
    return abs_distances, n_snps_matrix


def calc_pairwise_kosman_dists(
    vars_iter: Iterator[Variants], min_num_snps=None, num_processes=2
) -> Distances:
    """It calculates the distance between individuals using the Kosman
    distance.

    The Kosman distance is explained in DOI: 10.1111/j.1365-294X.2005.02416.x
    """
    chunk, vars_iter = _peek_vars_iter(vars_iter)
    samples = chunk.samples

    res = run_pipeline(
        vars_iter,
        map_functs=[_calc_kosman_dist_for_chunk],
        reduce_funct=_reduce_kosman_dists,
        reduce_initialializer=None,
        num_processes=num_processes,
    )
    abs_distances, n_snps_matrix = res

    if min_num_snps is not None:
        n_snps_matrix[n_snps_matrix < min_num_snps] = numpy.nan

    with numpy.errstate(invalid="ignore"):
        dists = abs_distances / n_snps_matrix

    dists = Distances(dists, samples)
    return dists


def hmean(array, axis=0, dtype=None):
    # Harmonic mean only defined if greater than zero
    if isinstance(array, numpy.ma.MaskedArray):
        size = array.count(axis)
    else:
        if axis is None:
            array = array.ravel()
            size = array.shape[0]
        else:
            size = array.shape[axis]
    with numpy.errstate(divide="ignore"):
        inverse_mean = numpy.sum(1.0 / array, axis=axis, dtype=dtype)
    is_inf = numpy.logical_not(numpy.isfinite(inverse_mean))
    hmean = size / inverse_mean
    hmean[is_inf] = numpy.nan

    return hmean


def _calc_pairwise_dest(
    gts, pop_idxs, sorted_pop_ids, alleles, min_num_genotypes, ploidy
):
    debug = False

    num_pops = 2
    pop1, pop2 = sorted_pop_ids
    # print(alleles)

    res = _count_alleles_per_var(
        gts,
        pops=pop_idxs,
        calc_freqs=True,
        alleles=None,
        min_num_genotypes=min_num_genotypes,
    )
    allele_freq1 = res["counts"][pop1]["allelic_freqs"].values
    allele_freq2 = res["counts"][pop2]["allelic_freqs"].values

    exp_het1 = 1 - numpy.sum(allele_freq1**ploidy, axis=1)
    exp_het2 = 1 - numpy.sum(allele_freq2**ploidy, axis=1)
    hs_per_var = (exp_het1 + exp_het2) / 2
    if debug:
        print("hs_per_var", hs_per_var)

    global_allele_freq = (allele_freq1 + allele_freq2) / 2
    global_exp_het = 1 - numpy.sum(global_allele_freq**ploidy, axis=1)
    ht_per_var = global_exp_het
    if debug:
        print("ht_per_var", ht_per_var)

    res = _calc_obs_het_per_var_for_gts(gts, pops=pop_idxs)
    obs_het_per_var = res["obs_het_per_var"]
    obs_het1 = obs_het_per_var[pop1].values
    obs_het2 = obs_het_per_var[pop2].values
    if debug:
        print(f"{obs_het1=}")
        print(f"{obs_het2=}")
    called_gts_per_var = res["called_gts_per_var"]
    called_gts1 = called_gts_per_var[pop1]
    called_gts2 = called_gts_per_var[pop2]

    called_gts = numpy.array([called_gts1, called_gts2])
    try:
        called_gts_hmean = hmean(called_gts, axis=0)
    except ValueError:
        called_gts_hmean = None

    if called_gts_hmean is None:
        num_vars = gts.shape[0]
        corrected_hs = numpy.full((num_vars,), numpy.nan)
        corrected_ht = numpy.full((num_vars,), numpy.nan)
    else:
        mean_obs_het_per_var = numpy.nanmean(numpy.array([obs_het1, obs_het2]), axis=0)
        corrected_hs = (called_gts_hmean / (called_gts_hmean - 1)) * (
            hs_per_var - (mean_obs_het_per_var / (2 * called_gts_hmean))
        )
        if debug:
            print("mean_obs_het_per_var", mean_obs_het_per_var)
            print("corrected_hs", corrected_hs)
        corrected_ht = (
            ht_per_var
            + (corrected_hs / (called_gts_hmean * num_pops))
            - (mean_obs_het_per_var / (2 * called_gts_hmean * num_pops))
        )
        if debug:
            print("corrected_ht", corrected_ht)

        not_enough_gts = numpy.logical_or(
            called_gts1 < min_num_genotypes, called_gts2 < min_num_genotypes
        )
        corrected_hs[not_enough_gts] = numpy.nan
        corrected_ht[not_enough_gts] = numpy.nan

    num_vars_in_chunk = numpy.count_nonzero(~numpy.isnan(corrected_hs))
    hs_in_chunk = numpy.nansum(corrected_hs)
    ht_in_chunk = numpy.nansum(corrected_ht)
    return {"hs": hs_in_chunk, "ht": ht_in_chunk, "num_vars": num_vars_in_chunk}


class _DestPopDistCalculator:
    def __init__(self, pop_idxs, sorted_pop_ids, alleles, min_num_genotypes, ploidy):
        self.pop_idxs = pop_idxs
        self.pop_ids = sorted_pop_ids
        self.alleles = alleles
        self.min_num_genotypes = min_num_genotypes
        self.ploidy = ploidy

    def __call__(self, vars: Variants):
        pop_idxs = self.pop_idxs
        gts = vars.gt_array
        pop_ids = self.pop_ids
        num_pops = len(pop_ids)

        corrected_hs = pandas.DataFrame(
            numpy.zeros(shape=(num_pops, num_pops), dtype=float),
            columns=pop_ids,
            index=pop_ids,
        )
        corrected_ht = pandas.DataFrame(
            numpy.zeros(shape=(num_pops, num_pops), dtype=float),
            columns=pop_ids,
            index=pop_ids,
        )
        num_vars = pandas.DataFrame(
            numpy.zeros(shape=(num_pops, num_pops), dtype=int),
            columns=pop_ids,
            index=pop_ids,
        )

        for pop1, pop2 in itertools.combinations(self.pop_ids, 2):
            res = _calc_pairwise_dest(
                gts,
                sorted_pop_ids=(pop1, pop2),
                pop_idxs=pop_idxs,
                alleles=self.alleles,
                min_num_genotypes=self.min_num_genotypes,
                ploidy=self.ploidy,
            )
            corrected_hs.loc[pop1, pop2] = res["hs"]
            corrected_ht.loc[pop1, pop2] = res["ht"]
            num_vars.loc[pop1, pop2] = res["num_vars"]
            corrected_hs.loc[pop2, pop1] = res["hs"]
            corrected_ht.loc[pop2, pop1] = res["ht"]
            num_vars.loc[pop2, pop1] = res["num_vars"]
        return {"hs": corrected_hs, "ht": corrected_ht, "num_vars": num_vars}


def _accumulate_dest_results(accumulated_result, new_result):
    if accumulated_result is None:
        accumulated_hs = new_result["hs"]
        accumulated_ht = new_result["ht"]
        total_num_vars = new_result["num_vars"]
    else:
        accumulated_hs = accumulated_result["hs"] + new_result["hs"]
        accumulated_ht = accumulated_result["ht"] + new_result["ht"]
        total_num_vars = accumulated_result["num_vars"] + new_result["num_vars"]
    return {"hs": accumulated_hs, "ht": accumulated_ht, "num_vars": total_num_vars}


def calc_jost_dest_pop_distance(
    vars_iter: Iterator[Variants],
    pops: dict[list[str]],
    alleles: list[int] | None = None,
    min_num_genotypes=MIN_NUM_GENOTYPES_FOR_POP_STAT,
) -> Distances:
    """This is an implementation of the formulas proposed in GenAlex"""

    chunk, vars_iter = _peek_vars_iter(vars_iter)
    samples = chunk.samples
    ploidy = chunk.ploidy
    pop_idxs = _calc_pops_idxs(pops, samples)
    sorted_pop_ids = sorted(pop_idxs.keys())
    calc_dest_dists = _DestPopDistCalculator(
        pop_idxs=pop_idxs,
        sorted_pop_ids=sorted_pop_ids,
        alleles=alleles,
        min_num_genotypes=min_num_genotypes,
        ploidy=ploidy,
    )

    res = run_pipeline(
        vars_iter, map_functs=[calc_dest_dists], reduce_funct=_accumulate_dest_results
    )
    accumulated_hs = res["hs"]
    accumulated_ht = res["ht"]
    num_vars = res["num_vars"]

    tot_n_pops = len(pops)
    dists = numpy.empty(int((tot_n_pops**2 - tot_n_pops) / 2))
    dists[:] = numpy.nan
    num_pops = 2
    for idx, (pop_id1, pop_id2) in enumerate(itertools.combinations(sorted_pop_ids, 2)):
        with numpy.errstate(invalid="ignore"):
            corrected_hs = (
                accumulated_hs.loc[pop_id1, pop_id2] / num_vars.loc[pop_id1, pop_id2]
            )
            corrected_ht = (
                accumulated_ht.loc[pop_id1, pop_id2] / num_vars.loc[pop_id1, pop_id2]
            )
        dest = (num_pops / (num_pops - 1)) * (
            (corrected_ht - corrected_hs) / (1 - corrected_hs)
        )
        dists[idx] = dest
    dists = Distances(dists, sorted_pop_ids)
    return dists
