from collections.abc import Iterator
import itertools

import numpy
import pandas
import scipy

from variants.variants import Variants
from variants.iterators import run_pipeline
from variants.globals import MISSING_INT


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
        gts = variations.genotypes.values
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


def calc_pairwise_kosman_dists(vars_iter: Iterator[Variants], min_num_snps=None):
    """It calculates the distance between individuals using the Kosman
    distance.

    The Kosman distance is explained in DOI: 10.1111/j.1365-294X.2005.02416.x
    """

    try:
        chunk = next(vars_iter)
    except StopIteration:
        raise ValueError("No vars to calculate distances")
    samples = chunk.samples
    vars_iter = itertools.chain([chunk], vars_iter)

    res = run_pipeline(
        vars_iter,
        map_functs=[_calc_kosman_dist_for_chunk],
        reduce_funct=_reduce_kosman_dists,
        reduce_initialializer=None,
    )
    abs_distances, n_snps_matrix = res

    if min_num_snps is not None:
        n_snps_matrix[n_snps_matrix < min_num_snps] = numpy.nan

    with numpy.errstate(invalid="ignore"):
        dists = abs_distances / n_snps_matrix

    d = len(samples)

    dists = Distances(dists, samples)
    return dists


# TODO
# calc_unbiased_nei_pop_dists
# calc_dest_pop_dists
