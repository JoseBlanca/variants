import numpy

from .test_utils import get_big_vcf
from variants import read_vcf
from variants.variants import Genotypes, Variants
from variants.globals import GT_ARRAY_ID
from variants.iterators import take_n_variants
from variants.distances import (
    calc_pairwise_kosman_dists,
    _kosman,
    _IndiPairwiseCalculator,
)


def test_pairwise_kosman_dists():
    num_vars_to_take = 100
    vars = read_vcf(get_big_vcf(), num_variants_per_chunk=10)
    vars = take_n_variants(vars, num_vars_to_take)

    dists = calc_pairwise_kosman_dists(vars).square_dists

    assert numpy.isclose([dists.loc["bgv015730", "tegucigalpa"]], [0])
    assert numpy.isclose([dists.loc["tegucigalpa", "pi378994"]], [0.440000])
    assert dists.shape == (598, 598)


def test_kosman_2_indis():
    a = numpy.array(
        [
            [-1, -1],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    b = numpy.array(
        [
            [1, 1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    gts = numpy.stack((a, b), axis=1)
    abs_distance, n_snps = _kosman(gts, 0, 1, {})
    distance = abs_distance / n_snps
    assert distance == 1 / 3

    c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    gts = numpy.stack((c, d), axis=1)
    abs_distance, n_snps = _kosman(gts, 0, 1, {})
    distance = abs_distance / n_snps
    assert distance == 0

    gts = numpy.stack((b, d), axis=1)
    abs_distance, n_snps = _kosman(gts, 0, 1, {})
    distance = abs_distance / n_snps
    assert distance == 0.45


def test_kosman_missing():
    a = numpy.array(
        [
            [-1, -1],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    b = numpy.array(
        [
            [1, 1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    gts = numpy.stack((a, b), axis=1)
    abs_distance, n_snps = _kosman(gts, 0, 1, {})
    distance_ab = abs_distance / n_snps

    a = numpy.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    b = numpy.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )
    gts = numpy.stack((a, b), axis=1)
    abs_distance, n_snps = _kosman(gts, 0, 1, {})
    distance_cd = abs_distance / n_snps

    assert distance_ab == distance_cd


def test_kosman_pairwise():
    a = numpy.array(
        [
            [-1, -1],
            [0, 0],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
        ]
    )
    b = numpy.array(
        [
            [1, 1],
            [-1, -1],
            [0, 0],
            [0, 0],
            [1, 1],
            [0, 1],
            [1, 0],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 2],
        ]
    )
    c = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    d = numpy.full(shape=(11, 2), fill_value=1, dtype=numpy.int16)
    gts = numpy.stack((a, b, c, d), axis=0)
    gts = numpy.transpose(gts, axes=(1, 0, 2)).astype(numpy.int16)
    gts = Genotypes(gts, [1, 2, 3, 4])
    vars = Variants({GT_ARRAY_ID: gts})

    pairwise_dist_calculator = _IndiPairwiseCalculator()
    abs_dist, n_snps = pairwise_dist_calculator.calc_dist(vars, method="kosman")
    distance = abs_dist / n_snps
    expected = [0.33333333, 0.75, 0.75, 0.5, 0.5, 0.0]
    assert numpy.allclose(distance, expected)
