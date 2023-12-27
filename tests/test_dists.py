import warnings

with warnings.catch_warnings(action="ignore"):
    import pandas
import numpy

from .test_utils import get_big_vcf
from variants import read_vcf
from variants.variants import Genotypes, Variants
from variants.globals import (
    GT_ARRAY_ID,
    VARIANTS_ARRAY_ID,
    CHROM_VARIANTS_COL,
    POS_VARIANTS_COL,
)


from variants.iterators import take_n_variants
from variants.distances import (
    calc_pairwise_kosman_dists,
    _kosman,
    _IndiPairwiseCalculator,
    calc_jost_dest_pop_distance,
    calc_jost_dest_dist_between_pops_per_var,
)
from variants.pop_stats import get_different_alleles


def test_pairwise_kosman_dists():
    num_vars_to_take = 100
    vars = read_vcf(get_big_vcf(), num_variants_per_chunk=10)
    vars = take_n_variants(vars, num_vars_to_take)

    dists = calc_pairwise_kosman_dists(vars).square_dists

    assert numpy.isclose([dists.loc["bgv015730", "tegucigalpa"]], [0])
    assert numpy.isclose([dists.loc["tegucigalpa", "pi378994"]], [0.440000])
    assert dists.shape == (598, 598)

    num_vars_to_take = 100
    vars = read_vcf(get_big_vcf(), num_variants_per_chunk=10)
    vars = take_n_variants(vars, num_vars_to_take)

    dists2 = calc_pairwise_kosman_dists(vars, num_processes=1).square_dists
    assert numpy.allclose(dists, dists2)


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


def test_dest_pop_dists():
    num_vars_to_take = 100
    vars = read_vcf(get_big_vcf(), num_variants_per_chunk=10)
    vars = take_n_variants(vars, num_vars_to_take)
    pops = {
        "pop1": [
            "ts-63",
            "ts-168",
            "ts-178",
            "ts-313",
            "ts-427",
            "ts-10",
            "ts-1",
            "ts-620",
            "bgv007981",
            "stupicke",
        ],
        "pop2": [
            "ts-160",
            "ts-48",
            "ts-437",
            "ts-41",
            "ts-243",
            "ts-520",
            "ts-555",
            "ts-630",
            "ts-576",
            "bgv014515",
            "bgv006454",
        ],
    }
    alleles = get_different_alleles(
        vars,
        samples_to_consider=[sample for samples in pops.values() for sample in samples],
    )

    vars = read_vcf(get_big_vcf(), num_variants_per_chunk=10)
    vars = take_n_variants(vars, num_vars_to_take)
    dists = calc_jost_dest_pop_distance(
        vars, pops, alleles=alleles, min_num_genotypes=0
    )
    assert numpy.allclose(dists.dist_vector, [0.00251768])


def test_dest_jost_distance():
    gts = [
        [  #          sample pop is_het tot_het freq_het
            (1, 1),  #    1     1
            (1, 3),  #    2     1     1
            (1, 2),  #    3     1     1
            (1, 4),  #    4     1     1
            (3, 3),  #    5     1             3     3/5=0.6
            (3, 2),  #    6     2     1
            (3, 4),  #    7     2     1
            (2, 2),  #    8     2
            (2, 4),  #    9     2     1
            (4, 4),  #   10     2
            (-1, -1),  # 11     2             3     3/6=0.5
        ],
        [
            (1, 3),
            (1, 1),
            (1, 1),
            (1, 3),
            (3, 3),
            (3, 2),
            (3, 4),
            (2, 2),
            (2, 4),
            (4, 4),
            (-1, -1),
        ],
    ]
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    gts = Genotypes(numpy.array(gts), samples=samples)
    variants_info = pandas.DataFrame(
        {CHROM_VARIANTS_COL: ["chrom1", "chrom2"], POS_VARIANTS_COL: [1000, 500]}
    )
    snps = Variants(arrays={GT_ARRAY_ID: gts, VARIANTS_ARRAY_ID: variants_info})

    pop1 = [1, 2, 3, 4, 5]
    pop2 = [6, 7, 8, 9, 10, 11]
    pops = {"pop1": pop1, "pop2": pop2}

    dists = calc_jost_dest_dist_between_pops_per_var(
        iter([snps]), pop1=pop1, pop2=pop2, min_num_genotypes=0
    )
    print(list(dists))
    assert numpy.allclose(list(dists)[0], [0.490909, 0.779310])

    snps = Variants(arrays={GT_ARRAY_ID: gts})
    dists = calc_jost_dest_dist_between_pops_per_var(
        iter([snps]), pop1=pop1, pop2=pop2, min_num_genotypes=0
    )
    assert numpy.allclose(list(dists)[0], [0.65490196, 0.65490196])

    alleles = get_different_alleles(iter([snps]))

    dists = calc_jost_dest_pop_distance(iter([snps]), pops=pops, min_num_genotypes=0)
    assert numpy.allclose(dists.dist_vector, [0.65490196])

    dists = calc_jost_dest_pop_distance(
        iter([snps]), pops=pops, alleles=alleles, min_num_genotypes=6
    )
    assert numpy.all(numpy.isnan(dists.dist_vector))
