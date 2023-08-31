import numpy

from variants.variants import Variants, Genotypes
from variants.pop_stats import (
    _calc_obs_het_per_var_for_chunk,
    _count_alleles_per_var,
    _calc_maf_per_var_for_chunk,
    _calc_expected_het_per_snp,
    _calc_unbiased_exp_het_per_snp,
    calc_obs_het_stats_per_var,
    get_different_alleles,
    calc_major_allele_stats_per_var,
    calc_exp_het_stats_per_var,
)
from variants.vars_io import GT_ARRAY_ID


def test_obs_het_stats():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [0, 0], [0, 1], [1, 0]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    chunk = Variants({GT_ARRAY_ID: Genotypes(gts, samples=[1, 2, 3, 4])})

    res = _calc_obs_het_per_var_for_chunk(chunk, pops={0: slice(None, None)})
    obs_het_per_var = res["obs_het_per_var"][0].values
    numpy.allclose(obs_het_per_var, [0.33333333, 0.66666667, numpy.nan])

    variants = iter([chunk])
    res = calc_obs_het_stats_per_var(variants, hist_kwargs={"num_bins": 4})
    assert numpy.allclose(res["mean"].loc[0], [0.5])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][0] == [0, 1, 1, 0])


def test_obs_het_per_var():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [0, 0], [0, 1], [1, 3]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    chunk = Variants({GT_ARRAY_ID: Genotypes(gts, samples=[1, 2, 3, 4])})
    variants = iter([chunk])
    assert get_different_alleles(variants) == [0, 1, 3]


def test_count_alleles_per_var():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [0, 4], [0, 1], [1, 3]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    res = _count_alleles_per_var(
        gts, pops={0: [0, 1, 2, 3]}, calc_freqs=True, min_num_genotypes=1
    )
    assert res["alleles"] == {0, 1, 3, 4}
    assert numpy.all(res["counts"][0]["missing_gts_per_var"] == [2, 1, 8])
    expected_counts = [3, 3, 0, 0], [2, 3, 1, 1], [0, 0, 0, 0]
    assert numpy.all(res["counts"][0]["allele_counts"].values == expected_counts)
    expected_freqs = [
        [
            0.5,
            0.5,
            0.0,
            0.0,
        ],
        [0.28571429, 0.42857143, 0.14285714, 0.14285714],
        [numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    ]
    assert numpy.allclose(
        res["counts"][0]["allelic_freqs"].values, expected_freqs, equal_nan=True
    )

    res = _count_alleles_per_var(
        gts, pops={0: [0, 1, 2, 3]}, calc_freqs=True, min_num_genotypes=3.1
    )
    assert res["alleles"] == {0, 1, 3, 4}
    assert numpy.all(res["counts"][0]["missing_gts_per_var"] == [2, 1, 8])
    expected_counts = [3, 3, 0, 0], [2, 3, 1, 1], [0, 0, 0, 0]
    assert numpy.all(res["counts"][0]["allele_counts"].values == expected_counts)
    expected_freqs = [
        [numpy.nan, numpy.nan, numpy.nan, numpy.nan],
        [0.28571429, 0.42857143, 0.14285714, 0.14285714],
        [numpy.nan, numpy.nan, numpy.nan, numpy.nan],
    ]
    assert numpy.allclose(
        res["counts"][0]["allelic_freqs"].values, expected_freqs, equal_nan=True
    )


def test_maf_stats():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [1, 0], [0, 1], [1, 0]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    chunk = Variants({GT_ARRAY_ID: Genotypes(gts, samples=[1, 2, 3, 4])})
    res = _calc_maf_per_var_for_chunk(
        chunk, pops={0: slice(None, None)}, min_num_genotypes=1
    )
    assert numpy.allclose(
        res["major_allele_freqs_per_var"][0].values,
        [0.5, 0.57142857, numpy.nan],
        equal_nan=True,
    )

    variants = iter([chunk])
    res = calc_major_allele_stats_per_var(
        variants, hist_kwargs={"num_bins": 4}, min_num_genotypes=1
    )
    assert numpy.allclose(res["mean"].loc[0], [0.535714])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][0] == [0, 0, 2, 0])


def test_calc_exp_het():
    gts = numpy.array(
        [
            [[0, 0], [2, 1], [0, 0], [0, 0], [0, -1]],
            [[0, 0], [0, 0], [0, 1], [1, 0], [-1, -1]],
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]],
        ]
    )
    #       allele counts  allele freqs            exp hom                          exp het      u exp het
    #       pop1    pop2   pop1         pop2       pop1               pop2          pop1   pop2  pop1   pop2
    # snp1  2 1 1   5 0 0  2/4 1/4 1/4  5/5 0   0  0.25 0.0625 0.0625 1    0    0   0.625  0     0.8333 0
    # snp2  4 0 0   2 2 0  4/4 0   0    2/4 2/4 0  1    0      0      0.25 0.25 0   0      0.5   0      0.6666
    # snp3  0 0 0   0 0 0  nan nan nan  nan nan nan nan nan    nan    nan  nan  nan nan    nan   nan    nan
    pops = {1: [0, 1], 2: [2, 3, 4]}
    res = _calc_expected_het_per_snp(gts, pops, min_num_genotypes=1)
    expected = [[0.625, 0.0], [0.0, 0.5], [numpy.nan, numpy.nan]]
    assert numpy.allclose(res["exp_het"].values, expected, equal_nan=True)
    expected = [[0, 1], [0, 2], [4, 6]]
    assert numpy.all(res["missing_allelic_gts"].values == expected)

    exp_het = _calc_unbiased_exp_het_per_snp(gts, pops, min_num_genotypes=1)
    expected = [[0.83333333, 0.0], [0.0, 0.66666667], [numpy.nan, numpy.nan]]
    assert numpy.allclose(exp_het.values, expected, equal_nan=True)

    chunk = Variants({GT_ARRAY_ID: Genotypes(gts, samples=[1, 2, 3, 4, 5])})
    variants = iter([chunk])
    res = calc_exp_het_stats_per_var(
        variants, hist_kwargs={"num_bins": 4}, min_num_genotypes=1
    )
    assert numpy.allclose(res["mean"].loc[0], [0.422619])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][0] == [0, 2, 0, 0])
