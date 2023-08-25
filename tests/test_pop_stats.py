import numpy

from variants.variants import Variants, Genotypes
from variants.pop_stats import (
    _calc_obs_het_per_var_for_chunk,
    calc_obs_het_stats_per_var,
    _get_different_alleles,
    _count_alleles_per_var,
    _calc_maf_per_var_for_chunk,
    calc_major_allele_stats_per_var,
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
    assert _get_different_alleles(variants) == [0, 1, 3]


def test_count_alleles_per_var():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [0, 4], [0, 1], [1, 3]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    res = _count_alleles_per_var(gts, pops={0: [0, 1, 2, 3]}, calc_freqs=True)
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
    res = _calc_maf_per_var_for_chunk(chunk, pops={0: slice(None, None)})
    assert numpy.allclose(
        res["major_allele_freqs_per_var"][0].values,
        [0.5, 0.571429, numpy.nan],
        equal_nan=True,
    )

    variants = iter([chunk])
    res = calc_major_allele_stats_per_var(variants, hist_kwargs={"num_bins": 4})
    assert numpy.allclose(res["mean"].loc[0], [0.535714])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][0] == [0, 0, 2, 0])
