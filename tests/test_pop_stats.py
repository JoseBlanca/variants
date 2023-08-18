import numpy

from variants.pop_stats import (
    _calc_obs_het_per_var_for_chunk,
    calc_obs_het_stats_per_var,
)
from variants.iterators import ArraysChunk, VariantsIterator
from variants.vars_io import GT_ARRAY_ID


def test_obs_het_per_var():
    gts = numpy.array(
        [
            # sample1 sample2 sample3 sample4
            [[0, 0], [1, 1], [-1, -1], [0, 1]],  # snp1
            [[-1, 1], [0, 0], [0, 1], [1, 0]],  # snp2
            [[-1, -1], [-1, -1], [-1, -1], [-1, -1]],  # snp3
        ]
    )
    gts = ArraysChunk({GT_ARRAY_ID: gts})
    res = _calc_obs_het_per_var_for_chunk(gts, pops={0: slice(None, None)})
    obs_het_per_var = res["obs_het_per_var"][0].values
    numpy.allclose(obs_het_per_var, [0.33333333, 0.66666667, numpy.nan])

    variants = VariantsIterator([gts], samples=[1, 2, 3, 4])
    res = calc_obs_het_stats_per_var(variants, hist_kwargs={"num_bins": 4})
    assert numpy.allclose(res["mean"].loc[0], [0.5])
    assert numpy.allclose(res["hist_bin_edges"], [0.0, 0.25, 0.5, 0.75, 1.0])
    assert all(res["hist_counts"][0] == [0, 1, 1, 0])
