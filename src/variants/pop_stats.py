
from variants.iterators import VariantsIterator, ArrayChunk

def _calc_obs_het_per_var_for_chunk(chunk, pops):
    gts = chunk[GT_ARRAY_ID]
    gt_is_missing = _calc_gt_is_missing(gts)
    gt_is_het = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))
    gt_is_het = _calc_gts_is_het(gts, gt_is_missing=gt_is_missing)

    obs_het_per_var = {}
    for pop_id, pop_slice in pops.items():
        num_vars_het_per_var = numpy.sum(gt_is_het[:, pop_slice], axis=1)
        num_non_missing_per_var = gts.shape[0] - numpy.sum(
            gt_is_missing[:, pop_slice], axis=1
        )
        obs_het_per_var[pop_id] = num_vars_het_per_var / num_non_missing_per_var

    obs_het_per_var = pandas.DataFrame(obs_het_per_var)
    return {"obs_het_per_var": obs_het_per_var}


def calc_obs_het(variants:VariantsIterator, pops) -> ArrayChunk:
    for chunk in variants: