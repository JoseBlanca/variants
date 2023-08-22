from .test_utils import get_sample_variants

from variants import VariantFilterer


def test_filter_vars():
    experiments = [
        (1.0, 1.0, [5], {}),
        (-0.01, 1.0, [], {"missing_rate": 5}),
        (1.0, 0.5, [3], {"obs_het": 2}),
        (0.5, 0.9, [5], {"missing_rate": 0, "obs_het": 0}),
    ]

    for max_missing_rate, max_obs_het, expected_num_rows, remove_stats in experiments:
        vars = get_sample_variants()
        filterer = VariantFilterer(
            max_missing_rate=max_missing_rate, max_var_obs_het=max_obs_het
        )
        flt_vars = list(filterer(vars))
        assert expected_num_rows == [chunk.num_rows for chunk in flt_vars]
        assert filterer.stats["num_vars_removed_per_filter"] == remove_stats


def test_filter_by_region():
    vars = get_sample_variants()
    filterer = VariantFilterer(regions_to_keep=[("20", 10000, 20000)])
    flt_vars = list(filterer(vars))
    flt_vars[0]["variants"]["pos"] == [14270, 17330]
    assert filterer.stats["num_vars_removed_per_filter"] == {"desired_region": 3}
