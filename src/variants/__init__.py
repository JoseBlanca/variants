from .iterators import VariantsIterator
from .vars_io import (
    read_vcf,
    write_variants,
    read_variants,
    GT_ARRAY_ID,
    VARIANTS_ARRAY_ID,
    ALLELES_ARRAY_ID,
    ORIG_VCF_ARRAY_ID,
)
from .pop_stats import calc_obs_het_stats_per_var
