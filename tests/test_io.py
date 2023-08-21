import io
import gzip
import tempfile
import os

from variants import read_vcf, write_variants, read_variants, GT_ARRAY_ID
from variants.vars_io import read_vcf_metadata, VariantsDir
from .test_utils import get_big_vcf, get_vcf_sample, get_sample_variants


def test_vcf_reader():
    assert read_vcf_metadata(get_vcf_sample())["samples"] == [
        "NA00001",
        "NA00002",
        "NA00003",
    ]
    fhand = get_vcf_sample()
    variants = read_vcf(fhand)
    chunk = next(variants)
    assert chunk.num_rows == 5

    assert len(read_vcf_metadata(get_big_vcf())["samples"]) == 598

    for num_vars, fhand in ((5, get_vcf_sample()), (33790, get_big_vcf())):
        variants = read_vcf(fhand)
        total_num_variants = sum((chunk.num_rows for chunk in variants))
        assert total_num_variants == num_vars


def test_write_chunks():
    # write chunks with arrays

    variants = get_sample_variants()
    with tempfile.TemporaryDirectory() as dir:
        os.rmdir(str(dir))
        write_variants(str(dir), variants)

        variants_dir = VariantsDir(str(dir))
        assert variants_dir.samples == [
            "NA00001",
            "NA00002",
            "NA00003",
        ]
        assert variants_dir.num_variants == 5

        variants = variants_dir.iterate_over_variants(desired_arrays=[GT_ARRAY_ID])
        chunk = list(variants)[0]
        assert chunk.source_metadata["samples"] == ["NA00001", "NA00002", "NA00003"]
        assert list(chunk.arrays.keys()) == [GT_ARRAY_ID]
