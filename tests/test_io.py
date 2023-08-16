import io
import gzip
from pathlib import Path
import tempfile
import os

from variants import read_vcf, write_variants, load_variants
from variants.iterators import ArrayChunk
from variants.vars_io import write_chunks, load_chunks
from .test_utils import create_normal_numpy_array, check_chunks_are_equal

VCF_SAMPLE = b"""##fileformat=VCFv4.0
##fileDate=20090805
##source=myImputationProgramV3.1
##reference=1000GenomesPilot-NCBI36
##phasing=partial
##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of Samples With Data">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##INFO=<ID=AF,Number=.,Type=Float,Description="Allele Frequency">
##INFO=<ID=AA,Number=1,Type=String,Description="Ancestral Allele">
##INFO=<ID=DB,Number=0,Type=Flag,Description="dbSNP membership, build 129">
##INFO=<ID=H2,Number=0,Type=Flag,Description="HapMap2 membership">
##FILTER=<ID=q10,Description="Quality below 10">
##FILTER=<ID=s50,Description="Less than 50% of samples have data">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##FORMAT=<ID=HQ,Number=2,Type=Integer,Description="Haplotype Quality">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	NA00001	NA00002	NA00003
20	14370	rs6054257	G	A	29	PASS	NS=3;DP=14;AF=0.5;DB;H2	GT:GQ:DP:HQ	0|0:48:1:51,51	1|0:48:8:51,51	1/1:43:5:.,.
20	17330	.	T	A	3	q10	NS=3;DP=11;AF=0.017	GT:GQ:DP:HQ	0|0:49:3:58,50	0|1:3:5:65,3	0/0:41:3
20	1110696	rs6040355	A	G,T	67	PASS	NS=2;DP=10;AF=0.333,0.667;AA=T;DB	GT:GQ:DP:HQ	1|2:21:6:23,27	2|1:2:0:18,2	2/2:35:4
20	1230237	.	T	.	47	PASS	NS=3;DP=13;AA=T	GT:GQ:DP:HQ	0|0:54:7:56,60	0|0:48:4:51,51	0/0:61:2
20	1234567	microsat1	GTCT	G,GTACT	50	PASS	NS=3;DP=9;AA=G	GT:GQ:DP	0/1:35:4	0/2:17:2	1/1:40:3"""


def get_vcf_sample():
    fhand = io.BytesIO(gzip.compress(VCF_SAMPLE))
    return fhand


def get_example_files_dir():
    return Path(__file__).parent / "example_files"


def get_big_vcf():
    return get_example_files_dir() / "tomato.vcf.gz"


def get_sample_variants():
    fhand = get_vcf_sample()
    variants = read_vcf(fhand)
    return variants


def test_vcf_reader():
    fhand = get_vcf_sample()
    variants = read_vcf(fhand)
    assert variants.samples == ["NA00001", "NA00002", "NA00003"]
    chunk = next(variants)
    assert chunk.num_rows == 5

    variants = read_vcf(get_big_vcf())
    assert len(variants.samples) == 598

    for num_vars, fhand in ((5, get_vcf_sample()), (33790, get_big_vcf())):
        variants = read_vcf(fhand)
        list(variants)
        assert variants.num_vars_processed == num_vars


def test_write_chunks():
    # write chunks with arrays
    variants = get_sample_variants()
    with tempfile.TemporaryDirectory() as dir:
        os.rmdir(str(dir))
        write_variants(str(dir), variants)
        variants = load_variants(dir)
        assert variants.samples == ["NA00001", "NA00002", "NA00003"]
        assert variants.num_vars_expected == 5

    ndarray_2d = create_normal_numpy_array(shape=(10, 5))
    chunk = ArrayChunk(ndarray_2d)
    with tempfile.TemporaryDirectory() as dir:
        os.rmdir(str(dir))
        write_chunks(str(dir), iter([chunk]))

        chunks = load_chunks(str(dir))
        check_chunks_are_equal(list(chunks), [chunk])
