from pathlib import Path
import io
import gzip

import numpy
import pandas

from variants.iterators import ArraysChunk
from variants import read_vcf


def create_normal_numpy_array(shape, loc=0.0, scale=1.0):
    return numpy.random.default_rng().normal(loc=loc, scale=scale, size=shape)


def check_arrays_in_two_dicts_are_equal(arrays1: dict, arrays2: dict):
    arrays1 = arrays1.cargo
    arrays2 = arrays2.cargo

    assert not set(arrays1.keys()).difference(arrays2.keys())

    for id in arrays1.keys():
        array1 = arrays1[id]
        array2 = arrays2[id]
        assert type(array1) == type(array2)

        if isinstance(array1, numpy.ndarray):
            if numpy.issubdtype(array1.dtype, float):
                assert numpy.allclose(array1, array2)
            else:
                assert numpy.allequal(array1, array2)
        elif isinstance(array1, pandas.DataFrame):
            assert array1.equals(array2)
        else:
            ValueError()


def check_chunks_are_equal(chunks1, chunks2):
    for arrays1, arrays2 in zip(chunks1, chunks2):
        if isinstance(chunks1, ArraysChunk) and isinstance(chunks2, ArraysChunk):
            check_arrays_in_two_dicts_are_equal(arrays1, arrays2)
        else:
            ValueError()


def get_example_files_dir():
    return Path(__file__).parent / "example_files"


def get_big_vcf():
    return get_example_files_dir() / "tomato.vcf.gz"


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


def get_sample_variants():
    fhand = get_vcf_sample()
    variants = read_vcf(fhand)
    return variants
