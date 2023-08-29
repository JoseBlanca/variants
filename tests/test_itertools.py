import numpy
import pandas

from variants.iterators import (
    ArraysChunk,
    resize_chunks,
    take_n_variants,
    _concatenate_chunks,
    get_samples_from_chunk,
    VariantsCounter,
    group_in_genomic_windows,
    group_in_chroms,
    sample_n_vars,
    sample_n_vars_per_genomic_window,
)
from .test_utils import create_normal_numpy_array, get_big_vcf, get_sample_variants
from variants import read_vcf
from variants.globals import VARIANTS_ARRAY_ID


def test_chunk():
    ndarray_2d = create_normal_numpy_array(shape=(10, 5))
    pandas_dframe = pandas.DataFrame(ndarray_2d)
    ndarray_1d = create_normal_numpy_array(shape=(10,))

    for array in [ndarray_2d, pandas_dframe, ndarray_1d]:
        chunk = ArraysChunk({0: array})
        assert chunk.num_rows == 10


def test_resize_chunks():
    ndarray_2d = create_normal_numpy_array(shape=(10, 5))

    for n_rows, expected_rows in [
        (15, [15, 5]),
        (3, [3, 3, 3, 3, 3, 3, 2]),
        (10, [10, 10]),
        (30, [20]),
    ]:
        chunks = iter([ArraysChunk({1: ndarray_2d}), ArraysChunk({1: ndarray_2d})])
        chunks = resize_chunks(chunks, n_rows)
        assert [chunk.num_rows for chunk in chunks] == expected_rows


def test_to_mem():
    ndarray_2d = create_normal_numpy_array(shape=(10, 5))

    expected = numpy.vstack([ndarray_2d, ndarray_2d])

    chunks = iter([{1: ndarray_2d}, {1: ndarray_2d}])
    arrays = _concatenate_chunks(chunks)
    assert numpy.allclose(arrays[1], expected)


def test_take_n():
    for num_variants, len_chunks in [(499, [499]), (500, [500]), (501, [500, 1])]:
        variants = read_vcf(get_big_vcf(), num_variants_per_chunk=500)
        variants = take_n_variants(variants, num_variants)
        variants = list(variants)
        assert len(get_samples_from_chunk(variants[0])) == 598
        assert len(variants) == len(len_chunks)
        assert [chunk.num_rows for chunk in variants] == len_chunks


def test_var_counter():
    vars = get_sample_variants()
    counter = VariantsCounter()
    vars = counter(vars)
    list(vars)
    assert counter.num_vars


def test_chrom_grouper():
    num_vars_to_take = 10000
    vars = read_vcf(get_big_vcf(), num_variants_per_chunk=1000)
    vars = take_n_variants(vars, num_vars_to_take)
    grouped_vars = group_in_chroms(vars)

    expected_chunk_lens = [
        [1000, 1000, 1000, 520],
        [1000, 1000, 1000, 1000, 356],
        [1000, 1000, 124],
    ]
    expected_chroms = ["SL4.0ch01", "SL4.0ch02", "SL4.0ch03"]
    total_num_vars = 0
    for group_idx, group_of_chunks in enumerate(grouped_vars):
        this_group_different_chroms = set()
        this_group_num_vars_per_chunk = []
        for chunk in group_of_chunks:
            this_chunk_different_chroms = numpy.unique(
                chunk.arrays[VARIANTS_ARRAY_ID]["chrom"]
            )
            assert len(set(this_chunk_different_chroms)) == 1
            this_group_different_chroms.add(this_chunk_different_chroms[0])
            this_group_num_vars_per_chunk.append(chunk.num_rows)
            total_num_vars += chunk.num_rows
        assert len(this_chunk_different_chroms) == 1
        assert this_group_num_vars_per_chunk == expected_chunk_lens[group_idx]
        assert this_chunk_different_chroms[0] == expected_chroms[group_idx]
    assert total_num_vars == num_vars_to_take


def test_win_grouper():
    for num_vars_per_chunk in [1, 2, 3, 4, 5]:
        vars = get_sample_variants(num_variants_per_chunk=num_vars_per_chunk)
        grouped_vars = group_in_genomic_windows(vars, 20000)
        expected_poss = [[14370, 17330], [1110696], [1230237, 1234567]]
        for group_idx, group_of_chunks in enumerate(grouped_vars):
            poss_in_group = []
            for chunk in group_of_chunks:
                poss_in_group.extend(chunk.arrays[VARIANTS_ARRAY_ID]["pos"])
            assert poss_in_group == expected_poss[group_idx]

    num_vars_to_take = 10000
    vars = read_vcf(get_big_vcf(), num_variants_per_chunk=1000)
    vars = take_n_variants(vars, num_vars_to_take)
    grouped_vars = group_in_genomic_windows(vars, 50000000)

    expected_chunk_lens = [
        [477],
        [1000, 1000, 1000, 43],
        [1000, 1000, 1000, 1000, 63],
        [293],
        [1000, 29],
        [1000, 95],
    ]
    expected_chroms = [
        "SL4.0ch01",
        "SL4.0ch01",
        "SL4.0ch02",
        "SL4.0ch02",
        "SL4.0ch03",
        "SL4.0ch03",
    ]
    total_num_vars = 0
    for group_idx, group_of_chunks in enumerate(grouped_vars):
        this_group_different_chroms = set()
        this_group_num_vars_per_chunk = []
        for chunk in group_of_chunks:
            this_chunk_different_chroms = numpy.unique(
                chunk.arrays[VARIANTS_ARRAY_ID]["chrom"]
            )
            assert len(set(this_chunk_different_chroms)) == 1
            this_group_different_chroms.add(this_chunk_different_chroms[0])
            this_group_num_vars_per_chunk.append(chunk.num_rows)
            total_num_vars += chunk.num_rows
        assert len(this_chunk_different_chroms) == 1
        assert this_chunk_different_chroms[0] == expected_chroms[group_idx]
        assert this_group_num_vars_per_chunk == expected_chunk_lens[group_idx]

    assert total_num_vars == num_vars_to_take


def test_sample_vars_at_random():
    vars = get_sample_variants()
    num_vars_to_sample = 2
    chunk = sample_n_vars(vars, num_vars_to_sample)
    assert chunk.num_rows == num_vars_to_sample

    for num_vars_to_sample in [49, 50, 100]:
        vars = read_vcf(get_big_vcf(), num_variants_per_chunk=50)
        vars = take_n_variants(vars, 200)
        vars = sample_n_vars(vars, num_vars_to_sample)
        assert vars.num_rows == num_vars_to_sample


def test_window_sampling():
    for chunk_len in [200, 500, 1000, 10000]:
        vars = read_vcf(get_big_vcf(), num_variants_per_chunk=chunk_len)
        vars = read_vcf(get_big_vcf(), num_variants_per_chunk=100)
        vars = read_vcf(get_big_vcf(), num_variants_per_chunk=500)
        vars = read_vcf(get_big_vcf(), num_variants_per_chunk=1000)
        num_vars_to_take = 10000
        vars = take_n_variants(vars, num_vars_to_take)
        counter = VariantsCounter()
        vars = counter(vars)
        sampled_vars = sample_n_vars_per_genomic_window(
            vars, win_len_in_bp=100000, num_vars_to_take=1
        )
        counter2 = VariantsCounter()
        sampled_vars = counter2(sampled_vars)
        sampled_vars = list(sampled_vars)
        assert counter.num_vars == num_vars_to_take
        assert counter2.num_vars == 516
