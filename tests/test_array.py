import numpy
import pandas

from variants.iterators import (
    ArraysChunk,
    resize_chunks,
    take_n_variants,
    _concatenate_chunks,
    get_samples_from_chunk,
)
from .test_utils import create_normal_numpy_array, get_big_vcf
from variants import read_vcf


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
