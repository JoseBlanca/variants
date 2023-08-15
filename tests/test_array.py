import pytest
import numpy
import pandas

from variants.array_iterator import (
    Array,
    ArrayChunk,
    ArraysChunk,
    ArrayChunkIterator,
    resize_chunks,
)
from .test_utils import create_normal_numpy_array


def test_array():
    ndarray_2d = create_normal_numpy_array(shape=(10, 5))
    pandas_dframe = pandas.DataFrame(ndarray_2d)
    ndarray_1d = create_normal_numpy_array(shape=(10,))

    for array in [ndarray_2d, pandas_dframe, ndarray_1d]:
        assert Array(array).num_rows == 10


def test_chunk():
    ndarray_2d = create_normal_numpy_array(shape=(10, 5))
    chunk = ArrayChunk(ndarray_2d)
    assert chunk.num_rows == 10

    chunk = ArraysChunk({1: ndarray_2d})
    assert chunk.num_rows == 10


def test_chunk_iterator():
    ndarray_2d = create_normal_numpy_array(shape=(10, 5))
    chunks = iter([ndarray_2d, ndarray_2d])
    chunks = len(list(ArrayChunkIterator(chunks))) == 2

    chunks = iter([ndarray_2d, ndarray_2d])
    with pytest.raises(ValueError):
        list(ArrayChunkIterator(chunks, expected_total_num_rows=21))

    chunks = iter([ndarray_2d, ndarray_2d])
    chunks = ArrayChunkIterator(chunks, expected_total_num_rows=20)
    assert chunks.num_rows_expected == 20
    list(chunks)


def test_resize_chunks():
    ndarray_2d = create_normal_numpy_array(shape=(10, 5))

    for n_rows, expected_rows in [
        (15, [15, 5]),
        (3, [3, 3, 3, 3, 3, 3, 2]),
        (10, [10, 10]),
        (30, [20]),
    ]:
        chunks = ArrayChunkIterator([ndarray_2d, ndarray_2d])
        chunks2 = ArrayChunkIterator([{1: ndarray_2d}, {1: ndarray_2d}])
        for chunks_ in [chunks, chunks2]:
            chunks_ = resize_chunks(chunks_, n_rows)
            assert [chunk.num_rows for chunk in chunks_] == expected_rows
