import pytest
import numpy
import pandas

from variants.array_iterator import Array, ArrayChunk, ArraysChunk, ArrayChunkIterator


def create_normal_numpy_array(shape, loc=0.0, scale=1.0):
    return numpy.random.default_rng().normal(loc=loc, scale=scale, size=shape)


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
