import numpy
import pandas

from variants.iterators import ArraysChunk, ArrayChunk


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
        elif isinstance(chunks1, ArrayChunk) and isinstance(chunks2, ArrayChunk):
            check_arrays_in_two_dicts_are_equal(
                ArraysChunk({0: arrays1.cargo}), ArraysChunk({0: arrays2.cargo})
            )
        else:
            ValueError()
