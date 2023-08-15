from collections.abc import Iterator

import numpy
import pandas


ArrayType = tuple[numpy.ndarray, pandas.DataFrame, pandas.Series]


class Array:
    def __init__(self, array: ArrayType):
        if not isinstance(array, (numpy.ndarray, pandas.DataFrame, pandas.Series)):
            raise ValueError("Non supported type for array")
        self.array = array

    @property
    def num_rows(self):
        array = self.array
        return array.shape[0]


class Chunk:
    @property
    def num_rows(self):
        raise NotImplementedError()


def _as_array(array):
    if isinstance(array, Array):
        return array
    else:
        return Array(array)


def _as_chunk(chunk):
    if isinstance(chunk, Chunk):
        return chunk
    if isinstance(chunk, dict):
        return ArraysChunk(chunk)
    return ArrayChunk(chunk)


class ArrayChunk(Chunk):
    def __init__(self, cargo: Array | ArrayType):
        self.cargo = _as_array(cargo)

    @property
    def num_rows(self):
        return self.cargo.num_rows


class ArraysChunk(Chunk):
    def __init__(self, cargo: dict[Array]):
        if not cargo:
            raise ValueError("At least one array should be given")

        num_rows = None
        revised_cargo = {}
        for id, array in cargo.items():
            array = _as_array(array)
            this_num_rows = array.num_rows
            if num_rows is None:
                num_rows = this_num_rows
            else:
                if num_rows != this_num_rows:
                    raise ValueError("All arrays should have the same number of rows")
            revised_cargo[id] = array

        self._num_rows = num_rows
        self.cargo = revised_cargo

    @property
    def num_rows(self):
        return self._num_rows


class ArrayChunkIterator(Iterator[Chunk]):
    def __init__(
        self, chunks: Iterator[Chunk], expected_total_num_rows: int | None = None
    ):
        self._expected_rows = expected_total_num_rows
        self._chunks = chunks
        self._num_rows_processed = 0

    @property
    def num_rows_expected(self):
        return self._expected_rows

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = _as_chunk(next(self._chunks))
            self._num_rows_processed += chunk.num_rows
        except StopIteration:
            if (
                self._expected_rows is not None
                and self._num_rows_processed != self._expected_rows
            ):
                raise ValueError(
                    f"error, {self._expected_rows} were expected, but {self._num_rows_processed} have been processed"
                )

            raise StopIteration
        return chunk
