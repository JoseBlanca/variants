from collections.abc import Iterator
from collections import defaultdict

import numpy
import pandas


ArrayType = tuple[numpy.ndarray, pandas.DataFrame, pandas.Series]


class Array:
    def __init__(self, array: ArrayType):
        if isinstance(array, Array):
            array = array.array
        if not isinstance(array, (numpy.ndarray, pandas.DataFrame, pandas.Series)):
            raise ValueError("Non supported type for array")
        self.array = array

    @property
    def num_rows(self):
        array = self.array
        return array.shape[0]

    def get_rows(self, index):
        array = self.array
        if isinstance(array, numpy.ndarray):
            array = array[index, ...]
        elif isinstance(array, pandas.DataFrame):
            array = array.iloc[index, :]
        elif isinstance(array, pandas.Series):
            array = array.iloc[index]
        return Array(array)


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

    def get_rows(self, index):
        return ArrayChunk(self.cargo.get_rows(index))


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

    def items(self):
        return self.cargo.items()

    def get_rows(self, index):
        result = {}
        for id, array in self.items():
            result[id] = array.get_rows(index)

        return ArraysChunk(result)


class ArrayChunkIterator(Iterator[Chunk]):
    def __init__(
        self, chunks: Iterator[Chunk], expected_total_num_rows: int | None = None
    ):
        self._expected_rows = expected_total_num_rows
        self._chunks = iter(chunks)
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


def concatenate_arrays(arrays: list[Array]) -> Array:
    if isinstance(arrays[0], Array):
        arrays = [array.array for array in arrays]

    if isinstance(arrays[0], numpy.ndarray):
        array = numpy.vstack(arrays)
    elif isinstance(arrays[0], pandas.DataFrame):
        array = pandas.concat(arrays, axis=0)
    elif isinstance(arrays[0], pandas.Series):
        array = pandas.concat(arrays)
    else:
        raise ValueError("unknown type for array: " + str(type(arrays[0])))
    return Array(array)


def _concatenate_arrays_chunks(chunks) -> ArraysChunk:
    arrays_to_concatenate = defaultdict(list)
    for chunk in chunks:
        for array_id, array in chunk.items():
            arrays_to_concatenate[array_id].append(array)

    num_arrays = [len(arrays) for arrays in arrays_to_concatenate.values()]
    if not all([num_arrays[0] == len_ for len_ in num_arrays]):
        raise ValueError("Nota all chunks have the same arrays")

    concatenated_chunk = {}
    for array_id, arrays in arrays_to_concatenate.items():
        concatenated_chunk[array_id] = concatenate_arrays(arrays)
    concatenated_chunk = ArraysChunk(concatenated_chunk)
    return concatenated_chunk


def concatenate_chunks(chunks: list[Chunk]):
    if len(chunks) == 1:
        return chunks[0]

    if isinstance(chunks[0], ArraysChunk):
        return _concatenate_arrays_chunks(chunks)
    else:
        return concatenate_arrays([chunk.cargo for chunk in chunks])


def _get_num_rows_in_chunk(buffered_chunk):
    if not buffered_chunk:
        return 0
    else:
        return buffered_chunk.num_rows


def _fill_buffer(buffered_chunk, chunks, desired_num_rows):
    num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
    if num_rows_in_buffer >= desired_num_rows:
        return buffered_chunk, False

    chunks_to_concat = []
    if num_rows_in_buffer:
        chunks_to_concat.append(buffered_chunk)

    total_num_rows = num_rows_in_buffer
    no_chunks_remaining = True
    for chunk in chunks:
        total_num_rows += chunk.num_rows
        chunks_to_concat.append(chunk)
        if total_num_rows >= desired_num_rows:
            no_chunks_remaining = False
            break

    if not chunks_to_concat:
        buffered_chunk = None
    elif len(chunks_to_concat) > 1:
        buffered_chunk = concatenate_chunks(chunks_to_concat)
    else:
        buffered_chunk = chunks_to_concat[0]
    return buffered_chunk, no_chunks_remaining


def _yield_chunks_from_buffer(buffered_chunk, desired_num_rows):
    num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
    if num_rows_in_buffer == desired_num_rows:
        chunks_to_yield = [buffered_chunk]
        buffered_chunk = None
        return buffered_chunk, chunks_to_yield

    start_row = 0
    chunks_to_yield = []
    end_row = None
    while True:
        previous_end_row = end_row
        end_row = start_row + desired_num_rows
        if end_row <= num_rows_in_buffer:
            chunks_to_yield.append(buffered_chunk.get_rows(slice(start_row, end_row)))
        else:
            end_row = previous_end_row
            break
        start_row = end_row

    remainder = buffered_chunk.get_rows(slice(end_row, None))
    buffered_chunk = remainder
    return buffered_chunk, chunks_to_yield


def resize_chunks(chunks: ArrayChunkIterator, desired_num_rows) -> ArrayChunkIterator:
    buffered_chunk = None

    while True:
        # fill buffer with equal or more than desired
        buffered_chunk, no_chunks_remaining = _fill_buffer(
            buffered_chunk, chunks, desired_num_rows
        )
        # yield chunks until buffer less than desired
        num_rows_in_buffer = _get_num_rows_in_chunk(buffered_chunk)
        if not num_rows_in_buffer:
            break
        buffered_chunk, chunks_to_yield = _yield_chunks_from_buffer(
            buffered_chunk, desired_num_rows
        )
        for chunk in chunks_to_yield:
            yield chunk

        if no_chunks_remaining:
            yield buffered_chunk
            break