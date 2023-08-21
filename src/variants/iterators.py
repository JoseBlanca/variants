from collections.abc import Iterator
from typing import Callable
from collections import defaultdict
from pathlib import Path
import json
import functools

import numpy
import pandas


ArrayType = tuple[numpy.ndarray, pandas.DataFrame, pandas.Series]
ARRAY_FILE_EXTENSIONS = {"DataFrame": ".parquet", "ndarray": ".npy"}


class DirWithMetadata:
    def __init__(self, dir: Path, exist_ok=False, create_dir=False):
        self.path = Path(dir)

        if create_dir:
            if not exist_ok and self.path.exists():
                raise ValueError(f"dir already exists: {dir}")

            if not self.path.exists():
                self.path.mkdir()

    def _get_metadata_path(self):
        return self.path / "metadata.json"

    def _get_metadata(self):
        metadata_path = self._get_metadata_path()
        if not metadata_path.exists():
            metadata = {}
        else:
            metadata = json.load(metadata_path.open("rt"))
        return metadata

    def _set_metadata(self, metadata):
        metadata_path = self._get_metadata_path()
        json.dump(metadata, metadata_path.open("wt"))

    metadata = property(_get_metadata, _set_metadata)


def _write_array(array, base_path):
    if not isinstance(array, (numpy.ndarray, pandas.DataFrame, pandas.Series)):
        raise ValueError(f"Don't know how to store type: {type(array)}")

    type_name = type(array).__name__
    path = str(base_path) + ARRAY_FILE_EXTENSIONS[type_name]

    if isinstance(array, pandas.DataFrame):
        array.to_parquet(path)
        save_method = "parquet"
    elif isinstance(array, pandas.Series):
        array.to_frame().to_parquet(path)
        save_method = "parquet"
    elif isinstance(array, numpy.ndarray):
        numpy.save(path, array)
        save_method = "npy"
    else:
        raise ValueError(f"Unknown array type: {type(array)}")

    return {"path": path, "save_method": save_method, "type_name": type_name}


def _get_array_num_rows(array: ArrayType):
    return array.shape[0]


def _get_array_rows(array, index):
    if isinstance(array, numpy.ndarray):
        array = array[index, ...]
    elif isinstance(array, pandas.DataFrame):
        array = array.iloc[index, :]
    elif isinstance(array, pandas.Series):
        array = array.iloc[index]
    return array


class ArraysChunk:
    def __init__(self, arrays: dict[ArrayType], source_metadata: dict | None = None):
        if not arrays:
            raise ValueError("At least one array should be given")

        if source_metadata is None:
            source_metadata = {}
        self.source_metadata = source_metadata

        num_rows = None
        revised_cargo = {}
        for id, array in arrays.items():
            this_num_rows = _get_array_num_rows(array)
            if num_rows is None:
                num_rows = this_num_rows
            else:
                if num_rows != this_num_rows:
                    raise ValueError("All arrays should have the same number of rows")
            revised_cargo[id] = array

        self._num_rows = num_rows
        self.arrays = revised_cargo

    @property
    def num_rows(self):
        return self._num_rows

    def items(self):
        return self.arrays.items()

    def __getitem__(self, key):
        return self.arrays[key]

    def write(self, chunk_dir: Path):
        dir = DirWithMetadata(dir=chunk_dir, create_dir=True)
        arrays_metadata = []
        for array_id, array in self.items():
            base_path = chunk_dir / f"id:{array_id}"
            array_metadata = _write_array(array, base_path)
            array_metadata["id"] = array_id
            array_metadata["path"] = str(array_metadata["path"])
            arrays_metadata.append(array_metadata)
        metadata = {"arrays_metadata": arrays_metadata, "type": "ArraysChunk"}
        dir.metadata = metadata

    def get_rows(self, index):
        arrays = {
            id: _get_array_rows(array, index) for id, array in self.arrays.items()
        }

        return self.__class__(arrays, source_metadata=self.source_metadata)


def _as_chunk(chunk):
    if isinstance(chunk, ArraysChunk):
        return chunk
    else:
        return ArraysChunk(chunk)


class VariantsChunk(ArraysChunk):
    @property
    def samples(self) -> list[str]:
        return self.source_metadata["samples"]


def _concatenate_arrays(arrays: list[ArrayType]) -> ArrayType:
    if isinstance(arrays[0], numpy.ndarray):
        array = numpy.vstack(arrays)
    elif isinstance(arrays[0], pandas.DataFrame):
        array = pandas.concat(arrays, axis=0)
    elif isinstance(arrays[0], pandas.Series):
        array = pandas.concat(arrays)
    else:
        raise ValueError("unknown type for array: " + str(type(arrays[0])))
    return array


def _concatenate_chunks(chunks: list[ArraysChunk]):
    chunks = list(chunks)

    if len(chunks) == 1:
        return chunks[0]

    arrays_to_concatenate = defaultdict(list)
    class_ = None
    for chunk in chunks:
        if class_ is None:
            class_ = chunk.__class__
        for array_id, array in chunk.items():
            arrays_to_concatenate[array_id].append(array)

    num_arrays = [len(arrays) for arrays in arrays_to_concatenate.values()]
    if not all([num_arrays[0] == len_ for len_ in num_arrays]):
        raise ValueError("Nota all chunks have the same arrays")

    concatenated_chunk = {}
    for array_id, arrays in arrays_to_concatenate.items():
        concatenated_chunk[array_id] = _concatenate_arrays(arrays)
    concatenated_chunk = class_(concatenated_chunk)
    return concatenated_chunk


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
        buffered_chunk = _concatenate_chunks(chunks_to_concat)
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


def resize_chunks(
    chunks: Iterator[ArraysChunk], desired_num_rows
) -> Iterator[ArraysChunk]:
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


def take_n_rows(chunks: Iterator[ArraysChunk], num_rows: int) -> Iterator[ArraysChunk]:
    rows_yielded = 0
    all_chunks_yielded = True
    for chunk in chunks:
        if chunk.num_rows + rows_yielded < num_rows:
            yield chunk
            rows_yielded += chunk.num_rows
        else:
            all_chunks_yielded = False
            break

    if not all_chunks_yielded:
        remaing_rows = num_rows - rows_yielded
        yield chunk.get_rows(slice(0, remaing_rows))


def take_n_variants(
    chunks: Iterator[ArraysChunk], num_variants: int
) -> Iterator[ArraysChunk]:
    return take_n_rows(chunks, num_rows=num_variants)


def run_pipeline(
    chunks: Iterator[ArraysChunk],
    map_functs: list[Callable] | None = None,
    reduce_funct: Callable | None = None,
    reduce_initialializer=None,
):
    if map_functs is None:
        map_functs = []

    all_mappers_return_same_num_lines = all(
        [getattr(funct, "returns_same_num_lines", False) for funct in map_functs]
    )
    if not reduce_funct and all_mappers_return_same_num_lines:
        num_rows_expected = getattr(chunks, "num_rows_expected", None)
    else:
        num_rows_expected = None
    try:
        samples = chunks.samples
        is_variants = True
    except AttributeError:
        is_variants = False

    def funct(item):
        processed_item = item
        for one_funct in map_functs:
            processed_item = one_funct(processed_item)
        return processed_item

    processed_chunks = map(funct, chunks)

    if reduce_funct:
        reduced_result = functools.reduce(
            reduce_funct, processed_chunks, reduce_initialializer
        )
        result = reduced_result
    else:
        result = processed_chunks

    return result
