from collections.abc import Iterator, Sequence
from typing import Callable
from collections import defaultdict
from pathlib import Path
import json
import functools
import math
import copy
import itertools


import numpy
import pandas

from variants.globals import VARIANTS_ARRAY_ID

ArrayType = tuple[numpy.ndarray, pandas.DataFrame, pandas.Series]
ARRAY_FILE_EXTENSIONS = {"DataFrame": ".parquet", "ndarray": ".npy"}


@functools.total_ordering
class GenomeLocation:
    __slots__ = ("chrom", "pos")

    def __init__(self, chrom: str, pos: int | float):
        self.chrom = chrom
        self.pos = pos

    def __str__(self):
        return f"{self.chrom}:{self.pos}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.chrom, self.pos})"

    def __gt__(self, other):
        if self.chrom > other.chrom:
            return True
        if self.chrom == other.chrom and self.pos > other.pos:
            return True
        return False

    def __eq__(self, other):
        if self.chrom == other.chrom and self.pos == other.pos:
            return True
        return False

    def to_dict(self):
        pos = self.pos
        if isinstance(pos, (int, numpy.int_)):
            pos = int(pos)
        elif isinstance(pos, float) and math.isinf(pos):
            pos = "inf"
        else:
            raise ValueError(f"pos should be int or math.inf: {type(pos)}({pos})")

        return {"chrom": self.chrom, "pos": pos}

    @classmethod
    def from_dict(cls, data: dict):
        if data["pos"] == "inf":
            pos = math.inf
        else:
            pos = int(data["pos"])
        return cls(chrom=str(data["chrom"]), pos=pos)


def as_genome_location(location):
    if isinstance(location, GenomeLocation):
        return location
    elif isinstance(location, (tuple, list)) and len(location) == 2:
        return GenomeLocation(*location)
    elif isinstance(location, dict):
        return GenomeLocation.from_dict(location)
    else:
        raise ValueError(
            "I don't know how to turn this object into a genome location:"
            + str(location)
        )


class GenomicRegion:
    def __init__(self, chrom: str, start: int, end: int | float):
        self.chrom = str(chrom)
        self.start = int(start)

        if isinstance(end, float) and math.isinf(end):
            pass
        else:
            end = int(end)
        self.end = end

    def intersects(self, region2):
        region1 = self

        if region1.chrom != region2.chrom:
            return False

        # reg1 +++++
        # reg2         -----
        if region1.end < region2.start:
            return False

        # reg1         +++++
        # reg2 -----
        if region1.start > region2.end:
            return False
        return True


def as_genomic_region(region):
    if isinstance(region, GenomicRegion):
        return region
    elif isinstance(region, (tuple, list)) and len(region) == 3:
        return GenomicRegion(chrom=region[0], start=region[1], end=region[2])
    else:
        raise ValueError(
            "I don't know how to turn this object into a genomic region:" + str(region)
        )


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

    @staticmethod
    def _fix_non_jsonable_from_dict(metadata):
        if "chunk_genome_spans" in metadata:
            metadata["chunk_genome_spans"] = {
                int(chunk_id): (
                    GenomeLocation.from_dict(start),
                    GenomeLocation.from_dict(end),
                )
                for chunk_id, (start, end) in metadata["chunk_genome_spans"].items()
            }

    def _get_metadata(self):
        metadata_path = self._get_metadata_path()
        if not metadata_path.exists():
            metadata = {}
        else:
            metadata = json.load(metadata_path.open("rt"))

        self._fix_non_jsonable_from_dict(metadata)
        return metadata

    @staticmethod
    def _fix_non_jsonable_to_dict(metadata):
        if "chunk_genome_spans" in metadata:
            metadata["chunk_genome_spans"] = {
                int(chunk_id): (start.to_dict(), end.to_dict())
                for chunk_id, (start, end) in metadata["chunk_genome_spans"].items()
            }

    def _set_metadata(self, metadata):
        self._fix_non_jsonable_to_dict(metadata)

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


def _set_array_rows(array1, index, array2):
    if isinstance(array1, numpy.ndarray):
        array1[index, ...] = array2
    elif isinstance(array1, pandas.DataFrame):
        array1 = array1.copy()
        array1.iloc[index, :] = array2
    elif isinstance(array1, pandas.Series):
        array1 = array1.copy()
        array1.iloc[index] = array2
    return array1


def _apply_mask(array, index):
    if isinstance(array, numpy.ndarray):
        array = array[index, ...]
    elif isinstance(array, pandas.DataFrame):
        array = array.loc[index, :]
    elif isinstance(array, pandas.Series):
        array = array.loc[index]
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

    def copy(self, desired_arrays: list[str] | None = None):
        cls = self.__class__

        if desired_arrays:
            arrays = {id: array for id, array in self.items() if id in desired_arrays}
        else:
            arrays = {id: array for id, array in self.items()}

        return cls(arrays=arrays, source_metadata=copy.deepcopy(self.source_metadata))

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

    def set_rows(self, index, chunk: "ArraysChunk"):
        arrays = {
            id: _set_array_rows(array, index, chunk[id])
            for id, array in self.arrays.items()
        }

        return self.__class__(arrays, source_metadata=self.source_metadata)

    def apply_mask(self, mask: Sequence[bool] | pandas.Series):
        if isinstance(mask, pandas.Series):
            mask = mask.values
        if isinstance(mask, pandas.core.arrays.boolean.BooleanArray):
            mask = mask.to_numpy(dtype=bool)
        arrays = {id: _apply_mask(array, mask) for id, array in self.arrays.items()}
        return self.__class__(arrays, source_metadata=self.source_metadata)


def _as_chunk(chunk):
    if isinstance(chunk, ArraysChunk):
        return chunk
    else:
        return ArraysChunk(chunk)


def get_samples_from_chunk(chunk):
    return chunk.source_metadata["samples"]


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


class VariantsCounter:
    def __init__(self):
        self._num_vars = 0

    def __call__(self, variations: Iterator[ArraysChunk]):
        for chunk in variations:
            self._num_vars += chunk.num_rows
            yield chunk

    @property
    def num_vars(self):
        return self._num_vars


def _get_first_chrom_from_chunk(chunk):
    return chunk.arrays[VARIANTS_ARRAY_ID]["chrom"].iloc[0]


def _split_chunks_to_have_only_one_chrom_per_chunk(vars):
    current_chrom = None
    current_chunk = None
    while True:
        if current_chunk is None:
            try:
                current_chunk = next(vars)
            except StopIteration:
                break

        if current_chrom is None:
            current_chrom = _get_first_chrom_from_chunk(current_chunk)

        current_chunk_chroms = current_chunk.arrays[VARIANTS_ARRAY_ID]["chrom"]
        to_yield = current_chunk_chroms == current_chrom
        to_yield = to_yield.values
        if isinstance(to_yield, pandas.core.arrays.arrow.array.ArrowExtensionArray):
            to_yield = to_yield.to_numpy(dtype=bool)

        if numpy.all(to_yield):
            yield current_chunk
            current_chunk = None
            current_chrom = None
            continue
        elif numpy.any(to_yield):
            yield current_chunk.get_rows(to_yield)
            remaining_chunk = current_chunk.get_rows(numpy.logical_not(to_yield))
            current_chunk = remaining_chunk
            current_chrom = None
        else:
            raise RuntimeError(
                "Fixme, it is not possible not having anything to yield at this point"
            )


def group_in_chroms(
    vars: Iterator[ArraysChunk], do_chunk_resizing=True
) -> Iterator[Iterator[ArraysChunk]]:
    vars = _split_chunks_to_have_only_one_chrom_per_chunk(vars)

    if do_chunk_resizing:
        try:
            chunk = next(vars)
        except StopIteration:
            return iter([])
        vars = itertools.chain([chunk], vars)
        num_rows_per_chunk = chunk.num_rows

    keys_and_grouped_chunks = itertools.groupby(
        vars, key=lambda chunk: _get_first_chrom_from_chunk(chunk)
    )
    grouped_chunks = (
        resize_chunks(grouped_chunks, num_rows_per_chunk)
        if do_chunk_resizing
        else grouped_chunks
        for _, grouped_chunks in keys_and_grouped_chunks
    )

    return grouped_chunks


def _get_first_pos_from_chunk(chunk):
    return chunk.arrays[VARIANTS_ARRAY_ID]["pos"].iloc[0]


def _group_vars_same_chrom_in_wins(vars, win_len):
    current_pos = None
    current_chunk = None
    while True:
        if current_chunk is None:
            try:
                current_chunk = next(vars)
            except StopIteration:
                break

        if current_pos is None:
            current_pos = _get_first_pos_from_chunk(current_chunk)

        current_chunk_poss = current_chunk.arrays[VARIANTS_ARRAY_ID]["pos"]
        to_yield = (current_chunk_poss - current_pos) < win_len
        to_yield = to_yield.values
        if isinstance(
            to_yield,
            (
                pandas.core.arrays.arrow.array.ArrowExtensionArray,
                pandas.core.arrays.boolean.BooleanArray,
            ),
        ):
            to_yield = to_yield.to_numpy(dtype=bool)

        if numpy.all(to_yield):
            yield current_pos, current_chunk
            current_chunk = None
            continue
        elif numpy.any(to_yield):
            yield current_pos, current_chunk.get_rows(to_yield)
            remaining_chunk = current_chunk.get_rows(numpy.logical_not(to_yield))
            current_chunk = remaining_chunk
            current_pos = None
        else:
            raise RuntimeError(
                "Fixme, it is not possible not having anything to yield at this point"
            )


def _group_genomic_windows_all_chunks_same_chrom(vars, win_len, num_rows_per_chunk):
    pos_and_grouped_chunks = _group_vars_same_chrom_in_wins(vars, win_len)

    pos_and_grouped_chunkss = itertools.groupby(
        pos_and_grouped_chunks, key=lambda item: item[0]
    )

    for pos, grouped_pos_and_chunks in pos_and_grouped_chunkss:
        grouped_chunks = (chunk for pos, chunk in grouped_pos_and_chunks)
        grouped_chunks = resize_chunks(grouped_chunks, num_rows_per_chunk)
        yield grouped_chunks


def group_in_genomic_windows(
    vars: Iterator[ArraysChunk], win_len: int
) -> Iterator[Iterator[ArraysChunk]]:
    try:
        chunk = next(vars)
    except StopIteration:
        return iter([])
    vars = itertools.chain([chunk], vars)
    num_rows_per_chunk = chunk.num_rows

    for chunks_for_chrom in group_in_chroms(vars, do_chunk_resizing=False):
        grouped_chunks_for_chrom = _group_genomic_windows_all_chunks_same_chrom(
            chunks_for_chrom, win_len, num_rows_per_chunk
        )
        for grouped_chunks in grouped_chunks_for_chrom:
            yield grouped_chunks


def _fill_reservoir(vars: Iterator[ArraysChunk], num_vars):
    chunks_for_reservoir = []
    num_remaining_vars_to_fill = num_vars
    for chunk in vars:
        num_rows = chunk.num_rows
        if num_rows <= num_remaining_vars_to_fill:
            chunks_for_reservoir.append(chunk)
            num_remaining_vars_to_fill -= num_rows
        else:
            chunks_for_reservoir.append(
                chunk.get_rows(slice(None, num_remaining_vars_to_fill))
            )
            remaining_chunk = chunk.get_rows(slice(num_remaining_vars_to_fill, None))
            break

    vars = itertools.chain([remaining_chunk], vars)
    reservoir = _concatenate_chunks(chunks_for_reservoir)

    if reservoir.num_rows < num_vars:
        raise ValueError(
            "{num_vars} asked to be sampled, but only {reservoir.num_rows} available"
        )

    return reservoir, vars


def _choose_new_sample(chunk, starting_num_row, num_items_asked):
    int_ranges_to_choose_from = numpy.arange(
        starting_num_row, chunk.num_rows + starting_num_row
    )

    random_nums_up_to_range = numpy.random.randint(int_ranges_to_choose_from)
    uniq_random_nums, uniq_random_num_last_pos = numpy.unique(
        numpy.flip(random_nums_up_to_range), return_index=True
    )
    random_nums_small_enough = uniq_random_nums < num_items_asked

    if not any(random_nums_small_enough):
        return None

    rows_to_get_from_this_chunk = uniq_random_num_last_pos[random_nums_small_enough]
    rows_to_remove_from_reservoir = uniq_random_nums[random_nums_small_enough]
    new_sampled_rows = chunk.get_rows(rows_to_get_from_this_chunk)

    return new_sampled_rows, rows_to_remove_from_reservoir


def sample_n_vars(
    vars: Iterator[ArraysChunk], num_vars: int, keep_order=False
) -> ArraysChunk:
    # This function uses a variation of the  Reservoir L algorithm

    if keep_order:
        raise NotImplementedError()

    reservoir, vars = _fill_reservoir(vars, num_vars)
    current_num_row = reservoir.num_rows
    for chunk in vars:
        res = _choose_new_sample(chunk, current_num_row, num_vars)
        if res is not None:
            new_sampled_vars, rows_to_remove_from_reservoir = res
            reservoir.set_rows(rows_to_remove_from_reservoir, new_sampled_vars)

        current_num_row += chunk.num_rows

    return reservoir
