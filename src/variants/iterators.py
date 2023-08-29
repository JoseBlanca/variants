from collections.abc import Iterator
from typing import Callable
from collections import defaultdict
import functools
import itertools


import numpy
import pandas

from variants.globals import VARIANTS_ARRAY_ID
from variants.variants import ArrayType, ArraysChunk, Genotypes, concat_genotypes


def get_samples_from_chunk(chunk):
    return chunk.samples


def _concatenate_arrays(arrays: list[ArrayType]) -> ArrayType:
    if isinstance(arrays[0], numpy.ndarray):
        array = numpy.vstack(arrays)
    elif isinstance(arrays[0], pandas.DataFrame):
        array = pandas.concat(arrays, axis=0)
    elif isinstance(arrays[0], pandas.Series):
        array = pandas.concat(arrays)
    elif isinstance(arrays[0], Genotypes):
        array = concat_genotypes(arrays)
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
    vars_iter: Iterator[ArraysChunk], num_variants: int
) -> Iterator[ArraysChunk]:
    return take_n_rows(vars_iter, num_rows=num_variants)


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

    def __call__(self, vars_iter: Iterator[ArraysChunk]):
        for chunk in vars_iter:
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
    vars_iter: Iterator[ArraysChunk], do_chunk_resizing=True
) -> Iterator[Iterator[ArraysChunk]]:
    vars = _split_chunks_to_have_only_one_chrom_per_chunk(vars_iter)

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
        elif numpy.all(numpy.logical_not(to_yield)):
            current_pos = None
            continue
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
    vars_iter: Iterator[ArraysChunk], win_len: int
) -> Iterator[Iterator[ArraysChunk]]:
    try:
        chunk = next(vars_iter)
    except StopIteration:
        return iter([])
    vars = itertools.chain([chunk], vars_iter)
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
    remaining_chunk = None
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
    if remaining_chunk is None:
        raise ValueError("No vars to sample")

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
    vars_iter: Iterator[ArraysChunk], num_vars: int, keep_order=False
) -> ArraysChunk:
    # This function uses a variation of the  Reservoir L algorithm

    if keep_order:
        raise NotImplementedError()

    reservoir, vars = _fill_reservoir(vars_iter, num_vars)
    current_num_row = reservoir.num_rows
    for chunk in vars:
        res = _choose_new_sample(chunk, current_num_row, num_vars)
        if res is not None:
            new_sampled_vars, rows_to_remove_from_reservoir = res
            reservoir.set_rows(rows_to_remove_from_reservoir, new_sampled_vars)

        current_num_row += chunk.num_rows

    return reservoir


def _sample_n_vars_per_window(
    vars_iter: Iterator[ArraysChunk], win_len_in_bp, num_vars_to_take: int = 1
):
    for idx, vars_in_win in enumerate(
        group_in_genomic_windows(vars_iter, win_len=win_len_in_bp)
    ):
        try:
            vars = sample_n_vars(vars_in_win, num_vars=num_vars_to_take)
        except ValueError:
            continue
        yield vars


def sample_n_vars_per_genomic_window(
    vars_iter: Iterator[ArraysChunk], win_len_in_bp, num_vars_to_take: int = 1
):
    try:
        chunk = next(vars_iter)
    except StopIteration:
        return iter([])
    num_vars_per_chunk = chunk.num_rows
    vars_iter = itertools.chain([chunk], vars_iter)

    vars_iter = _sample_n_vars_per_window(
        vars_iter=vars_iter,
        win_len_in_bp=win_len_in_bp,
        num_vars_to_take=num_vars_to_take,
    )

    return resize_chunks(vars_iter, desired_num_rows=num_vars_per_chunk)
