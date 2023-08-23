from collections.abc import Iterator
import gzip
import itertools
from io import BytesIO
from pathlib import Path
import copy
import math


import numpy
import pandas
import more_itertools
import allel

from variants.iterators import (
    ArraysChunk,
    DirWithMetadata,
    GenomeLocation,
    as_genomic_region,
    GenomicRegion,
)

GT_ARRAY_ID = "gts"
VARIANTS_ARRAY_ID = "variants"
ALLELES_ARRAY_ID = "alleles"
ORIG_VCF_ARRAY_ID = "orig_vcf"
MISSING_INT = -1
DEFAULT_NUM_VARIANTS_PER_CHUNK = 10000

STRING_PANDAS_DTYPE = "string[pyarrow]"
CHROM_FIELD = {
    "scikit-allel_path": "variants/CHROM",
    "dframe_column": "chrom",
    "pandas_dtype": STRING_PANDAS_DTYPE,
}
POS_FIELD = {
    "scikit-allel_path": "variants/POS",
    "dframe_column": "pos",
    "pandas_dtype": pandas.Int64Dtype(),
}

VARIANT_FIELDS = [CHROM_FIELD, POS_FIELD]


def _get_fhand_rb(fhand):
    if isinstance(fhand, (str, Path)):
        return open(fhand, "rb")
    return fhand


def _read_vcf_metadata(lines):
    header_lines = []
    for line in lines:
        if line.startswith(b"#"):
            header_lines.append(line)
        else:
            lines = itertools.chain([line], lines)
            break

    header_lines = [line.decode() for line in header_lines]

    items = header_lines[-1].strip().split("\t")
    assert items[8] == "FORMAT"
    samples = [sample for sample in items[9:]]

    return {"header_lines": header_lines, "samples": samples, "lines": lines}


def _get_vcf_chunks(lines, source_metadata, num_variants_per_chunk):
    header_lines = [line.encode() for line in source_metadata["header_lines"]]
    for lines_in_chunk in more_itertools.chunked(lines, num_variants_per_chunk):
        vcf_chunk = BytesIO(b"".join(header_lines + lines_in_chunk))
        allel_arrays = allel.read_vcf(vcf_chunk)
        del vcf_chunk

        variants_dframe = {}
        for field_data in VARIANT_FIELDS:
            col_name = field_data["dframe_column"]
            col_data = pandas.Series(
                allel_arrays[field_data["scikit-allel_path"]],
                dtype=field_data["pandas_dtype"],
            )
            variants_dframe[col_name] = col_data
        variants_dframe = pandas.DataFrame(variants_dframe)

        ref = allel_arrays["variants/REF"]
        ref = ref.reshape((ref.shape[0], 1))
        alt = allel_arrays["variants/ALT"]
        alleles = pandas.DataFrame(numpy.hstack([ref, alt]), dtype=STRING_PANDAS_DTYPE)

        gts = allel_arrays["calldata/GT"]

        orig_vcf = pandas.Series(lines_in_chunk, dtype=STRING_PANDAS_DTYPE)
        orig_vcf = pandas.DataFrame({"vcf_line": orig_vcf})

        variant_chunk = ArraysChunk(
            {
                VARIANTS_ARRAY_ID: variants_dframe,
                ALLELES_ARRAY_ID: alleles,
                GT_ARRAY_ID: gts,
                ORIG_VCF_ARRAY_ID: orig_vcf,
            },
            source_metadata=source_metadata,
        )
        yield variant_chunk


def read_vcf(
    fhand, num_variants_per_chunk=DEFAULT_NUM_VARIANTS_PER_CHUNK
) -> Iterator[ArraysChunk]:
    fhand = _get_fhand_rb(fhand)

    lines = gzip.open(fhand, "rb")

    try:
        res = _read_vcf_metadata(lines)
    except StopIteration:
        raise RuntimeError("Failed to get metadata from VCF")

    source_metadata = {"header_lines": res["header_lines"], "samples": res["samples"]}
    lines = res["lines"]

    yield from _get_vcf_chunks(lines, source_metadata, num_variants_per_chunk)


def read_vcf_metadata(fhand):
    fhand = _get_fhand_rb(fhand)
    lines = gzip.open(fhand, "rb")
    res = _read_vcf_metadata(lines)
    del res["lines"]

    return res


def write_variants(dir: Path, variants: Iterator[ArraysChunk]):
    try:
        chunk = next(variants)
    except StopIteration:
        raise ValueError("No variants to write")

    source_metadata = chunk.source_metadata
    try:
        metadata = {"samples": source_metadata["samples"]}
    except KeyError:
        raise ValueError("No samples in source_metadata for these variant chunks")

    if "total_num_rows" in source_metadata:
        expected_num_rows = source_metadata["total_num_rows"]
    else:
        expected_num_rows = None

    variants = itertools.chain([chunk], variants)

    # samples num_variants
    write_chunks(dir, variants, additional_metadata=metadata)

    dir = DirWithMetadata(dir, create_dir=False)
    metadata = dir.metadata
    if expected_num_rows:
        assert metadata["total_num_rows"] == expected_num_rows
    metadata["total_num_variants"] = metadata["total_num_rows"]
    dir.metadata = metadata


def write_chunks(
    dir: Path, chunks: Iterator[ArraysChunk], additional_metadata: dict | None = None
):
    dir = DirWithMetadata(dir, exist_ok=False, create_dir=True)

    chunks_metadata = []
    metadata = {"chunks_metadata": chunks_metadata}
    if additional_metadata:
        metadata.update(additional_metadata)

    pos_are_defined_and_sorted = False
    last_chrom = ""
    last_pos = 0
    num_rows_processed = 0
    chunk_spans = {}
    for idx, chunk in enumerate(chunks):
        id = idx
        chunk_dir = dir.path / f"dataset_chunk:{id:08}"
        chunk.write(chunk_dir)

        if VARIANTS_ARRAY_ID in chunk.arrays:
            variants_dframe = chunk.arrays[VARIANTS_ARRAY_ID]
            cols = variants_dframe.columns
            if "chrom" in cols and "pos" in cols:
                pos_are_defined_and_sorted = True
        if pos_are_defined_and_sorted:
            variants_dframe = chunk.arrays[VARIANTS_ARRAY_ID]
            chroms = variants_dframe["chrom"]
            poss = variants_dframe["pos"]

            prev_chroms = numpy.concatenate(([last_chrom], chroms.iloc[0:-1].values))
            prev_poss = numpy.concatenate(([last_pos], poss.iloc[0:-1].values))
            poss_are_lt_prev_poss = numpy.all(
                numpy.logical_or(
                    chroms > prev_chroms,
                    numpy.logical_and(chroms == prev_chroms, poss >= prev_poss),
                )
            )
            if not poss_are_lt_prev_poss:
                pos_are_defined_and_sorted = False

            first_chunk_loc = GenomeLocation(chroms.iloc[0], poss.iloc[0])
            last_chunk_loc = GenomeLocation(chroms.iloc[-1], poss.iloc[-1])
            chunk_spans[id] = (first_chunk_loc, last_chunk_loc)

            last_chrom = chroms.iloc[-1]
            last_pos = poss.iloc[-1]

        num_rows = chunk.num_rows
        num_rows_processed += num_rows
        chunks_metadata.append({"id": id, "dir": str(chunk_dir), "num_rows": num_rows})
    metadata["total_num_rows"] = num_rows_processed
    if pos_are_defined_and_sorted:
        metadata["chunk_genome_spans"] = chunk_spans
    dir.metadata = metadata


class VariantsDir(DirWithMetadata):
    @property
    def samples(self):
        return self.metadata["samples"]

    @property
    def num_variants(self):
        return self.metadata["total_num_rows"]

    @property
    def num_samples(self):
        return len(self.samples)

    def iterate_over_variants(
        self,
        desired_arrays: list[str] | None = None,
        get_only_chunks_intersecting_regions: list[(GenomeLocation, GenomeLocation)]
        | None = None,
        sorted_chroms: list[str] | None = None,
    ) -> Iterator[ArraysChunk]:
        if get_only_chunks_intersecting_regions is not None:
            get_only_chunks_intersecting_regions = [
                as_genomic_region(region)
                for region in get_only_chunks_intersecting_regions
            ]

            if sorted_chroms is None:
                raise ValueError(
                    "If you want to only read chunks for certain regions you have to supply the list of sorted chroms "
                )

        source_metadata = self.metadata
        chunks_metadata = source_metadata["chunks_metadata"]
        chunk_genome_spans = source_metadata.get("chunk_genome_spans")

        source_metadata = {
            "samples": self.samples,
            "total_num_variants": self.num_variants,
        }

        return _read_chunks(
            chunks_metadata=chunks_metadata,
            source_metadata=source_metadata,
            desired_arrays=desired_arrays,
            get_only_chunks_intersecting_regions=get_only_chunks_intersecting_regions,
            chunk_genome_spans=chunk_genome_spans,
            sorted_chroms=sorted_chroms,
        )


def _load_array(path, array_type, file_format):
    path = Path(path)
    if array_type == "ndarray" and file_format == "npy":
        array = numpy.load(path)
    elif array_type == "DataFrame" and file_format == "parquet":
        array = pandas.read_parquet(path)
    elif array_type == "Series" and file_format == "parquet":
        array = pandas.read_parquet(path).squeeze()
    else:
        raise ValueError(
            f"Don't know how to load array of type {array_type} from file format {file_format}"
        )
    return array


def _load_arrays_chunk(arrays_metadata, desired_arrays, source_metadata):
    arrays = {}
    for array_metadata in arrays_metadata:
        id = array_metadata["id"]
        if desired_arrays and id not in desired_arrays:
            continue
        array = _load_array(
            array_metadata["path"],
            array_type=array_metadata["type_name"],
            file_format=array_metadata["save_method"],
        )
        arrays[id] = array
    chunk = ArraysChunk(arrays, source_metadata)
    return chunk


def _read_chunks(
    chunks_metadata,
    source_metadata,
    desired_arrays: list[str] | None = None,
    get_only_chunks_intersecting_regions=None,
    chunk_genome_spans=None,
    sorted_chroms=None,
):
    filter_by_region = bool(get_only_chunks_intersecting_regions)

    for chunk_info in chunks_metadata:
        if filter_by_region:
            if chunk_genome_spans is None:
                raise RuntimeError("No metadata for chunk spans in the source variants")
            chunk_span_start, chunk_span_end = chunk_genome_spans[chunk_info["id"]]
            chunk_regions = []
            if chunk_span_start.chrom == chunk_span_end.chrom:
                chunk_regions.append(
                    GenomicRegion(
                        chunk_span_start.chrom, chunk_span_start.pos, chunk_span_end.pos
                    )
                )
            else:
                chunk_regions.append(
                    GenomicRegion(
                        chunk_span_start.chrom, chunk_span_start.pos, math.inf
                    )
                )
                chunk_regions.append(
                    GenomicRegion(chunk_span_end.chrom, 0, chunk_span_end.pos)
                )
                chrom0_index = sorted_chroms.index(chunk_span_start.chrom)
                chrom1_index = sorted_chroms.index(chunk_span_end.chrom)
                for intermediate_chrom in sorted_chroms[
                    chrom0_index + 1 : chrom1_index
                ]:
                    chunk_regions.append(GenomicRegion(intermediate_chrom, 0, math.inf))

            yield_chunk = False
            for chunk_region in chunk_regions:
                if any(
                    region.intersects(chunk_region)
                    for region in get_only_chunks_intersecting_regions
                ):
                    yield_chunk = True
                    break
        else:
            yield_chunk = True

        if not yield_chunk:
            continue

        chunk_dir = DirWithMetadata(chunk_info["dir"], create_dir=False)
        chunk_metadata = chunk_dir.metadata
        arrays_metadata = chunk_metadata["arrays_metadata"]
        chunk = _load_arrays_chunk(
            arrays_metadata,
            desired_arrays,
            source_metadata=copy.deepcopy(source_metadata),
        )
        yield chunk


def read_variants(path, desired_arrays: list[str] | None = None):
    dir = VariantsDir(path)
    return dir.iterate_over_variants(desired_arrays=desired_arrays)
