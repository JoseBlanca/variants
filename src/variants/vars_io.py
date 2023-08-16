import gzip
import itertools
from io import BytesIO
from pathlib import Path

import numpy
import pandas
import more_itertools
import allel

from chunked_array_set import ChunkedArraySet
from variants.iterators import (
    VariantsIterator,
    ArrayChunk,
    ArraysChunk,
    ArrayChunkIterator,
    DirWithMetadata,
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


class _VCFChunker(VariantsIterator):
    def __init__(self, vcf_fhand, num_variants_per_chunk):
        self._lines = gzip.open(vcf_fhand, "rb")
        self.num_variants_per_chunk = num_variants_per_chunk

        try:
            self._metadata = self._read_metadata()
        except StopIteration:
            raise RuntimeError("Failed to get metadata from VCF")

        self._expected_rows = None
        self._num_rows_processed = 0

    @property
    def samples(self):
        return self._metadata["samples"]

    def _read_metadata(self):
        header_lines = []
        for line in self._lines:
            if line.startswith(b"#"):
                header_lines.append(line)
            else:
                lines = itertools.chain([line], self._lines)
                break
        self._lines = lines

        header_lines = [line.decode() for line in header_lines]

        items = header_lines[-1].strip().split("\t")
        assert items[8] == "FORMAT"
        samples = [sample for sample in items[9:]]
        self._chunks = self._get_chunks()

        return {"header_lines": header_lines, "samples": samples}

    def _get_chunks(self):
        header_lines = [line.encode() for line in self._metadata["header_lines"]]
        for lines_in_chunk in more_itertools.chunked(
            self._lines, self.num_variants_per_chunk
        ):
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
            alleles = pandas.DataFrame(
                numpy.hstack([ref, alt]), dtype=STRING_PANDAS_DTYPE
            )

            gts = allel_arrays["calldata/GT"]

            orig_vcf = pandas.Series(lines_in_chunk, dtype=STRING_PANDAS_DTYPE)
            orig_vcf = pandas.DataFrame({"vcf_line": orig_vcf})

            variant_chunk = ArraysChunk(
                {
                    VARIANTS_ARRAY_ID: variants_dframe,
                    ALLELES_ARRAY_ID: alleles,
                    GT_ARRAY_ID: gts,
                    ORIG_VCF_ARRAY_ID: orig_vcf,
                }
            )
            self._num_rows_processed += variant_chunk.num_rows
            yield variant_chunk

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)


def _get_fhand_rb(fhand):
    if isinstance(fhand, (str, Path)):
        return open(fhand, "rb")
    return fhand


def read_vcf(
    fhand, num_variants_per_chunk=DEFAULT_NUM_VARIANTS_PER_CHUNK
) -> VariantsIterator:
    fhand = _get_fhand_rb(fhand)

    return _VCFChunker(fhand, num_variants_per_chunk=num_variants_per_chunk)


def write_variants(dir: Path, variants: VariantsIterator):
    return write_chunks(
        dir, variants, additional_metadata={"samples": variants.samples}
    )


def write_chunks(
    dir: Path, chunks: ArrayChunkIterator, additional_metadata: dict | None = None
):
    dir = DirWithMetadata(dir, exist_ok=False, create_dir=True)

    chunks_metadata = []
    metadata = {"chunks_metadata": chunks_metadata}
    if additional_metadata:
        metadata.update(additional_metadata)

    num_rows_processed = 0
    for idx, chunk in enumerate(chunks):
        id = idx
        chunk_dir = dir.path / f"dataset_chunk:{id:08}"
        chunk.write(chunk_dir)
        num_rows = chunk.num_rows
        num_rows_processed += num_rows
        chunks_metadata.append({"id": id, "dir": str(chunk_dir), "num_rows": num_rows})
    metadata["num_rows"] = num_rows_processed
    dir.metadata = metadata

    return {"num_rows_processed": num_rows_processed}


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
            f"Don't know how to load array of type {array_type} from file {file_format}"
        )
    return array


def _load_array_chunk(array_metadata):
    array = _load_array(
        array_metadata["path"],
        array_type=array_metadata["type_name"],
        file_format=array_metadata["save_method"],
    )
    chunk = ArrayChunk(array)
    return chunk


def _load_arrays_chunk(arrays_metadata):
    arrays = {}
    for array_metadata in arrays_metadata:
        array = _load_array(
            array_metadata["path"],
            array_type=array_metadata["type_name"],
            file_format=array_metadata["type_name"],
        )
        id = array_metadata["id"]
        arrays[id] = array
    chunk = ArraysChunk(arrays)
    return chunk


def _load_chunks_from_metadata(chunks_metadata):
    for chunk_info in chunks_metadata:
        chunk_dir = DirWithMetadata(chunk_info["dir"], create_dir=False)
        chunk_metadata = chunk_dir.metadata
        if chunk_metadata["type"] == "ArrayChunk":
            array_metadata = chunk_metadata["array_metadata"]
            chunk = _load_array_chunk(array_metadata)
        elif chunk_metadata["type"] == "ArraysChunk":
            arrays_metadata = chunk_metadata["arrays_metadata"]
            chunk = _load_arrays_chunk(arrays_metadata)
        yield chunk


def _load_chunks_and_metadata(dir: Path):
    dir = DirWithMetadata(dir, create_dir=False)
    metadata = dir.metadata
    num_rows = metadata.get("num_rows")
    chunks_metadata = metadata["chunks_metadata"]
    chunks = _load_chunks_from_metadata(chunks_metadata)
    return {
        "chunks": ArrayChunkIterator(chunks, expected_total_num_rows=num_rows),
        "metadata": metadata,
    }


def load_chunks(dir: Path) -> ArrayChunkIterator:
    return _load_chunks_and_metadata(dir)["chunks"]


def load_variants(dir: Path) -> VariantsIterator:
    res = _load_chunks_and_metadata(dir)
    print("expected", res["chunks"].num_rows_expected)
    return VariantsIterator(res["chunks"], samples=res["metadata"]["samples"])


if __name__ == "__main__":
    vcf_path = "/home/jose/analyses/g2psol/source_data/core_collection/G2PSOLMerge.final.snp-around-indels-removed.gwas_reseq.snp_only.soft-dp-10-gq-20.hard-mean-dp-45-maxmiss-5-monomorph.maf-ge-0.05.biallelic_only.vcf.gz"
    variants = read_vcf(vcf_path)
    for chunk in variants:
        print(chunk.num_rows)
    1 / 0
    from pathlib import Path

    dir = Path(
        "/home/jose/analyses/g2psol/variants/core_collection_snp_only.soft-dp-10-gq-20.hard-mean-dp-45-maxmiss-5-monomorph.maf-ge-0.05.biallelic_only"
    )
    if dir.exists():
        raise RuntimeError("No vuelvas a borrarlo")
    array_set = ChunkedArraySet(dir=dir)
    array_set.metadata = vcf_chunker.metadata
    array_set.extend_chunks(vcf_chunker.get_chunks())
    print("VCF read")
