from collections.abc import Iterator
import gzip
import itertools
from io import BytesIO
import json
from pathlib import Path

import numpy
import pandas
import more_itertools
import allel

import more_itertools
import allel

import variants

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
ARRAY_FILE_EXTENSIONS = {"DataFrame": ".parquet", "ndarray": ".npy"}


class ArrayChunk:
    pass


class VariantsChunk(ArrayChunk):
    def __init__(self, arrays: dict):
        self.arrays = arrays

    @staticmethod
    def _get_metadata_chunk_path(chunk_dir):
        return chunk_dir / "metadata.json"

    def load_from_dir(self, chunk_dir):
        raise NotImplementedError()

    def write(self, chunk_dir):
        arrays_metadata = []
        num_rows = None
        for array_id, array in self.arrays.items():
            type_name = type(array).__name__
            if type_name not in ("DataFrame", "ndarray"):
                raise ValueError(f"Don't know how to store type: {type(array)}")

            array_path = chunk_dir / f"id:{array_id}{ARRAY_FILE_EXTENSIONS[type_name]}"

            if type_name == "DataFrame":
                array.to_parquet(array_path)
                save_method = "parquet"
            elif type_name == "ndarray":
                numpy.save(array_path, array)
                save_method = "npy"

            this_array_num_rows = array.shape[0]
            if num_rows is None:
                num_rows = this_array_num_rows
            else:
                if num_rows != this_array_num_rows:
                    raise ValueError("Arrays have different number of rows")

            arrays_metadata.append(
                {
                    "type_name": type_name,
                    "array_path": str(array_path),
                    "save_method": save_method,
                    "id": array_id,
                }
            )

        metadata = {"arrays_metadata": arrays_metadata}
        metadata_path = self._get_metadata_chunk_path(chunk_dir)
        with metadata_path.open("wt") as fhand:
            json.dump(metadata, fhand)

        return {"num_rows": num_rows}


class ArrayChunkIterator(Iterator):
    @property
    def num_rows(self):
        return None


class VariantIterator(ArrayChunkIterator):
    def samples(self):
        raise NotImplementedError()

    @property
    def num_variants(self):
        return super().num_rows


class VCFReader(VariantIterator):
    def __init__(
        self,
        vcf_path,
        num_variants_per_chunk: int = variants.DEFAULT_NUM_VARIANTS_IN_CHUNK,
    ) -> Iterator[VariantsChunk]:
        self.vcf_path = vcf_path
        self.num_variants_per_chunk = num_variants_per_chunk

        self._header_lines, self._lines_chunks = self._get_lines(
            self.vcf_path, self.num_variants_per_chunk
        )

    @property
    def samples(self):
        items = self._header_lines[-1].strip().split(b"\t")
        assert items[8] == b"FORMAT"
        return [sample.decode() for sample in items[9:]]

    @staticmethod
    def _get_lines(vcf_path, num_variants_per_chunk):
        lines = gzip.open(vcf_path, "rb")
        header_lines = []
        for line in lines:
            if line.startswith(b"#"):
                header_lines.append(line)
            else:
                lines = itertools.chain([line], lines)
                break

        lines_chunks = more_itertools.chunked(lines, num_variants_per_chunk)
        return header_lines, lines_chunks

    def __iter__(self):
        return self

    def __next__(self):
        try:
            lines_in_chunk = next(self._lines_chunks)
        except StopIteration:
            raise StopIteration

        vcf_chunk = BytesIO(b"".join(self._header_lines + lines_in_chunk))
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

        variant_chunk = VariantsChunk(
            {
                variants.VARIANTS_ARRAY_ID: variants_dframe,
                variants.ALLELES_ARRAY_ID: alleles,
                variants.GT_ARRAY_ID: gts,
                variants.ORIG_VCF_ARRAY_ID: orig_vcf,
            }
        )
        return variant_chunk


class ChunkedArrayDir:
    def __init__(self, dir: Path, exist_ok=False):
        self.dir = Path(dir)

        if not exist_ok and self.dir.exists():
            raise ValueError(f"dir already exists: {dir}")

        if not self.dir.exists():
            self.dir.mkdir()

    def _get_metadata_path(self):
        return self.dir / "metadata.json"

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


def write_variants(dir: Path, chunks: VariantIterator):
    dir = ChunkedArrayDir(dir, exist_ok=False)

    metadata = dir.metadata
    chunks_metadata = metadata.get("chunks_metadata", [])

    num_previous_chunks = len(chunks_metadata)
    for idx, chunk in enumerate(chunks):
        id = idx + num_previous_chunks
        chunk_dir = dir.dir / f"dataset_chunk:{id:08}"
        chunk_dir.mkdir()
        res = chunk.write(chunk_dir)
        num_rows = res["num_rows"]
        chunks_metadata.append({"id": id, "dir": chunk_dir, "num_rows": num_rows})

    dir.metadata = metadata


if __name__ == "__main__":
    vcf_path = "/home/jose/analyses/g2psol/source_data/core_collection/G2PSOLMerge.final.snp-around-indels-removed.gwas_reseq.snp_only.soft-dp-10-gq-20.hard-mean-dp-45-maxmiss-5-monomorph.maf-ge-0.05.biallelic_only.vcf.gz"
    vcf_reader = VCFReader(vcf_path)

    from pathlib import Path

    dir = Path(
        "/home/jose/analyses/g2psol/variants/core_collection_snp_only.soft-dp-10-gq-20.hard-mean-dp-45-maxmiss-5-monomorph.maf-ge-0.05.biallelic_only"
    )
    if dir.exists():
        raise RuntimeError("No vuelvas a borrarlo")

    write_variants(dir, vcf_reader)

    print("VCF read and written")

# TODO
# write_variants should be able to change the number of vars per chunk
# rechunk_iterator(ChunkIterator) -> iterator, keeps total number of rows
# VariantsReader()
# read_vcf -> Iterator
# read_variants -> Iterator
# get_first_variants(VariantIterator, num_variants)
# sample_variants(VariantIterator, num_variants)
