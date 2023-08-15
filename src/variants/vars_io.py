import gzip
import itertools
from io import BytesIO

import numpy
import pandas
import more_itertools
import allel

from chunked_array_set import ChunkedArraySet
from .array_iterator import VariantsIterator, ArraysChunk

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


class _VCFChunker:
    def __init__(self, vcf_fhand, num_variants_per_chunk):
        self._lines = gzip.open(vcf_fhand, "rb")
        self.num_variants_per_chunk = num_variants_per_chunk

        try:
            self._metadata = self._read_metadata()
        except StopIteration:
            raise RuntimeError("Failed to get metadata from VCF")

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

            variant_chunk = {
                VARIANTS_ARRAY_ID: variants_dframe,
                ALLELES_ARRAY_ID: alleles,
                GT_ARRAY_ID: gts,
                ORIG_VCF_ARRAY_ID: orig_vcf,
            }
            yield ArraysChunk(variant_chunk)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._chunks)


def read_vcf(
    fhand, num_variants_per_chunk=DEFAULT_NUM_VARIANTS_PER_CHUNK
) -> VariantsIterator:
    return _VCFChunker(fhand, num_variants_per_chunk=num_variants_per_chunk)


if __name__ == "__main__":
    vcf_path = "/home/jose/analyses/g2psol/source_data/core_collection/G2PSOLMerge.final.snp-around-indels-removed.gwas_reseq.snp_only.soft-dp-10-gq-20.hard-mean-dp-45-maxmiss-5-monomorph.maf-ge-0.05.biallelic_only.vcf.gz"
    vcf_chunker = VCFChunker(vcf_path, 10000)
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
