import gzip
import itertools
from io import BytesIO

import numpy
import pandas
import more_itertools
import allel

from chunked_array_set import ChunkedArraySet
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


class VCFChunker:
    def __init__(self, vcf_path, num_variants_per_chunk):
        self.vcf_path = vcf_path
        self.num_variants_per_chunk = num_variants_per_chunk

        self.metadata = None
        try:
            self.metadata = next(self._get_chunks())["metadata"]
            self.metadata["header_lines"] = [
                line.decode() for line in self.metadata["header_lines"]
            ]
            self.metadata["samples"] = list(self.metadata["samples"])
        except StopIteration:
            raise RuntimeError("Failed to get metadata from VCF")

    def _get_chunks(self):
        lines = gzip.open(vcf_path, "rb")
        header_lines = []
        for line in lines:
            if line.startswith(b"#"):
                header_lines.append(line)
            else:
                lines = itertools.chain([line], lines)
                break

        metadata = {"header_lines": header_lines}
        for lines_in_chunk in more_itertools.chunked(
            lines, self.num_variants_per_chunk
        ):
            vcf_chunk = BytesIO(b"".join(header_lines + lines_in_chunk))
            allel_arrays = allel.read_vcf(vcf_chunk)
            del vcf_chunk

            metadata["samples"] = allel_arrays["samples"]

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
                variants.VARIANTS_ARRAY_ID: variants_dframe,
                variants.ALLELES_ARRAY_ID: alleles,
                variants.GT_ARRAY_ID: gts,
                variants.ORIG_VCF_ARRAY_ID: orig_vcf,
            }
            yield {"chunk": variant_chunk, "metadata": metadata}

    def get_chunks(self):
        return (chunk["chunk"] for chunk in self._get_chunks())


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
