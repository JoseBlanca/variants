from functools import partial

import numpy
import pandas

from chunked_array_set import ChunkedArraySet, create_empty_array_like, set_array_chunk

GT_ARRAY_ID = "gts"
VARIANTS_ARRAY_ID = "variants"
ALLELES_ARRAY_ID = "alleles"
ORIG_VCF_ARRAY_ID = "orig_vcf"
MISSING_INT = -1


def _calc_gt_is_missing(gts):
    allele_is_missing = gts == MISSING_INT
    allele_is_missing = numpy.any(allele_is_missing, axis=2)
    return allele_is_missing


def _calc_obs_het_per_var_for_chunk(chunk, pops):
    gts = chunk[GT_ARRAY_ID]
    gt_is_het = numpy.logical_not(
        numpy.all(gts == gts[:, :, 0][:, :, numpy.newaxis], axis=2)
    )
    gt_is_missing = _calc_gt_is_missing(gts)
    gt_is_het = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))

    obs_het_per_var = {}
    for pop_id, pop_slice in pops.items():
        num_vars_het_per_var = numpy.sum(gt_is_het[:, pop_slice], axis=1)
        num_non_missing_per_var = gts.shape[0] - numpy.sum(
            gt_is_missing[:, pop_slice], axis=1
        )
        obs_het_per_var[pop_id] = num_vars_het_per_var / num_non_missing_per_var

    obs_het_per_var = pandas.DataFrame(obs_het_per_var)
    return {"obs_het_per_var": obs_het_per_var}


def _count_alleles_per_var_for_chunk(chunk, pops, alleles, missing_gt):
    gts = chunk[GT_ARRAY_ID]

    alleles_in_chunk = set(numpy.unique(gts)).difference([missing_gt])

    if alleles is not None:
        if alleles_in_chunk.difference(alleles):
            raise RuntimeError("Chunk more alleles than the given ones")

    result = {}
    for pop_id, pop_slice in pops.items():
        pop_gts = gts[:, pop_slice, :]
        allele_counts = numpy.empty(
            shape=(pop_gts.shape[0], len(alleles)), dtype=numpy.int64
        )
        missing_counts = None
        for idx, allele in enumerate([missing_gt] + alleles):
            allele_counts_per_row = numpy.sum(pop_gts == allele, axis=(1, 2))
            if idx == 0:
                missing_counts = allele_counts_per_row.reshape(
                    (allele_counts_per_row.shape[0], 1)
                )
            else:
                allele_counts[:, idx - 1] = allele_counts_per_row
        allele_counts = pandas.DataFrame(allele_counts, columns=alleles)
        result[pop_id] = {
            "allele_counts": allele_counts,
            "missing_gts": missing_counts,
        }

    return {"counts": result, "alleles": alleles_in_chunk}


class Variants:
    def __init__(self, array_set_dir):
        self.array_set = ChunkedArraySet(dir=array_set_dir)

    @property
    def samples(self):
        samples = self.array_set.metadata["samples"]
        return samples

    @property
    def num_vars(self):
        return self.array_set.num_rows

    def _calc_pops_idxs(self, pops: dict[list[str]] | None):
        if pops is None:
            pops_idxs = {0: slice(None, None)}
        else:
            samples_idx = {sample: idx for idx, sample in enumerate(self.samples)}
            pops_idxs = {}
            for pop_id, pop_samples in pops.items():
                pop_idxs = [samples_idx[sample] for sample in pop_samples]
                pops_idxs[pop_id] = pop_idxs
        return pops_idxs

    def calc_obs_het_per_var(
        self, pops: dict[list[str]] | None = None
    ) -> pandas.DataFrame:
        pops = self._calc_pops_idxs(pops)

        calc_obs_het_per_var_for_chunk = partial(
            _calc_obs_het_per_var_for_chunk, pops=pops
        )

        collect_results = _ResultCollectorForArrayDict(num_rows=self.array_set.num_rows)
        result = self.array_set.run_pipeline(
            map_functs=[calc_obs_het_per_var_for_chunk],
            desired_arrays_to_load_in_chunk=[GT_ARRAY_ID],
            reduce_funct=collect_results,
            reduce_initialializer=None,
        )
        return result["obs_het_per_var"]

    def calc_obs_het_per_sample(self):
        raise NotImplementedError()

    def count_alleles_per_var(
        self,
        pops=None,
        alleles=None,
        calc_freqs=False,
        missing_gt=MISSING_INT,
    ):
        pops = self._calc_pops_idxs(pops)

        if alleles is None:
            alleles = list(range(10))
            alleles_asked = False
        else:
            alleles_asked = True

        count_alleles_per_var_for_chunk = partial(
            _count_alleles_per_var_for_chunk,
            pops=pops,
            alleles=alleles,
            missing_gt=missing_gt,
        )

        try:
            collect_allele_counts = _AlleleCountCollector(
                num_rows=self.array_set.num_rows
            )
            result = self.array_set.run_pipeline(
                map_functs=[count_alleles_per_var_for_chunk],
                desired_arrays_to_load_in_chunk=[GT_ARRAY_ID],
                reduce_funct=collect_allele_counts,
                reduce_initialializer=None,
            )
        except RuntimeError:
            count_alleles_per_var_for_chunk = partial(
                _count_alleles_per_var_for_chunk,
                pops=pops,
                alleles=self.get_different_alleles(),
                missing_gt=missing_gt,
            )
            result = self.array_set.run_pipeline(
                map_functs=[count_alleles_per_var_for_chunk],
                desired_arrays_to_load_in_chunk=[GT_ARRAY_ID],
                reduce_funct=collect_allele_counts,
                reduce_initialializer=None,
            )

        if not alleles_asked:
            # remove_extra_alleles
            alleles = sorted(result["alleles"])
            for pop, pop_result in result["counts"].items():
                result["counts"][pop]["allele_counts"] = pop_result[
                    "allele_counts"
                ].loc[:, alleles]

        result = result["counts"]
        return result

    def get_different_alleles(self, missing_gt=MISSING_INT):
        def accumulate_alleles(accumulated_alleles, new_alleles):
            accumulated_alleles.update(new_alleles)
            return accumulated_alleles

        result = self.array_set.run_pipeline(
            map_functs=[lambda chunk: numpy.unique(chunk[GT_ARRAY_ID])],
            desired_arrays_to_load_in_chunk=[GT_ARRAY_ID],
            reduce_funct=accumulate_alleles,
            reduce_initialializer=set(),
        )
        result = sorted(result.difference([missing_gt]))
        return result


class _ResultCollectorForArrayDict:
    def __init__(self, num_rows):
        self.num_rows = num_rows
        self.row_start = 0

    def __call__(self, accumulated_result, processed_chunk):
        if accumulated_result is None:
            accumulated_result = self._create_empty_complete_result(
                accumulated_result, processed_chunk
            )

        return self._accumulate_result(accumulated_result, processed_chunk)

    def _create_empty_complete_result(self, accumulated_result, processed_chunk):
        accumulated_result = {}
        for array_id, array_chunk in processed_chunk.items():
            complete_array = create_empty_array_like(
                array_chunk, num_rows=self.num_rows
            )
            accumulated_result[array_id] = complete_array
        return accumulated_result

    def _accumulate_result(self, accumulated_result, processed_chunk):
        row_end = None

        for array_id, array_chunk in processed_chunk.items():
            if row_end is None:
                chunk_num_rows = array_chunk.shape[0]
                row_end = self.row_start + chunk_num_rows

            set_array_chunk(
                accumulated_result[array_id],
                array_chunk,
                self.row_start,
                row_end,
            )

        self.row_start = row_end
        return accumulated_result


class _AlleleCountCollector:
    def __init__(self, num_rows):
        self.num_rows = num_rows
        self.row_start = 0

    def __call__(self, complete_array_results, processed_chunk):
        if complete_array_results is None:
            # create empty complete arrays
            complete_array_results = {
                "counts": {pop: {} for pop in processed_chunk["counts"].keys()},
                "alleles": set(),
            }
            for pop, pop_result in processed_chunk["counts"].items():
                complete_array_results["counts"][pop][
                    "missing_gts"
                ] = create_empty_array_like(
                    pop_result["missing_gts"], num_rows=self.num_rows
                )
                complete_array_results["counts"][pop][
                    "allele_counts"
                ] = create_empty_array_like(
                    pop_result["allele_counts"], num_rows=self.num_rows
                )
                complete_array_results["counts"][pop]

        # put processed chunk in complete array
        row_end = None
        for pop, pop_result in processed_chunk["counts"].items():
            if row_end is None:
                chunk_num_rows = pop_result["missing_gts"].shape[0]
                row_end = self.row_start + chunk_num_rows

            for pop, pop_result in processed_chunk["counts"].items():
                set_array_chunk(
                    complete_array_results["counts"][pop]["missing_gts"],
                    pop_result["missing_gts"],
                    self.row_start,
                    row_end,
                )
                set_array_chunk(
                    complete_array_results["counts"][pop]["allele_counts"],
                    pop_result["allele_counts"],
                    self.row_start,
                    row_end,
                )
        complete_array_results["alleles"].update(processed_chunk["alleles"])

        self.row_start = row_end
        return complete_array_results


if __name__ == "__main__":
    vars = Variants(
        array_set_dir="/home/jose/analyses/g2psol/variants/core_collection_snp_only.soft-dp-10-gq-20.hard-mean-dp-45-maxmiss-5-monomorph.maf-ge-0.05.biallelic_only"
    )
    print(f"num samples: {len(vars.samples)}")
    print(f"num vars: {vars.num_vars}")
    pops = {
        "pop1": ["GPT000140", "GPT000170", "GPT000210", "GPT000270", "GPT000280"],
        "pop2": [
            "GPT029600",
            "GPT029760",
            "GPT030150",
            "GPT030290",
            "GPT030350",
            "GPT030610",
        ],
    }
    if True:
        print(vars.calc_obs_het_per_var(pops=pops))
        # variants.calc_obs_het_per_var()
    elif True:
        print(vars.count_alleles_per_var(pops=pops))
    elif False:
        print(vars.get_different_alleles())

"""
TODO
maff i don't need alleles
missing data ratio per snp
filter per genomic regions, for any region chrom== <pos<
allele freqs
"""
