from functools import partial

import numpy
import pandas

from chunked_array_set import (
    ChunkedArraySet,
    create_empty_array_like,
    set_array_chunk,
    filter_array_chunk_rows,
)

GT_ARRAY_ID = "gts"
VARIANTS_ARRAY_ID = "variants"
ALLELES_ARRAY_ID = "alleles"
ORIG_VCF_ARRAY_ID = "orig_vcf"
MISSING_INT = -1
DEFAULT_NUM_VARIANTS_IN_CHUNK = 10000


def _calc_gt_is_missing(gts, gt_is_missing=None):
    if gt_is_missing is not None:
        return gt_is_missing
    allele_is_missing = gts == MISSING_INT
    allele_is_missing = numpy.any(allele_is_missing, axis=2)
    return allele_is_missing


def _calc_missing_rate_per_var(gts, gt_is_missing=None):
    gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)

    missing_rate = gt_is_missing.sum(axis=1) / gt_is_missing.shape[1]
    return missing_rate


def _calc_gts_is_het(gts, gt_is_missing=None):
    gt_is_het = numpy.logical_not(
        numpy.all(gts == gts[:, :, 0][:, :, numpy.newaxis], axis=2)
    )
    gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)
    gt_is_het = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))
    return gt_is_het


def _calc_obs_het_rate_per_var(gts, gt_is_missing=None):
    gt_is_het = _calc_gts_is_het(gts, gt_is_missing=gt_is_missing)
    obs_het_rate = gt_is_het.sum(axis=1) / gt_is_missing.shape[1]
    return obs_het_rate


def _calc_obs_het_per_var_for_chunk(chunk, pops):
    gts = chunk[GT_ARRAY_ID]
    gt_is_missing = _calc_gt_is_missing(gts)
    gt_is_het = numpy.logical_and(gt_is_het, numpy.logical_not(gt_is_missing))
    gt_is_het = _calc_gts_is_het(gts, gt_is_missing=gt_is_missing)

    obs_het_per_var = {}
    for pop_id, pop_slice in pops.items():
        num_vars_het_per_var = numpy.sum(gt_is_het[:, pop_slice], axis=1)
        num_non_missing_per_var = gts.shape[0] - numpy.sum(
            gt_is_missing[:, pop_slice], axis=1
        )
        obs_het_per_var[pop_id] = num_vars_het_per_var / num_non_missing_per_var

    obs_het_per_var = pandas.DataFrame(obs_het_per_var)
    return {"obs_het_per_var": obs_het_per_var}


def _count_alleles_per_var_for_chunk(chunk, pops, alleles, missing_gt, calc_freqs):
    gts = chunk[GT_ARRAY_ID]

    alleles_in_chunk = set(numpy.unique(gts)).difference([missing_gt])

    if alleles is not None:
        if alleles_in_chunk.difference(alleles):
            raise RuntimeError("Chunk more alleles than the given ones")
    else:
        alleles = sorted(alleles_in_chunk)

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

        if calc_freqs:
            expected_num_allelic_gts_in_snp = pop_gts.shape[1] * pop_gts.shape[2]
            num_allelic_gts_per_snp = expected_num_allelic_gts_in_snp - missing_counts
            allelic_freqs_per_snp = allele_counts / num_allelic_gts_per_snp
            result[pop_id]["allelic_freqs"] = allelic_freqs_per_snp

    return {"counts": result, "alleles": alleles_in_chunk}


def _calc_maf_per_var_for_chunk(chunk, pops, missing_gt):
    res = _count_alleles_per_var_for_chunk(
        chunk, pops, alleles=None, missing_gt=missing_gt, calc_freqs=True
    )
    major_allele_freqs = {}
    for pop, pop_res in res["counts"].items():
        pop_allelic_freqs = pop_res["allelic_freqs"]
        major_allele_freqs[pop] = pop_allelic_freqs.max(axis=1)
    major_allele_freqs = pandas.DataFrame(major_allele_freqs)
    return {"major_allele_freqs": major_allele_freqs}


def _update_remove_mask(remove_mask, new_remove_mask, filtering_info, filter_name):
    if remove_mask is None:
        remove_mask = new_remove_mask
    else:
        remove_mask = numpy.logical_or(remove_mask, new_remove_mask)

    num_vars_to_remove = numpy.sum(remove_mask)
    previous_num_vars_removed = filtering_info.get("vars_removed", 0)
    num_vars_removed_by_this_filter = num_vars_to_remove - previous_num_vars_removed

    try:
        num_vars_removed_per_filter = filtering_info["num_vars_removed_per_filter"]
    except KeyError:
        num_vars_removed_per_filter = {}
        filtering_info["num_vars_removed_per_filter"] = num_vars_removed_per_filter

    num_vars_removed_per_filter[filter_name] = num_vars_removed_by_this_filter

    return remove_mask


def _flt_chunk(chunk, max_var_obs_het, max_missing_rate):
    gts = chunk[GT_ARRAY_ID]

    remove_mask = None
    filtering_info = {}

    gt_is_missing = None
    if max_missing_rate < 1:
        gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)
        missing_rate = _calc_missing_rate_per_var(gts, gt_is_missing=gt_is_missing)
        this_remove_mask = missing_rate > max_missing_rate
        remove_mask = _update_remove_mask(
            remove_mask, this_remove_mask, filtering_info, "missing_rate"
        )
    if max_var_obs_het < 1:
        gt_is_missing = _calc_gt_is_missing(gts, gt_is_missing)
        obs_het_rate = _calc_obs_het_rate_per_var(gts, gt_is_missing=gt_is_missing)
        this_remove_mask = obs_het_rate > max_var_obs_het
        remove_mask = _update_remove_mask(
            remove_mask, this_remove_mask, filtering_info, "obs_het"
        )

    flt_chunk = {}
    for array_id, array in chunk.items():
        flt_array_chunk = filter_array_chunk_rows(array, numpy.logical_not(remove_mask))
        flt_chunk[array_id] = flt_array_chunk

    return {"chunk": flt_chunk, "filtering_info": filtering_info}


class Variants:
    def __init__(self, array_set_dir, missing_gt=MISSING_INT):
        self.array_set = ChunkedArraySet(dir=array_set_dir)
        self.missing_gt = missing_gt

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
    ):
        return self._count_alleles_per_var(
            pops=pops,
            alleles=alleles,
            calc_freqs=False,
        )

    def calc_allelic_freq_per_var(
        self,
        pops=None,
        alleles=None,
    ):
        return self._count_alleles_per_var(
            pops=pops,
            alleles=alleles,
            calc_freqs=True,
        )

    def _count_alleles_per_var(
        self,
        pops=None,
        alleles=None,
        calc_freqs=False,
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
            missing_gt=self.missing_gt,
            calc_freqs=calc_freqs,
        )

        try:
            collect_allele_counts = _AlleleCountCollector(
                num_rows=self.array_set.num_rows, calc_freqs=calc_freqs
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
                missing_gt=self.missing_gt,
            )
            result = self.array_set.run_pipeline(
                map_functs=[count_alleles_per_var_for_chunk],
                desired_arrays_to_load_in_chunk=[GT_ARRAY_ID],
                reduce_funct=collect_allele_counts,
                reduce_initialializer=None,
            )

        if not alleles_asked:
            if calc_freqs:
                main_result_key = "allelic_freqs"
            else:
                main_result_key = "allele_counts"
            # remove_extra_alleles
            alleles = sorted(result["alleles"])
            for pop, pop_result in result["counts"].items():
                result["counts"][pop][main_result_key] = pop_result[
                    main_result_key
                ].loc[:, alleles]

        result = result["counts"]

        if calc_freqs:
            result = {
                pop: pop_result["allelic_freqs"] for pop, pop_result in result.items()
            }

        return result

    def get_different_alleles(self):
        def accumulate_alleles(accumulated_alleles, new_alleles):
            accumulated_alleles.update(new_alleles)
            return accumulated_alleles

        result = self.array_set.run_pipeline(
            map_functs=[lambda chunk: numpy.unique(chunk[GT_ARRAY_ID])],
            desired_arrays_to_load_in_chunk=[GT_ARRAY_ID],
            reduce_funct=accumulate_alleles,
            reduce_initialializer=set(),
        )
        result = sorted(result.difference([self.missing_gt]))
        return result

    def calc_major_allele_freq_per_var(self, pops=None):
        pops = self._calc_pops_idxs(pops)

        calc_maf_per_var_for_chunk = partial(
            _calc_maf_per_var_for_chunk, pops=pops, missing_gt=self.missing_gt
        )

        collect_mafs = _ResultCollectorForArrayDict(num_rows=self.num_vars)

        result = self.array_set.run_pipeline(
            map_functs=[calc_maf_per_var_for_chunk],
            desired_arrays_to_load_in_chunk=[GT_ARRAY_ID],
            reduce_funct=collect_mafs,
            reduce_initialializer=None,
        )
        return result["major_allele_freqs"]

    def filter_variants(
        self,
        out_array_set: ChunkedArraySet,
        max_var_obs_het: float = 1.0,
        max_missing_rate: float = 1.0,
    ):
        filter_chunk = _ChunkFilterer(
            max_var_obs_het=max_var_obs_het,
            max_missing_rate=max_missing_rate,
        )
        flt_chunks = self.array_set.run_pipeline(
            map_functs=[filter_chunk],
        )
        out_array_set.extend_chunks(flt_chunks)
        print(filter_chunk.stats)


class _ChunkFilterer:
    def __init__(self, max_var_obs_het, max_missing_rate):
        self.max_var_obs_het = max_var_obs_het
        self.max_missing_rate = max_missing_rate
        self.stats = None

    def __call__(self, chunk):
        flt_chunk_res = _flt_chunk(
            chunk,
            max_var_obs_het=self.max_var_obs_het,
            max_missing_rate=self.max_missing_rate,
        )

        if self.stats is None:
            stats = {"num_vars_removed_per_filter": {}}
            self.stats = stats
        else:
            stats = self.stats

        for filter_name, vars_removed in flt_chunk_res["filtering_info"][
            "num_vars_removed_per_filter"
        ].items():
            num_vars_previously_removed = stats["num_vars_removed_per_filter"].get(
                filter_name, 0
            )
            stats["num_vars_removed_per_filter"][filter_name] = (
                num_vars_previously_removed + vars_removed
            )

        return flt_chunk_res["chunk"]

    def no_more_chunks_so_close(self):
        chunks = self.num_rows_normalizer.empty_last()
        try:
            chunk = self.array_set.extend_chunks(chunks)
        except RuntimeError:
            chunk = None
        if chunk:
            self.out_array_set.extend_chunks(chunk)


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
    def __init__(self, num_rows, calc_freqs):
        self.num_rows = num_rows
        self.row_start = 0
        self.calc_freqs = calc_freqs

    def __call__(self, complete_array_results, processed_chunk):
        if self.calc_freqs:
            main_result_key = "allelic_freqs"
            collect_missing_gts = False
        else:
            main_result_key = "allele_counts"
            collect_missing_gts = True

        if complete_array_results is None:
            # create empty complete arrays
            complete_array_results = {
                "counts": {pop: {} for pop in processed_chunk["counts"].keys()},
                "alleles": set(),
            }
            for pop, pop_result in processed_chunk["counts"].items():
                if collect_missing_gts:
                    complete_array_results["counts"][pop][
                        "missing_gts"
                    ] = create_empty_array_like(
                        pop_result["missing_gts"], num_rows=self.num_rows
                    )
                complete_array_results["counts"][pop][
                    main_result_key
                ] = create_empty_array_like(
                    pop_result[main_result_key], num_rows=self.num_rows
                )
                complete_array_results["counts"][pop]

        # put processed chunk in complete array
        row_end = None
        for pop, pop_result in processed_chunk["counts"].items():
            if row_end is None:
                chunk_num_rows = pop_result["missing_gts"].shape[0]
                row_end = self.row_start + chunk_num_rows

            for pop, pop_result in processed_chunk["counts"].items():
                if collect_missing_gts:
                    set_array_chunk(
                        complete_array_results["counts"][pop]["missing_gts"],
                        pop_result["missing_gts"],
                        self.row_start,
                        row_end,
                    )
                set_array_chunk(
                    complete_array_results["counts"][pop][main_result_key],
                    pop_result[main_result_key],
                    self.row_start,
                    row_end,
                )
        complete_array_results["alleles"].update(processed_chunk["alleles"])

        self.row_start = row_end
        return complete_array_results


if __name__ == "__main__":
    import pathlib
    import shutil

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
    if False:
        print(vars.calc_obs_het_per_var(pops=pops))
        # variants.calc_obs_het_per_var()
    elif True:
        array_set_dir = pathlib.Path("/home/jose/analyses/g2psol/tmp/flt_variants")
        if array_set_dir.exists():
            shutil.rmtree(
                array_set_dir,
            )

        vars.filter_variants(
            out_array_set=ChunkedArraySet(
                dir=array_set_dir,
                desired_num_rows_per_chunk=DEFAULT_NUM_VARIANTS_IN_CHUNK,
            ),
            max_var_obs_het=0.1,
            max_missing_rate=0.3,
        )
    elif True:
        print(vars.calc_major_allele_freq_per_var(pops=pops))
    elif False:
        print(vars.calc_allelic_freq_per_var(pops=pops))
    elif True:
        print(vars.count_alleles_per_var(pops=pops))
    elif False:
        print(vars.get_different_alleles())

"""
TODO

filter per genomic regions, for any region chrom== <pos<
unbiased exp. het.
procedure for the aprox. histograms when range is not given:
- count
- take random sample
- do histogram

"""
