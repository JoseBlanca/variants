from pathlib import Path
import json
import copy
from collections.abc import Sequence

import numpy
import pandas

from variants.regions import GenomeLocation
from variants.globals import GT_ARRAY_ID

ArrayType = tuple[numpy.ndarray, pandas.DataFrame, pandas.Series, "Genotypes"]
ARRAY_FILE_EXTENSIONS = {
    "DataFrame": ".parquet",
    "ndarray": ".npy",
    "Genotypes": ".genotypes",
}


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
    if not isinstance(
        array, (numpy.ndarray, pandas.DataFrame, pandas.Series, Genotypes)
    ):
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
    elif isinstance(array, Genotypes):
        array.save(path)
        save_method = "genotypes"
    else:
        raise ValueError(f"Unknown array type: {type(array)}")

    return {"path": path, "save_method": save_method, "type_name": type_name}


def _get_array_num_rows(array: ArrayType):
    return array.shape[0]


def _get_array_rows(array, index):
    if isinstance(array, Genotypes):
        array = array.get_rows(index)
    elif isinstance(array, numpy.ndarray):
        array = array[index, ...]
    elif isinstance(array, pandas.DataFrame):
        array = array.iloc[index, :]
    elif isinstance(array, pandas.Series):
        array = array.iloc[index]
    else:
        raise ValueError(f"unknown array type: {type(array)}")
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
    elif isinstance(array1, Genotypes):
        assert array1.samples == array2.samples
        array1.values[index, ...] = array2.values
    else:
        raise ValueError(f"unknown array type: {type(array1)}")
    return array1


def _apply_mask(array, index):
    if isinstance(array, numpy.ndarray):
        array = array[index, ...]
    elif isinstance(array, pandas.DataFrame):
        array = array.loc[index, :]
    elif isinstance(array, pandas.Series):
        array = array.loc[index]
    elif isinstance(array, Genotypes):
        array = array.get_rows(index)
    else:
        raise ValueError(f"unknown array type: {type(array)}")
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


class Variants(ArraysChunk):
    @property
    def samples(self):
        return self.arrays[GT_ARRAY_ID].samples


def _as_chunk(chunk):
    if isinstance(chunk, ArraysChunk):
        return chunk
    else:
        return ArraysChunk(chunk)


def _build_genotypes_samples_path(path):
    return Path(str(path) + ".samples.json")


def _build_genotypes_gts_path(path):
    return Path(str(path) + ".gts.npy")


class Genotypes:
    def __init__(self, genotypes: numpy.ndarray, samples=Sequence[str]):
        assert genotypes.ndim == 3
        samples = list(samples)
        if genotypes.shape[1] != len(samples):
            raise ValueError(
                f"Number of samples in gts ({genotypes.shape[1]}) and number of given samples ({len(samples)}) do not match"
            )
        self._gts = genotypes
        self._samples = samples

    @property
    def samples(self):
        return self._samples

    @property
    def values(self):
        return self._gts

    @property
    def shape(self):
        return self._gts.shape

    def get_rows(self, index):
        gts = self._gts[index, ...]
        return self.__class__(genotypes=gts, samples=self.samples)

    def save(self, path):
        numpy.save(_build_genotypes_gts_path(path), self._gts)
        json_path = _build_genotypes_samples_path(path)
        json.dump(self._samples, json_path.open("wt"))

    @classmethod
    def load(cls, path):
        gts = numpy.load(_build_genotypes_gts_path(path))
        samples = json.load(_build_genotypes_samples_path(path).open("rt"))
        return cls(gts, samples)


def concat_genotypes(genotypes: Sequence[Genotypes]):
    gtss = [gts.values for gts in genotypes]
    gts = numpy.vstack(gtss)
    return Genotypes(genotypes=gts, samples=genotypes[0].samples)
