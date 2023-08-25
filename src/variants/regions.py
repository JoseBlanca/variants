import math
import functools

import numpy


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
