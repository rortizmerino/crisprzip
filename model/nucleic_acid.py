from abc import ABC
import random
from typing import Union, List

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike


def format_point_mutations(protospacer: str,
                           target_sequence: str) -> List[str]:
    """Compares protospacer and target sequence and writes point
    mutations as a list (e.g. [A04G, C07T])."""
    if len(protospacer) != len(target_sequence):
        raise ValueError("Protospacer and target should be equally long.")

    mm_list = []
    for i, b in enumerate(protospacer):
        if target_sequence[i] == b:
            continue

        mm_list += [f"{b}{i+1:02d}{target_sequence[i]}"]

    return mm_list


class MismatchPattern:
    """A class indicating the positions of the mismatched
    bases in a target sequence. Assumes a 3'-to-5' DNA direction.

    Attributes
    ----------
    pattern: np.ndarray
        Array with True indicating mismatched basepairs
    length: int
        Guide length
    mm_num: int
        Number of mismatches in the array
    is_on_target: bool
        Indicates whether the array is the on-target array

    Methods
    -------
    from_string(mm_array_string)
        Alternative constructor, reading strings
    from_mm_pos(guide_length[, mm_pos_list])
        Alternative constructor, based on mismatch positions
    make_random(guide_length, mm_num[, rng])
        Create mismatch array with randomly positioned mismatches
    get_mm_pos()
        Gives positions of the mismatches
    """

    def __init__(self, array: np.typing.ArrayLike):
        array = np.array(array)
        if array.ndim != 1:
            raise ValueError('Array should be 1-dimensional')
        if not (np.all((array == 0) | (array == 1)) or
                np.all((array is False) | (array is True)) or
                np.all((np.isclose(array, 0.0)) | (np.isclose(array, 0.0)))):
            raise ValueError('Array should only contain 0 and 1 values')

        self.pattern = np.asarray(array, dtype='bool')
        self.length = self.pattern.size
        self.mm_num = int(np.sum(self.pattern))
        self.is_on_target = self.mm_num == 0

    def __repr__(self):
        return "".join(["1" if mm else "0" for mm in self.pattern])

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_string(cls, mm_array_string):
        return cls(np.array(list(mm_array_string), dtype='int'))

    @classmethod
    def from_mm_pos(cls, guide_length: int, mm_pos_list: list = None,
                    zero_based_index=False):
        """Alternative constructor. Uses 1-based indexing by default. """
        array = np.zeros(guide_length)

        if not zero_based_index:
            mm_pos_list = [x - 1 for x in mm_pos_list]

        if mm_pos_list is not None:
            array[mm_pos_list] = 1
        return cls(array)

    @classmethod
    def from_target_sequence(cls, protospacer: str,
                             target_sequence: str) -> 'MismatchPattern':
        """Alternative constructor"""
        pmut_list = format_point_mutations(protospacer, target_sequence)
        return cls.from_mm_pos(
            len(protospacer),
            [int(pmut[1:3]) for pmut in pmut_list]
        )

    @classmethod
    def make_random(cls, guide_length: int, mm_num: int,
                    rng: Union[int, Generator] = None):
        if type(rng) is int or rng is None:
            rng = default_rng(rng)
        target = np.zeros(guide_length)
        mm_pos = rng.choice(range(20), size=mm_num, replace=False).tolist()
        target[mm_pos] = 1
        return cls(target)

    def get_mm_pos(self):
        return [i for i, mm in enumerate(self.pattern) if mm]


class NucleicAcid(ABC):
    pass


class SsNa(NucleicAcid, ABC):
    nucleotides: list

    def __init__(self, sequence: str, fwd_direction=True):
        sequence = sequence.upper()
        self.__check_sequence(sequence)
        self.sequence = sequence
        self.fwd_direction = fwd_direction  # fwd = 5'-to-3'direction

    def __len__(self):
        return len(self.sequence)

    def __repr__(self):
        if self.fwd_direction:
            return f"5\'-{self.sequence}-3\'"
        else:
            return f"3\'-{self.sequence}-5\'"

    def __check_sequence(self, sequence: str):
        for nt in sequence:
            if nt not in self.nucleotides:
                raise ValueError(f"Couldn't recognize nucleotide {nt}")

    @classmethod
    def make_random(cls, length, fwd_direction=True, seed=None) -> 'SsNa':
        random.seed(seed)
        return cls(''.join(random.choices(cls.nucleotides, k=length)),
                   fwd_direction)

    def flip(self) -> 'SsNa':
        return type(self)(
            self.sequence[::-1],
            not self.fwd_direction
        )


class SsDna(SsNa):
    nucleotides = ['A', 'C', 'G', 'T']

    def __repr__(self):
        return super().__repr__() + " (DNA)"


class SsRna(SsNa):
    nucleotides = ['A', 'C', 'G', 'U']

    def __repr__(self):
        return super().__repr__() + " (RNA)"


class DsNa(NucleicAcid, ABC):
    na_types: list
    basepairs: dict

    def __init__(self, strand1: Union[str, SsNa], strand2: Union[str, SsNa]):

        self.strand1 = (strand1 if isinstance(strand1, SsNa)
                        else self.na_types[0](strand1, fwd_direction=True))
        self.strand2 = (strand2 if isinstance(strand2, SsNa)
                        else self.na_types[1](strand2, fwd_direction=False))
        self.__check_length()
        self.__check_directionality()
        self.__check_na_types()

    def __len__(self):
        return len(self.strand1)

    def __repr__(self):
        return str(self.strand1) + "\n" + str(self.strand2)

    def __check_length(self):
        l1 = len(self.strand1)
        l2 = len(self.strand2)
        if l1 != l2:
            raise ValueError(f"Strands should have equal length but are "
                             f"{l1} and {l2} nt long.")

    def __check_directionality(self):
        d1 = self.strand1.fwd_direction
        d2 = self.strand2.fwd_direction
        if d1 == d2:
            raise ValueError(f"Strands should have opposite directionality"
                             f"but do not.")

    def __check_na_types(self):
        t1 = type(self.strand1)
        t2 = type(self.strand2)
        if t1 != self.na_types[0]:
            raise ValueError(f"Strand 1 should be of type {self.na_types[0]}")
        if t2 != self.na_types[1]:
            raise ValueError(f"Strand 2 should be of type {self.na_types[1]}")

    def __find_wc_basepairs(self):
        wc_basepairs = len(self) * [0, ]
        for i, nt in enumerate(self.strand1.sequence):
            if self.basepairs[nt] == self.strand2.sequence[i]:
                wc_basepairs[i] = 1
        return wc_basepairs

    def __make_labels(self, count):
        reps = len(self) // count + 1
        labs = ''.join(["{0:>{1}d}".format(count * i, count)
                        for i in range(1, reps + 1)])
        labs = ("1" + labs[1:len(self)-count] +
                "{0:>{1}d}".format(len(self), count))
        sep = 3 * " "
        return sep + labs + sep

    @classmethod
    def match_single_strand(cls, strand1: Union[str, SsNa]):
        strand1 = (strand1 if isinstance(strand1, SsNa)
                   else cls.na_types[0](strand1, fwd_direction=True))
        seq2 = ''
        for nt in strand1.sequence:
            seq2 += cls.basepairs[nt]
        strand2 = cls.na_types[1](sequence=seq2,
                                  fwd_direction=(not strand1.fwd_direction))
        print(type(strand2))
        return cls(strand1, strand2)

    def show(self, basepairing=True, count=5) -> None:
        if count is not None:
            print(self.__make_labels(count))
        print(str(self.strand1))
        if basepairing:
            wc_basepairs = self.__find_wc_basepairs()
            bp_string = (''.join(map(str, wc_basepairs))
                         .replace('0', ' ')
                         .replace('1', '|'))
            print(3*" " + bp_string + 3*" ")
        print(str(self.strand2))

    def flip(self):
        return type(self)(self.strand1.flip(), self.strand2.flip())

    def turn(self):
        return type(self)(self.strand2, self.strand1).flip()


class DsDna(DsNa):
    na_types = [SsDna, SsDna]
    basepairs = {'A': 'T',
                 'C': 'G',
                 'G': 'C',
                 'T': 'A'}


class DsRna(DsNa):
    na_types = [SsRna, SsRna]
    basepairs = {'A': 'U',
                 'C': 'G',
                 'G': 'C',
                 'U': 'A'}


class HybridNa(DsNa):
    na_types = [SsDna, SsRna]
    basepairs = {'A': 'U',
                 'C': 'G',
                 'G': 'C',
                 'T': 'A'}


class BaseStack(DsNa):
    pass
