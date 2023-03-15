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

