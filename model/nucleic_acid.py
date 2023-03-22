"""
Contains classes representing nucleic acid in different ways. Primarily
supports the hybridization_kinetics module.

Classes:
    MismatchPattern
    TargetDna
    GuideTargetHybrid

Functions:
    format_point_mutations()
"""


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


# General comment: I'm sorry for the misuse of the terms 'upstream'
# and 'downstream' below, they have not been used according to their
# formal definitions. It's inconvenient that literature reports
# most nucleic acid sequences in 5'-to-3' notation, but the R-loop
# formation follows the opposite direction (for Cas9, at least),
# making it logical to make pltos in 3'-to-5' notation. Just be careful
# with the directionality of your sequence input.

class TargetDna:
    """
    Represents the double-stranded DNA site to be opened during
    R-loop formation.

    Attributes
    ----------
    seq1: str
        The target strand (=spacer), in 5'-to-3' notation
    seq2: str
        The nontarget strand (=protospacer), in 3'-to-5'notation
    upstream_bp: str
        The basepair upstream (5'-side) of the target strand. For Cas9,
        corresponds to the last base of the PAM.
    dnstream_bp: str
        The basepair downstream of the target strand. For Cas9,
        is complementary to the last base of the PAM.

    Methods
    -------
    from_target_strand()
        Makes a TargetDna instance from the target strand (=spacer)
    from_nontarget_strand()
        Makes a TargetDna instance from the nontarget strand (=protospacer)
    """

    bp_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

    def __init__(self, target_sequence,
                 fwd_direction=True,
                 upstream_nt: str = None, downstream_nt: str = None):

        self.seq1 = (target_sequence if fwd_direction
                     else target_sequence[::-1])
        self.seq2 = self.__reverse_transcript(self.seq1)

        if upstream_nt:
            if fwd_direction:
                self.upstream_bp = (upstream_nt + "-" +
                                    self.bp_map[upstream_nt])
            else:
                self.upstream_bp = (downstream_nt + "-" +
                                    self.bp_map[downstream_nt])
        else:
            self.upstream_bp = None

        if downstream_nt:
            if fwd_direction:
                self.dnstream_bp = (downstream_nt + "-" +
                                    self.bp_map[downstream_nt])
            else:
                self.dnstream_bp = (upstream_nt + "-" +
                                    self.bp_map[upstream_nt])
        else:
            self.dnstream_bp = None

    @classmethod
    def __reverse_transcript(cls, sequence):
        """Gives complementary sequence (in opposite direction!)"""
        return ''.join([cls.bp_map[n] for n in sequence])

    @classmethod
    def from_target_strand(cls, target_sequence: str,
                           fwd_direction=True,
                           upstream_nt: str = None,
                           downstream_nt: str = None) -> 'TargetDna':
        """
        Makes a TargetDna instance from the target strand (=spacer)

        Arguments
        ---------
        target_sequence: str
            Nucleotide sequence of the target strand.
        fwd_direction: bool
            If True (default), target_sequence is read with 5'-to-3'
            direction. If False, read with 3'-to-5'.
        upstream_nt: str
            If fwd_direction is true: nucleotide at 5' side of the
            target sequence, corresponding to 3rd PAM nucleotide.
            If fwd_direction is false: nucleotide at the 3' side of the
            target sequence.
        downstream_nt:
            If fwd_direction is true: nucleotide at 5' side of the
            target sequence. If fwd_direction is false: nucleotide at
            the 3' side of the target sequence, corresponding to 3rd
            PAM nucleotide.
        """

        return cls(target_sequence, fwd_direction,
                   upstream_nt, downstream_nt)

    @classmethod
    def from_nontarget_strand(cls, nontarget_sequence,
                              fwd_direction=True,
                              upstream_nt: str = None,
                              downstream_nt: str = None) -> 'TargetDna':
        """
        Makes a TargetDna instance from the nontarget strand
        (=protospacer)

        Arguments
        ---------
        nontarget_sequence: str
            Nucleotide sequence of the nontarget strand.
        fwd_direction: bool
            If True (default), nontarget_sequence is read with 3'-to-5'
            direction. If False, read with 5'-to-3'.
        upstream_nt: str
            If fwd_direction is true: nucleotide at 3' side of the
            nontarget sequence, complementing the 3rd PAM nucleotide.
            If fwd_direction is false: nucleotide at the 5' side of the
            nontarget sequence.
        downstream_nt:
            If fwd_direction is true: nucleotide at 5' side of the
            nontarget sequence. If fwd_direction is false: nucleotide at
            the 3' side of the nontarget sequence, complementing to 3rd
            PAM nucleotide.
        """

        target_sequence = cls.__reverse_transcript(nontarget_sequence)

        return cls(target_sequence, fwd_direction,
                   upstream_nt=(None if downstream_nt is None
                                else cls.bp_map[downstream_nt]),
                   downstream_nt=(None if upstream_nt is None
                                  else cls.bp_map[upstream_nt]))

    @classmethod
    def make_random(cls, length, seed=None):
        random.seed(seed)
        nucleotides = list(cls.bp_map.keys())
        seq = ''.join(random.choices(nucleotides, k=length + 2))
        return cls(target_sequence=seq[1:length+1],
                   fwd_direction=True,
                   upstream_nt=seq[0],
                   downstream_nt=seq[-1])

    def __repr__(self):
        """Generates a handy string representation of the DNA duplex."""

        strand1 = self.seq1
        strand2 = self.seq2

        if self.upstream_bp:
            strand1 = self.upstream_bp[0] + strand1
            strand2 = self.upstream_bp[-1] + strand2
        else:
            strand1 = "N" + strand1
            strand2 = "N" + strand2

        if self.dnstream_bp:
            strand1 = strand1 + self.dnstream_bp[0]
            strand2 = strand2 + self.dnstream_bp[-1]
        else:
            strand1 = strand1 + "N"
            strand2 = strand2 + "N"

        # Formatting labels to indicate bp index
        count = 5
        reps = (len(self.seq1) - 3) // count
        labs = ''.join(["{0:>{1}d}".format(count * i, count)
                        for i in range(1, reps + 1)])
        labs = (
            " 1" +
            labs[1:len(self.seq1)] +
            "{0:>{1}d}".format(len(self.seq1),
                               len(self.seq1) - len(labs))
        )

        return "\n".join([
            f"5\'-{strand1}-3\' (DNA target strand)",
            3 * " " + (2 + len(self.seq1)) * "|",
            f"3\'-{strand2}-5\' (DNA nontarget strand)",
            3 * " " + labs
        ])


class GuideTargetHybrid:
    """
    Represents an ssRNA guide interacting with ds DNA site through
    R-loop formation.

    Attributes
    ----------
    guide: str
        The RNA guide strand, in 3'-to-5' notation
    target: TargetDna
        The dsDNA site to be interrogated
    state: int
        Length of the R-loop. Only for illustration purposes for now.

    Methods
    -------
    set_rloop_state()
        Update the R-loop state.
    find_mismatches()
        Identifies the positions of mismatching guide-target basepairs.
    """

    bp_map = {'A': 'T', 'C': 'G', 'G': 'C', 'U': 'A'}

    def __init__(self, guide: str, target: TargetDna):
        self.guide = guide
        self.target = target
        self.state = 0

    def set_rloop_state(self, rloop_state):
        self.state = rloop_state

    def find_mismatches(self):
        """Identifies the positions of mismatching guide-target
        basepairs."""
        mismatches = []
        for i, n in enumerate(self.guide):
            mismatches += [0 if self.bp_map[n] == self.target.seq1[i]
                           else 1]
        return mismatches

    def __repr__(self):
        """Generates a handy string representation of the R-loop.
        Use set_rloop_state() to update this representation."""
        dna_repr = str(self.target).split('\n')
        hybrid_bps = (''.join(map(str, self.find_mismatches()))
                         .replace('1', "\u00B7")
                         .replace('0', '|'))
        return "\n".join([
            f" 3\'-{self.guide}-5\'  (RNA guide)",
            4 * " " + hybrid_bps[:self.state],
            dna_repr[0],
            "   |" + self.state * " " + dna_repr[1][self.state+4:],
            dna_repr[2],
            dna_repr[3]
        ])

