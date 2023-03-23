"""
Contains classes representing nucleic acid in different ways. Primarily
supports the hybridization_kinetics module.

Classes:
    MismatchPattern
    TargetDna
    GuideTargetHybrid
    NearestNeighborModel

Functions:
    format_point_mutations()
"""

import json
import os
import random
from pathlib import Path
from typing import Union, List

import numpy as np
from joblib import Memory
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

        if mm_pos_list is None:
            mm_pos_list = []

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

# make temporary directory to store cache
tempdir = Path(os.environ["TMP"]).joinpath("crisprzipper")
memory = Memory(tempdir, verbose=0)


@memory.cache
def get_hybridization_energy(guide_sequence: str,
                             target_sequence: str = None,
                             nontarget_sequence: str = None,
                             upstream_nt: str = None,
                             downstream_nt: str = None,
                             fwd_direction: bool = True) -> np.ndarray:

    """
    Calculates the free energy cost associated with the progressive
    formation of an R-loop between an RNA guide and target DNA strand.

    Parameters
    ----------
    guide_sequence: str
        Sequence of the RNA guide, in 3'-to-5' notation.
    target_sequence: str
        Sequence of the DNA target strand (=spacer), in 5'-to-3'
        notation. If both target and nontarget sequence are specified
        target sequence overrules the other.
    nontarget_sequence: str
        Sequence of the DNA nontarget strand (=protospacer), in 5'-to-3'
        notation.
    upstream_nt: str
        Nucleotide that is positioned at the 5' side of the target strand.
        Corresponds to the 3rd nucleotide in the PAM motif.
    downstream_nt: str
        Nucleotide that is positioned at the 3' side of the target strand.
    fwd_direction: bool
        True by default. If False, all directionality is reversed and the
        upstream and downstream nucleotides are swapped.

    Returns
    -------
    hybridization_energy: np.ndarray
        Free energies required to create an R-loop.

    """

    # Recursively calling the same function for opposite directionality
    if not fwd_direction:
        return get_hybridization_energy(
            guide_sequence=guide_sequence[::-1],
            target_sequence=(None if target_sequence is None
                             else target_sequence[::-1]),
            nontarget_sequence=(None if nontarget_sequence is None
                                else nontarget_sequence[::-1]),
            upstream_nt=downstream_nt,
            downstream_nt=upstream_nt,
            fwd_direction=True
        )

    # Prepare target DNA and guide RNA
    if target_sequence is not None:
        target = TargetDna.from_target_strand(
            target_sequence=target_sequence,
            fwd_direction=True,
            upstream_nt=upstream_nt,
            downstream_nt=downstream_nt
        )
    elif nontarget_sequence is not None:
        target = TargetDna.from_nontarget_strand(
            nontarget_sequence=nontarget_sequence,
            fwd_direction=True,
            upstream_nt=upstream_nt,
            downstream_nt=downstream_nt
        )
    hybrid = GuideTargetHybrid(guide_sequence, target)

    # prepare NearestNeighborModel
    nnmodel = NearestNeighborModel()
    nnmodel.load_data()
    nnmodel.set_energy_unit("kbt")

    # do calculations
    hybridization_energy = nnmodel.get_hybridization_energy(hybrid)

    return hybridization_energy


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

        if fwd_direction and upstream_nt is not None:
            self.upstream_bp = (upstream_nt + "-" +
                                self.bp_map[upstream_nt])
        elif not fwd_direction and downstream_nt is not None:
            self.upstream_bp = (downstream_nt + "-" +
                                self.bp_map[downstream_nt])
        else:
            self.upstream_bp = None

        if fwd_direction and downstream_nt is not None:
            self.dnstream_bp = (downstream_nt + "-" +
                                self.bp_map[downstream_nt])
        elif not fwd_direction and upstream_nt is not None:
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

    def __str__(self):
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

    def get_mismatch_pattern(self) -> MismatchPattern:
        return MismatchPattern(self.find_mismatches())

    def __str__(self):
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


class NearestNeighborModel:
    """
    An implementation of the nearest neighbor model predicting energies
    for guide RNA-target DNA R-loops. Instantiating this class is only
    necessary to load the parameter files, a single object can be used
    to make all energy landscapes.

    Methods
    -------
    load_data()
        Loads the energy parameters.
    set_energy_unit()
        Change energy units betweel kBT and kcal/mol.
    get_hybridization_energy()
        Finds the hybridization energies for the R-loop formation.

    Notes
    -----
    Method adapted from Alkan et al. (2018). DNA duplex parameters from
    SantaLucia & Hicks (2004), RNA-DNA hybrid duplex parameters from
    Alkan et al. (2018).

    There are 4 contributions to the R-loop energy.

    1) Basestacks in the DNA duplex that should be broken. These
        parameters can be loaded directly from the SantaLucia & Hicks
        dataset. Unlike Alkan et al., we also consider basestacks with the
        basepairs flanking the target region. If these are unknown,
        we take the average energy from all 4 possible basestacks.
    2) Basestacks in the RNA/DNA hybrid that are created. Some of these
        energies are experimentally determined, others are an average
        of dsDNA and dsRNA values.
    3) Internal loops, corresponding to (regions of) mismatches flanked
        by matching basepairs. For internal loops of length 1 and 2,
        these have specific energies, for length > 2, their energies
        are the sum of the left and right basestack and a
        length-specific energy contribution.
    4) Basepair terminals at the end and beginning of the R-loop.
        Alkan et al. consider only external loops, which appear only when
        the guide-target hybrid starts or ends with a mismatch, but
        we always consider the energy contribution due to the first and
        last matching basepair. These energies are typically quite
        small.

    References
    ----------
    .. [1] Alkan F, Wenzel A, Anthon C, Havgaard JH, Gorodkin J (2018).
        CRISPR-Cas9 off-targeting assessment with nucleic acid duplex
        energy parameters. doi.org/10.1186/s13059-018-1534-x
    .. [2] SantaLucia J, Hicks D (2004). The Thermodynamics of DNA
        Structural Motifs. doi.org/10.1146/annurev.biophys.32.110601.141800

    """

    # paths relative to crisprzipper source root
    dna_dna_params_file = "nucleicacid_params/santaluciahicks2004.json"
    rna_dna_params_file = "nucleicacid_params/alkan2018.json"
    dna_dna_params: dict
    rna_dna_params: dict

    energy_unit = "kbt"  # alternative: kcalmol

    @classmethod
    def load_data(cls):
        with open(Path(__file__).parents[1].joinpath(cls.dna_dna_params_file),
                  'rb') as file:
            cls.dna_dna_params = json.load(file)

        with open(Path(__file__).parents[1].joinpath(cls.rna_dna_params_file),
                  'rb') as file:
            cls.rna_dna_params = json.load(file)

    @classmethod
    def set_energy_unit(cls, unit: str):
        cls.energy_unit = unit

    @classmethod
    def convert_units(cls, energy_value: Union[float, np.ndarray]):

        if cls.energy_unit == "kcalmol":
            return energy_value

        elif cls.energy_unit == "kbt":
            ref_temp = 310.15  # 310.15 deg K = 37 deg C
            gas_constant = 1.9872E-3  # R = N_A * k_B [units kcal / (K mol)]
            return energy_value / (gas_constant * ref_temp)

    @classmethod
    def get_hybridization_energy(cls, hybrid: GuideTargetHybrid) -> np.ndarray:
        """Calculates the energy that is required to open an R-loop
         between the guide RNA and target DNA of the hybrid object
         for each R-loop length. Converts energy units if necessary."""
        dna_opening_energy = cls.__dna_opening_energy(hybrid)
        rna_duplex_energy = cls.__rna_duplex_energy(hybrid)
        return cls.convert_units(
            dna_opening_energy + rna_duplex_energy
        )

    @classmethod
    def __dna_opening_energy(cls, hybrid: GuideTargetHybrid) -> np.ndarray:
        """Calculated following the methods from Alkan et al. (2018).
        The DNA opening energy is the sum of all the basestack energies
        in the sequence (negative)."""

        stacking_energies = cls.dna_dna_params["stacking energies"]

        def average_basestack(side='downstream'):
            """If down- or upstream basepair is unknown, this function
            loops over all options to find the average basestack
            energy."""
            basestack_energy = 0
            possible_bps = [('A', 'T'), ('C', 'G'),
                            ('G', 'C'), ('T', 'A')]

            for bp in possible_bps:
                if side == 'upstream':
                    basestack = (
                        f"d{bp[0]}{hybrid.target.seq1[0]}/"
                        f"d{bp[1]}{hybrid.target.seq2[0]}"
                    )
                elif side == 'downstream':
                    basestack = (
                        f"d{hybrid.target.seq1[-1]}{bp[0]}/"
                        f"d{hybrid.target.seq2[-1]}{bp[1]}"
                    )
                else:
                    raise ValueError("Side can only be downstream or"
                                     "upstream.")
                basestack_energy += (stacking_energies[basestack] / 4)
            return basestack_energy

        open_energy = np.zeros(len(hybrid.guide) + 1)

        # Handling the left basestack
        if hybrid.target.upstream_bp is not None:
            left_basestack = (f"d{hybrid.target.upstream_bp[0]}"
                              f"{hybrid.target.seq1[0]}/"
                              f"d{hybrid.target.upstream_bp[-1]}"
                              f"{hybrid.target.seq2[0]}")
            open_energy[1:] -= stacking_energies[left_basestack]
        else:
            open_energy[1:] -= average_basestack("upstream")

        # Handling right basestacks
        for i in range(0, len(hybrid.guide) - 1):
            right_basestack = (f"d{hybrid.target.seq1[i:i+2]}/"
                               f"d{hybrid.target.seq2[i:i+2]}")
            open_energy[i+1:] -= stacking_energies[right_basestack]

        # Final right basestack
        if hybrid.target.dnstream_bp is not None:
            left_basestack = (f"d{hybrid.target.seq1[-1]}"
                               f"{hybrid.target.dnstream_bp[0]}/"
                               f"d{hybrid.target.seq2[-1]}"
                               f"{hybrid.target.dnstream_bp[-1]}")
            open_energy[-1] -= stacking_energies[left_basestack]
        else:
            open_energy[-1] -= average_basestack("downstream")

        return open_energy

    @classmethod
    def __rna_duplex_energy(cls, hybrid: GuideTargetHybrid) -> np.ndarray:
        """Calculated following the methods from Alkan et al. (2018).
        The RNA duplex energy has three contributions: 1) basestacks,
        2) internal loops, 3) external loops / terminals. Alkan et al.
        only look at external loops, but here, we instead look
        at both basepair terminals, whether or not they are part of
        an external loop."""

        loop_energies = cls.rna_dna_params['loop energies']
        terminal_energies = cls.rna_dna_params['terminal penalties']
        stacking_energies = cls.rna_dna_params['stacking energies']

        def basestack_energy(i: int):
            """Locates basestacks in R-loop of length i and
            returns their total energy."""
            # # print(i)
            if i < 2:
                return 0.

            energy = 0.
            mm_pos = hybrid.find_mismatches()[:i]
            for j, nt in enumerate(hybrid.guide[:i-1]):
                if mm_pos[j] == 0 and mm_pos[j+1] == 0:
                    # Reversed order to match 5'-to-3' notation
                    basestack = f"r{hybrid.guide[j+1]}" \
                                f"{hybrid.guide[j]}/" \
                                f"d{hybrid.target.seq1[j+1]}" \
                                f"{hybrid.target.seq1[j]}"
                    energy += stacking_energies["2mer"][basestack]
            return energy

        def internal_loops_energy(i: int):
            """Locates internal loops in R-loop of length i and
            returns their total energy."""
            if i < 3:
                return 0.

            energy = 0.
            mm_pos = hybrid.find_mismatches()[:i]
            for j in range(1, i - 1):

                # identify left side of internal loop
                if mm_pos[j-1] == 0 and mm_pos[j] == 1:
                    looplen = 1

                    # find right side of internal loop
                    for k in range(j+1, i):
                        if mm_pos[k] == 0:
                            break
                        looplen += 1
                    # if no closing basepair has been found; stop
                    if j + looplen == i:
                        break

                    if looplen == 1:
                        # Reversed order to match 5'-to-3' notation
                        basestacks = (
                            f"r{hybrid.guide[j + 1]}"
                            f"{hybrid.guide[j]}"
                            f"{hybrid.guide[j - 1]}/"
                            f"d{hybrid.target.seq1[j + 1]}"
                            f"{hybrid.target.seq1[j]}"
                            f"{hybrid.target.seq1[j - 1]}"
                        )
                        energy += stacking_energies["3mer"][basestacks]

                    elif looplen == 2:
                        # Reversed order to match 5'-to-3' notation
                        basestacks = (
                            f"r{hybrid.guide[j + 2]}"
                            f"{hybrid.guide[j + 1]}"
                            f"{hybrid.guide[j]}"
                            f"{hybrid.guide[j - 1]}/"
                            f"d{hybrid.target.seq1[j + 2]}"
                            f"{hybrid.target.seq1[j + 1]}"
                            f"{hybrid.target.seq1[j]}"
                            f"{hybrid.target.seq1[j - 1]}"
                        )
                        energy += stacking_energies["4mer"][basestacks]

                    elif looplen >= 3:
                        left_basestack = (
                            f"r{hybrid.guide[j]}"
                            f"{hybrid.guide[j - 1]}/"
                            f"d{hybrid.target.seq1[j]}"
                            f"{hybrid.target.seq1[j - 1]}"
                        )
                        right_basestack = (
                            f"r{hybrid.guide[j+looplen]}"
                            f"{hybrid.guide[j+looplen-1]}/"
                            f"d{hybrid.target.seq1[j+looplen]}"
                            f"{hybrid.target.seq1[j+looplen-1]}"
                        )
                        energy += (
                            stacking_energies["2mer"][left_basestack] +
                            stacking_energies["2mer"][right_basestack] +
                            loop_energies[f"{2*looplen} nt"]
                        )

            return energy

        def external_loops_energy(i: int):
            """Locates external loops in R-loop of length i and
            returns their total energy. Unused right now because
            we're considering all terminal energies instead."""
            mm_pos = hybrid.find_mismatches()[:i]

            if sum(mm_pos) == len(mm_pos):
                return 0.

            energy = 0.

            # open left side
            if mm_pos[0] == 1:
                # find right side of internal loop
                for k in range(0, i):
                    if mm_pos[k] == 0:
                        basepair = f"r{hybrid.guide[k]}-" \
                                   f"d{hybrid.target.seq1[k]}"
                        energy += terminal_energies[basepair]
                        break

            # open right side
            if mm_pos[-1] == 1:
                # find right side of internal loop
                for k in range(i, 0, -1):
                    if mm_pos[k] == 0:
                        basepair = f"r{hybrid.guide[k]}-" \
                                   f"d{hybrid.target.seq1[k]}"
                        energy += terminal_energies[basepair]
                        break

            return energy

        def terminals_energy(i: int):
            """Locates terminals in R-loop of length i and
            returns their total energy."""

            mm_pos = hybrid.find_mismatches()[:i]
            if sum(mm_pos) == len(mm_pos):
                return 0.

            energy = 0.
            # find leftmost basepair
            for k in range(0, i):
                if mm_pos[k] == 0:
                    basepair = f"r{hybrid.guide[k]}-" \
                               f"d{hybrid.target.seq1[k]}"
                    energy += terminal_energies[basepair]
                    break

            # find rightmost basepair
            for k in list(np.arange(i, 0, -1) - 1):
                if mm_pos[k] == 0:
                    basepair = f"r{hybrid.guide[k]}-" \
                               f"d{hybrid.target.seq1[k]}"
                    energy += terminal_energies[basepair]
                    break

            return energy

        # Looping over R-loop states, adding all energy contributions
        duplex_energy = np.array([
            (basestack_energy(i) +
             internal_loops_energy(i) +
             terminals_energy(i))
            for i in range(len(hybrid.guide) + 1)
        ])
        return duplex_energy
