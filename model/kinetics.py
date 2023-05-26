"""
The kinetics module is the core of the CRISPRzipper model.
It defines the basic properties of a CRISPR(-like) searcher and uses
these to simulate its R-loop hybridization dynamics.

Classes:
    MismatchPattern
    Searcher
    SearcherTargetComplex(Searcher)
"""

from typing import *

import numpy as np
from numba import njit
from numpy.typing import ArrayLike

from .matrix_expon import *
from .nucleic_acid import MismatchPattern, GuideTargetHybrid, \
    get_hybridization_energy

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
from .nucleic_acid import MismatchPattern, GuideTargetHybrid, \
    get_hybridization_energy


class Searcher:
    """Characterizes the hybridization landscape of a nucleic acid guided
    searcher. Assumes a reference concentration of 1 nM.

    Attributes
    ----------
    guide_length: int
        N, length of the nucleic acid guide (in bp)
    on_target_landscape: array_like
        Contains the hybridization energies of the intermediate R-loop
        states on an on-target, relative to the PAM energy. In presence
        of a PAM state, it has length N (otherwise N-1).
    mismatch_penalties: array_like
        Contains the energetic penalties associated with a mismatch
        at a particular R-loop position. Has length N.
    internal_rates: dict
        Specifies the context-independent rates in the model. Should
        contain 'k_off', 'k_f' and 'k_clv'.
    pam_detection: bool
        If true, the landscape includes a PAM recognition state. True
        by default.

    Methods
    -------
    probe_target(target_mismatches)
        Returns a SearcherTargetComplex object
    plot_landscape()
        Creates a line plot of the on- or off-target landscape
    plot_penalties()
        Creates a bar plot of the mismatch penalties
    """

    def __init__(self,
                 on_target_landscape: ArrayLike,
                 mismatch_penalties: ArrayLike,
                 internal_rates: dict,
                 pam_detection=True):
        """Constructor method"""

        # convert on_target_landscape and mismatch_penalties to numpy
        if (type(on_target_landscape) != np.ndarray or
                type(mismatch_penalties) != np.ndarray):
            on_target_landscape = np.array(on_target_landscape)
            mismatch_penalties = np.array(mismatch_penalties)

        # check whether parameters are 1d arrays
        if on_target_landscape.ndim > 1 or mismatch_penalties.ndim > 1:
            raise ValueError('Landscape parameters must be 1d arrays')

        # check whether landscape dimensions agree with guide length
        guide_length = mismatch_penalties.size
        if on_target_landscape.size != pam_detection + guide_length - 1:
            raise ValueError('Landscape dimensions do not match guide length')

        # check whether internal_rates dictionary contains proper keys
        if not ('k_off' in internal_rates and
                'k_f' in internal_rates and
                'k_clv' in internal_rates):
            raise ValueError('Forward rates dictionary should include k_off, '
                             'k_f and k_clv as keys')

        # assign values
        self.guide_length = guide_length
        self.pam_detection = pam_detection

        self.on_target_landscape = on_target_landscape
        self.mismatch_penalties = mismatch_penalties
        self.internal_rates = internal_rates

    def get_forward_rate_array(self, k_on):
        """Turns the forward rate dictionary into proper array"""
        forward_rate_array = np.concatenate(
            #  solution state
            (k_on * np.ones(1),
             # PAM and intermediate R-loop states
             self.internal_rates['k_f'] *
             np.ones(self.on_target_landscape.size),
             # final/complete R-loop state
             self.internal_rates['k_clv'] * np.ones(1),
             # cleaved state
             np.zeros(1))
        )
        return forward_rate_array

    def generate_dead_clone(self):
        """Returns Searcher object with zero catalytic activity"""
        dead_forward_rate_dict = self.internal_rates.copy()
        dead_forward_rate_dict['k_clv'] = 0
        dead_searcher = Searcher(
            on_target_landscape=self.on_target_landscape,
            mismatch_penalties=self.mismatch_penalties,
            internal_rates=dead_forward_rate_dict
        )
        return dead_searcher

    def probe_target(self, target_mismatches: MismatchPattern) \
            -> 'SearcherTargetComplex':
        return SearcherTargetComplex(self.on_target_landscape,
                                     self.mismatch_penalties,
                                     self.internal_rates,
                                     target_mismatches)

    def probe_explicit_target(self, guide_target_hybrid: GuideTargetHybrid) \
            -> 'SearcherSequenceComplex':
        return SearcherSequenceComplex(self.on_target_landscape,
                                       self.mismatch_penalties,
                                       self.internal_rates,
                                       guide_target_hybrid)

    def calculate_solution_energy(self, k_on):
        """Given an on-rate, returns the effective free energy of the
        solution state (under the assumption of detailed balance)"""
        return np.log(k_on / self.internal_rates['k_off'])


class SearcherTargetComplex(Searcher):
    """
    Characterizes the hybridization landscape of a nucleic acid guided
    searcher on a particular (off-)target sequence. Assumes a reference
    concentration of 1 nM.

    Attributes
    ----------
    target_mismatches: ndarray
        Positions of mismatches in the guide-target hybrid: has length
        N, with entries 0 (matches) and 1 (mismatches).
    off_target_landscape: ndarray
        Contains the hybridization energies of the intermediate R-loop
        states on the current off-target.  In presence
        of a PAM state, it has length N (otherwise N-1).

    Methods
    -------
    get_cleavage_probability()
        Returns the probability that a searcher in the PAM state (if
        present, otherwise b=1) cleaves a target before having left it
        (for active searchers)
    solve_master_equation()
        Solves Master equation, giving time evolution of the landscape
        occupancy
    get_cleaved_fraction()
        Returns the fraction of cleaved targets after a specified time
        (for active searchers)
    get_bound_fraction()
        Returns the fraction of bound targets after a specified time
        (for dead searchers)
    plot_landscape()
        Creates a line plot of the off-target landscape
    """

    def __init__(self, on_target_landscape: np.ndarray,
                 mismatch_penalties: np.ndarray, internal_rates: dict,
                 target_mismatches: MismatchPattern):
        Searcher.__init__(self, on_target_landscape, mismatch_penalties,
                          internal_rates)

        # check dimensions of mismatch position array
        if target_mismatches.length != self.guide_length:
            raise ValueError('Target array should be of same length as guide')
        else:
            self.target_mismatches = target_mismatches

        self.off_target_landscape = self._get_off_target_landscape()
        self.backward_rate_array = self._get_backward_rate_array()

    def generate_dead_clone(self):
        """Returns SearcherTargetComplex object with zero catalytic
        activity"""
        dead_searcher = Searcher.generate_dead_clone(self)
        dead_complex = dead_searcher.probe_target(self.target_mismatches)
        return dead_complex

    def _get_off_target_landscape(self):
        """Adds penalties due to mismatches to landscape"""
        landscape_penalties = np.cumsum(
            self.target_mismatches.pattern *
            self.mismatch_penalties
        )
        if self.pam_detection:
            off_target_landscape = (self.on_target_landscape +
                                    landscape_penalties)
        else:
            # # without PAM detection, the penalty on the first state
            # # is not added yet to have a consistent output
            # off_target_landscape = (self.on_target_landscape +
            #                         landscape_penalties[1:])
            raise ValueError('No support yet for off target landscape'
                             'without PAM detection')

        return off_target_landscape

    def _get_landscape_diff(self):
        """Returns the difference between landscape states"""
        if self.pam_detection:
            hybrid_landscape = np.concatenate((
                np.zeros(1),  # preceding zero representing the PAM state
                self.off_target_landscape
            ))
        else:
            # add potential penalty on the first state (= not PAM)
            hybrid_landscape = np.concatenate((
                np.array([
                    (self.target_mismatches.pattern[0] *
                     self.mismatch_penalties[0])
                ]),
                self.off_target_landscape
            ))
        return np.diff(hybrid_landscape, prepend=np.zeros(1))

    def _get_backward_rate_array(self):
        """Obtains backward rates from detailed balance condition"""
        boltzmann_factors = np.exp(self._get_landscape_diff())
        backward_rate_array = np.concatenate(
            #  solution state
            (np.zeros(1),
             # PAM state
             np.array([self.internal_rates['k_off']]),
             # R-loop states
             self.internal_rates['k_f'] * boltzmann_factors[1:],
             # cleaved state
             np.zeros(1))
        )
        return backward_rate_array

    def get_rate_matrix(self, on_rate: float) -> np.ndarray:
        """Sets up the rate matrix describing the master equation"""

        # shallow copy to prevent overwriting due to concentration
        forward_rates = self.get_forward_rate_array(k_on=on_rate)
        backward_rates = self.backward_rate_array.copy()

        diagonal1 = -(forward_rates + backward_rates)
        diagonal2 = backward_rates[1:]
        diagonal3 = forward_rates[:-1]
        rate_matrix = (np.diag(diagonal1, k=0) +
                       np.diag(diagonal2, k=1) +
                       np.diag(diagonal3, k=-1))

        return rate_matrix

    def solve_master_equation(self, initial_condition: np.ndarray,
                              time: Union[float, np.ndarray],
                              on_rate: Union[float, np.ndarray],
                              rebinding=True, mode='fast') -> np.ndarray:

        """
        Calculates how the occupancy of the landscape states evolves by
        evaluating the master equation. Absorbing states (solution and
        cleaved state) are explicitly incorporated. Can vary either
        time or on_rate but not both.

        Parameters
        ----------
        initial_condition: ndarray
            Vector showing the initial occupancy on the hybridization
            landscape. Has length guide_length+3 (if PAM_detection is
            true), and should sum to 1.
        time: Union[float, np.ndarray]
            Times at which the master equation is evaluated.
        on_rate: Union[float, np.ndarray]
            Rate (Hz) with which the searcher binds the target from solution.
        rebinding: bool
            If true, on-rate is left intact, if false, on-rate is set
            to zero and solution state becomes absorbing.
        mode: str
            If 'fast' (default), uses Numba implementation to do fast
            matrix exponentiation. If 'iterative', uses the
            (~20x slower) iterative procedure. Whenever the fast
            implementation breaks down, falls back to the iterative.

        Returns
        -------
        landscape_occupancy: ndarray
            Occupancy of the landscape states at specified time. Has
            length guide_length+3 (if PAM_detection is true), and sums
            to 1.
        """

        # check dimensions initial condition
        if initial_condition.size != (3 + self.on_target_landscape.size):
            raise ValueError('Initial condition should be of same length as'
                             'hybridization landscape')

        # if rebinding is prohibited, on-rate should be zero
        if not rebinding:
            on_rate = 0.

        # determines whether to sweep time or k_on (not both)
        vary_time = (False if ((not isinstance(time, np.ndarray)) or
                               time.size == 1)
                     else True)
        vary_k_on = (False if ((not isinstance(on_rate, np.ndarray)) or
                               on_rate.size == 1)
                     else True)

        # variable time and k_on: no support (yet)
        if vary_time and vary_k_on:
            raise ValueError("Cannot iterate over both time and k_on.")

        # unique time & k_on: handle as variable time
        if not (vary_time or vary_k_on):
            vary_time = True

        # variable time
        if vary_time:
            rate_matrix = self.get_rate_matrix(on_rate)

            # trivial case
            if not isinstance(time, np.ndarray) and np.isclose(time, 0.):
                return initial_condition

            # making sure that time is a 1d ndarray
            time = np.atleast_1d(time)

            # where the magic happens; evaluating the master equation
            if mode == 'fast':
                exp_matrix = exponentiate_fast(rate_matrix, time)

                # this is a safe alternative for exponentiate_fast
                if exp_matrix is None:
                    exp_matrix = exponentiate_iterative(rate_matrix, time)

            elif mode == 'iterative':
                exp_matrix = exponentiate_iterative(rate_matrix, time)
            else:
                raise ValueError(f'Cannot recognize mode {mode}')

        # variable k_on
        elif vary_k_on:
            # This reference rate matrix will be updated repeatedly
            ref_rate_matrix = self.get_rate_matrix(0.)

            # where the magic happens; evaluating the master equation
            if mode == 'fast':
                exp_matrix = exponentiate_fast_var_onrate(
                    ref_rate_matrix, float(time), on_rate
                )

                # this is a safe alternative for exponentiate_fast
                if exp_matrix is None:
                    exp_matrix = exponentiate_iterative_var_onrate(
                        ref_rate_matrix, time, on_rate
                    )

            elif mode == 'iterative':
                exp_matrix = exponentiate_iterative_var_onrate(
                    ref_rate_matrix, time, on_rate
                )
            else:
                raise ValueError(f'Cannot recognize mode {mode}')

        # This case should never be true
        else:
            raise Exception

        # Shared final maths for variable time & on_rate

        # calculate occupancy: P(t) = exp(Mt) P0
        landscape_occupancy = exp_matrix.dot(initial_condition)

        # avoid negative occupancy (if present, these should be tiny)
        landscape_occupancy = np.maximum(landscape_occupancy,
                                         np.zeros(landscape_occupancy.shape))

        # normalize P(t) to correct for rounding errors
        total_occupancy = np.sum(landscape_occupancy, axis=0)

        # recognize unsafe entries for division (zero/nan/inf)
        unsafe = np.any(np.stack((total_occupancy == 0.,
                                  np.isnan(total_occupancy),
                                  np.isinf(total_occupancy)),
                                 axis=1),
                        axis=1)

        # normalize or assign nan
        landscape_occupancy[:, ~unsafe] = (landscape_occupancy[:, ~unsafe] /
                                           total_occupancy[~unsafe])
        landscape_occupancy[:, unsafe] = (landscape_occupancy[:, unsafe] *
                                          float('nan'))

        return np.squeeze(landscape_occupancy.T)

    def get_cleavage_probability(self) -> float:
        """Returns the probability that a searcher in the PAM state (if
        present, otherwise b=1) cleaves a target before having left
        it."""

        forward_rates = self.get_forward_rate_array[1:-1]
        backward_rates = self.backward_rate_array[1:-1]
        gamma = backward_rates / forward_rates
        cleavage_probability = 1 / (1 + gamma.cumprod().sum())
        return cleavage_probability

    def get_cleaved_fraction(self, time: Union[float, np.ndarray],
                             on_rate: float = 1E-3) -> np.ndarray:
        """
        Returns the fraction of cleaved targets after a specified time

        Parameters
        ----------
        time: Union[float, np.ndarray]
            Times at which the cleaved fraction is calculated
        on_rate: float
            Rate (Hz) with which the searcher binds the target from solution.

        Returns
        -------
        cleaved_fraction: np.ndarray
            Fraction of targets that is expected to be cleaved by time
            t.
        """

        unbound_state = np.concatenate(
            (np.ones(1), np.zeros(self.on_target_landscape.size + 2))
        )
        prob_distr = self.solve_master_equation(unbound_state, time,
                                                on_rate)
        cleaved_fraction = prob_distr.T[-1]
        return cleaved_fraction

    def get_bound_fraction(self, time: float,
                           on_rate: Union[float, np.ndarray] = 1E-3) \
            -> np.ndarray:
        """
        Returns the fraction of bound targets after a specified time,
        assuming that searcher is catalytically dead/inactive.

        Parameters
        ----------
        time: float
            Time at which the cleaved fraction is calculated
        on_rate: Union[float, np.ndarray]
            Rates (Hz) with which the searcher binds the target from solution.

        Returns
        -------
        cleaved_fraction: array_like
            Fraction of targets that is expected to be bound by time
            t and with binding rates on_rate.
        """

        unbound_state = np.concatenate(
            (np.ones(1), np.zeros(self.on_target_landscape.size + 2))
        )
        # setting up clone SearcherTargetComplex object with zero
        # catalytic activity, k_clv=0
        dead_searcher_complex = self.generate_dead_clone()

        prob_distr = \
            dead_searcher_complex.solve_master_equation(unbound_state, time,
                                                        on_rate)
        bound_fraction = 1 - prob_distr.T[0]
        return bound_fraction


class SearcherSequenceComplex(SearcherTargetComplex):

    def __init__(self, on_target_landscape: np.ndarray,
                 mismatch_penalties: np.ndarray, internal_rates: dict,
                 guide_target_hybrid: GuideTargetHybrid):

        self.hybrid = guide_target_hybrid
        mismatch_pattern = guide_target_hybrid.get_mismatch_pattern()
        super().__init__(on_target_landscape, mismatch_penalties,
                         internal_rates, mismatch_pattern)

    def _get_off_target_landscape(self):
        internal_na_energy = get_hybridization_energy(
            guide_sequence=self.hybrid.guide,
            target_sequence=self.hybrid.target.seq1,
            upstream_nt=(None if self.hybrid.target.upstream_bp is None
                         else self.hybrid.target.upstream_bp[0]),
            downstream_nt=(None if self.hybrid.target.dnstream_bp is None
                           else self.hybrid.target.dnstream_bp[0])
        )[1:]
        protein_na_energy = (
            SearcherTargetComplex._get_off_target_landscape(self)
        )
        return protein_na_energy + internal_na_energy

    def generate_dead_clone(self):
        """Returns SearcherSequenceComplex object with zero catalytic
        activity"""
        dead_searcher = Searcher.generate_dead_clone(self)
        dead_complex = dead_searcher.probe_explicit_target(self.hybrid)
        return dead_complex
