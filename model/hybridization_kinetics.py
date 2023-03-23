"""
The hybridization_kinetics module is the core of the CRISPRzipper model.
It defines the basic properties of a CRISPR(-like) searcher and uses
these to simulate its dynamics. It also has visualization functionality.

Classes:
    Searcher
    SearcherTargetComplex(Searcher)
    CoarseGrainedComplex(SearcherTargetComplex)
    SearcherPlotter

Functions:
    coarse_grain_landscape(searcher_target_complex, itermediate_range=(7, 14))
"""

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.linalg import inv

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
        # self.forward_rate_array = self.__get_forward_rate_array()

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
                              time: ArrayLike,
                              on_rate: float,
                              rebinding=True) -> np.ndarray:

        """
        Calculates how the occupancy of the landscape states evolves by
        evaluating the master equation. Absorbing states (solution and
        cleaved state) are explicitly incorporated.

        Parameters
        ----------
        initial_condition: ndarray
            Vector showing the initial occupancy on the hybridization
            landscape. Has length guide_length+3 (if PAM_detection is
            true), and should sum to 1.
        time: array_like
            Times at which the master equation is evaluated.
        on_rate: float
            Rate (Hz) with which the searcher binds the target from solution.
        rebinding: bool
            If true, on-rate is left intact, if false, on-rate is set
            to zero and solution state becomes absorbing.

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
        rate_matrix = self.get_rate_matrix(on_rate)

        # if rebinding is prohibited, on-rate should be zero
        if not rebinding:
            rate_matrix[:, 0] = 0

        # trivial case
        if time is int(0):
            return initial_condition

        # making sure that time is a 1d ndarray
        time = np.atleast_1d(time)

        # where the magic happens; evaluating the master equation
        exp_matrix = _exponentiate_fast(rate_matrix, time)
        if exp_matrix is None:  # this is a safe alternative for e
            exp_matrix = _exponentiate_iterative(rate_matrix, time)

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

    def get_cleaved_fraction(self, time: ArrayLike,
                             on_rate: float = 1E-3) -> ArrayLike:
        """
        Returns the fraction of cleaved targets after a specified time

        Parameters
        ----------
        time: array_like
            Times at which the cleaved fraction is calculated
        on_rate: float
            Rate (Hz) with which the searcher binds the target from solution.

        Returns
        -------
        cleaved_fraction: array_like
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

    def get_bound_fraction(self, time: ArrayLike,
                           on_rate: float = 1E-3) -> ArrayLike:
        """
        Returns the fraction of bound targets after a specified time,
        assuming that searcher is catalytically dead/inactive.

        Parameters
        ----------
        time: array_like
            Times at which the cleaved fraction is calculated
        on_rate: float
            Rate (Hz) with which the searcher binds the target from solution.

        Returns
        -------
        cleaved_fraction: array_like
            Fraction of targets that is expected to be bound by time
            t.
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


def _exponentiate_fast(matrix: np.ndarray, time: np.ndarray):
    """
    Fast method to calculate exp(M*t), by diagonalizing matrix M.
    Returns None if diagnolization is problematic:
    - singular rate matrix
    - rate matrix with complex eigenvals
    - overflow in the exponentiatial
    - negative terms in exp_matrix
    """

    # 1. diagonalize M = U D U_inv
    try:
        eigenvals, eigenvecs = linalg.eig(matrix)
        eigenvecs_inv = linalg.inv(eigenvecs)
    except np.linalg.LinAlgError:
        return None  # handles singular matrices
    if np.any(np.iscomplex(eigenvals)):
        return None  # skip rate matrices with complex eigenvalues

    # 2. B = exp(Dt)
    exponent_matrix = np.tensordot(eigenvals.real, time, axes=0)
    if np.any(exponent_matrix > 700.):
        return None  # prevents np.exp overflow
    b_matrix = np.exp(exponent_matrix)
    diag_b_matrix = (b_matrix *  # 2D b_matrix put on 3D diagonal
                     np.repeat(np.eye(b_matrix.shape[0])[:, :, np.newaxis],
                               b_matrix.shape[1], axis=2))

    # 3. exp(Mt) = U B U_inv = U exp(Dt) U_inv
    exp_matrix = np.tensordot(eigenvecs.dot(diag_b_matrix),
                              eigenvecs_inv,
                              axes=((1,), (0,)))

    if np.any(exp_matrix < -1E-3):
        return None  # strong negative terms

    return exp_matrix


def _exponentiate_iterative(matrix: np.ndarray, time: np.ndarray):
    """The safer method to calculate exp(M*t), looping over the values
    in t and using the scipy function for matrix exponentiation."""

    exp_matrix = np.zeros(shape=((matrix.shape[0],
                                  time.shape[0],
                                  matrix.shape[1])))
    for i in range(time.size):
        exp_matrix[:, i, :] = linalg.expm(matrix * time[i])
    return exp_matrix


def coarse_grain_landscape(searcher_target_complex: SearcherTargetComplex,
                           intermediate_range: Tuple[int] = (7, 14)):
    """Calculates the coarse-grained rates over the two barrier regions
    that are expected to exist in a hybridization landscape.

    Parameters
    ----------
    searcher_target_complex: SearcherTargetComplex
        The off-target hybridization landscape from this instance is
        used to calculate the coarse-grained rates from
    intermediate_range: tuple
        The states a and b between which intermediate state is looked
        for. The lowest-energy state in the range [a, b) is the
        intermediate state. Default is [7, 14).

    Returns
    -------
    coarse_grained_rates: dict
        Dictionary containing the rates k_OI, k_IC, k_IO, k_CI
        (Open, Intermediate, Closed). Confusingly, these have later
        been renamed (O -> P for PAM and C -> O).
    intermediate_id: int
        Location of the intermediate state.

    """
    return CoarseGrainedComplex(
        searcher_target_complex.on_target_landscape,
        searcher_target_complex.mismatch_penalties,
        searcher_target_complex.internal_rates,
        searcher_target_complex.target_mismatches
    ).get_coarse_grained_rates(intermediate_range)


class CoarseGrainedComplex(SearcherTargetComplex):
    """Object to calculate the coarse grained rates over the barrier
    regions in a SearcherTargetComplex hybridization landscape. This
    object is the backend to the calculation, the recommended usage
    is the function coarse_grain_landscape defined above."""

    def get_coarse_grained_rates(self, intermediate_range=(7, 14)):
        """
        Calculates the coarse-grained rates between the open (=PAM),
        intermediate, and closed (=bound) state. Assumes PAM sensing.

        Parameters
        ----------
        intermediate_range: tuple
            The states a and b between which intermediate state is looked
            for. The lowest-energy state in the range [a, b) is the
            intermediate state. Default is [7, 14).

        Returns
        -------
        cg_rates: dict
            Dictionary containging the rates k_OI, k_IC, k_IO, k_CI
            (Open, Intermediate, Closed). Confusingly, these have later
            been renamed (O -> P for PAM and C -> O).
        intermediate_id: int
            Location of the intermediate state.

        """
        # Because the target landscape is defined to have length N,
        # instead of N+1, we add one to the intermediate id
        intermediate_id = 1 + (
                np.argmin(self.off_target_landscape[intermediate_range[0]:
                                                    intermediate_range[1]]) +
                intermediate_range[0]
        )

        # These follow the old definitions, so have N+1 nonzero energy states
        # and 2 zero energy states (solution / cleaved)
        kf = self.get_forward_rate_array(k_on=0.)
        kb = self.backward_rate_array

        # From open (=PAM) to intermediate
        k_oi = self.__calculate_effective_rate(kf, kb,
                                               start=1,
                                               stop=1 + intermediate_id)

        # From intermediate to closed (=bound)
        k_ic = self.__calculate_effective_rate(kf, kb,
                                               start=1 + intermediate_id,
                                               stop=len(kf) - 2)

        # From intermediate to open (=PAM)
        k_io = self.__calculate_effective_rate(kb[::-1], kf[::-1],
                                               start=(len(kf) - 1) - (
                                                       1 + intermediate_id),
                                               stop=(len(kf) - 1) - 1)

        # From closed (=bound) to intermediate
        k_ci = self.__calculate_effective_rate(kb[::-1], kf[::-1],
                                               start=1,
                                               stop=(len(kf) - 1) - (
                                                       1 + intermediate_id))

        cg_rates = {'k_OI': k_oi, 'k_IC': k_ic, 'k_IO': k_io, 'k_CI': k_ci}
        return cg_rates, intermediate_id

    @classmethod
    def __calculate_effective_rate(cls, forward_rate_array,
                                   backward_rate_array, start=0, stop=None):
        """Calculates the effective rate from state start to state stop.
        This calculation is from the old supplementary materials of
        Eslami et al."""

        partial_k = cls.__setup_partial_rate_matrix(forward_rate_array,
                                                    backward_rate_array,
                                                    start=start, stop=stop,
                                                    final_state_absorbs=True)

        initial_state = np.zeros(partial_k.shape[0] - 1)
        initial_state[0] = 1.

        # Eslami et al.
        eff_rate = np.matmul(-inv(partial_k[:-1, :-1]),
                             initial_state).sum() ** -1

        return eff_rate

    @staticmethod
    def __setup_partial_rate_matrix(forward_rate_array: np.ndarray,
                                    backward_rate_array: np.ndarray,
                                    start: int = 0, stop: int = None,
                                    final_state_absorbs=False) -> np.ndarray:
        """Returns a rate matrix of the states between start and stop
        (including state 'stop'). Rate matrix set up is just like that in
        the SearcherTargetComplex class. Stop=None gives everything up to
        the final state. Final_state_absorbs=True makes the final state
        absorbing."""

        if stop is None:
            partial_kf = forward_rate_array.copy()[start:]
            partial_kb = backward_rate_array.copy()[start:]
        else:
            partial_kf = forward_rate_array.copy()[start:stop + 1]
            partial_kb = backward_rate_array.copy()[start:stop + 1]

        partial_kb[0] = 0.
        partial_kf[-1] = 0.
        if final_state_absorbs:
            partial_kb[-1] = 0.

        diagonal1 = -(partial_kf + partial_kb)
        diagonal2 = partial_kb[1:]
        diagonal3 = partial_kf[:-1]
        rate_matrix = (np.diag(diagonal1, k=0) +
                       np.diag(diagonal2, k=1) +
                       np.diag(diagonal3, k=-1))

        return rate_matrix
