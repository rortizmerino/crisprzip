"""
The kinetics module is the core of the CRISPRzipper model.
It defines the basic properties of a CRISPR(-like) searcher and uses
these to simulate its R-loop hybridization dynamics.

Classes:
    Searcher
    BareSearcher(Searcher)
    GuidedSearcher(BareSearcher)
    SearcherTargetComplex(Searcher)
    SearcherSequenceComplex(GuidedSearcher, SearcherTargetComplex)
"""

import numpy as np

from .matrix_expon import *
from .nucleic_acid import *


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
                 pam_detection=True, *args, **kwargs):
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

    def get_forward_rate_array(self, k_on, dead=False):
        """Turns the forward rate dictionary into proper array"""
        if not dead:
            k_clv = self.internal_rates['k_clv'] * np.ones(1)
        else:
            k_clv = np.zeros(1)

        forward_rate_array = np.concatenate(
            #  solution state
            (k_on * np.ones(1),
             # PAM and intermediate R-loop states
             self.internal_rates['k_f'] *
             np.ones(self.on_target_landscape.size),
             # final/complete R-loop state
             k_clv,
             # cleaved state
             np.zeros(1))
        )
        return forward_rate_array

    def probe_target(self, target_mismatches: MismatchPattern) \
            -> 'SearcherTargetComplex':
        return SearcherTargetComplex(self.on_target_landscape,
                                     self.mismatch_penalties,
                                     self.internal_rates,
                                     target_mismatches)

    def calculate_solution_energy(self, k_on):
        """Given an on-rate, returns the effective free energy of the
        solution state (under the assumption of detailed balance)"""
        return np.log(k_on / self.internal_rates['k_off'])


class BareSearcher(Searcher):
    """The BareSearcher has no built-in energy contributions from
    nucleic acid; its energy parameters are only due to the nonspecific
    interactions between the protein and the target DNA. The difference
    between the Searcher and BareSearcher values of the on_target_landscape
    and mismatch_penalties attributes are due to nucleic acid interactions
    as determined in the nucleic_acid module.
    """

    @classmethod
    def from_searcher(cls, searcher, protospacer: str,
                      weight: Union[float, Tuple[float, float]] = None,
                      *args, **kwargs):
        """Subtracts the nearest-neighbour hybridization energies from
        the definition of a 'normal' Searcher. Averages over the
        penalties due to single point mutations to find the difference
        with the original mm_penalties variable.

        Parameters
        ----------
        searcher: Searcher
            Searcher object to be transformed.
        protospacer: str
            Full sequence of the protospacer/on-target: 5'-20nt-PAM-3',
            possibly preceded by the upstream sequence.
        weight: Union[float, Tuple[float, float]]
            Optional weighing of the dna opening energy and rna duplex energy.
            If None (default), no weighin is applied. If float, both dna and
            rna energies are multiplied by the weight parameter. If tuple
            of two floats, the first value is used as a multiplier for the
            DNA opening energy, and the second is used as a multiplier for the
            RNA-DNA hybridization energy.
        """

        ontarget_na_energies = get_hybridization_energy(
            protospacer=protospacer,
            weight=weight
        )
        average_mm_penalties = find_average_mm_penalties(
            protospacer=protospacer,
            weight=weight
        )

        return cls(
            on_target_landscape=(searcher.on_target_landscape -
                                 ontarget_na_energies[1:]),
            mismatch_penalties=(searcher.mismatch_penalties -
                                average_mm_penalties),
            internal_rates=searcher.internal_rates,
            pam_detection=searcher.pam_detection,
            protospacer=protospacer,  # for the GuidedSearcher constructor
            *args, **kwargs
        )

    def to_searcher(self, protospacer: str,
                    weight: Union[float,
                                  Tuple[float, float]] = None) -> Searcher:
        """Adds the nearest-neighbour hybridization energies to
        BareSearcher energy parameters to retrieve a 'normal' Searcher
        object with effective landscape and (average) mismatch penalties.

        Parameters
        ----------
        protospacer: str
            Full sequence of the protospacer/on-target: 5'-20nt-PAM-3',
            possibly preceded by the upstream sequence.
        weight: Union[float, Tuple[float, float]]
            Optional weighing of the dna opening energy and rna duplex energy.
            If None (default), no weighin is applied. If float, both dna and
            rna energies are multiplied by the weight parameter. If tuple
            of two floats, the first value is used as a multiplier for the
            DNA opening energy, and the second is used as a multiplier for the
            RNA-DNA hybridization energy.
        """

        ontarget_na_energies = get_hybridization_energy(
            protospacer=protospacer,
            weight=weight
        )
        average_mm_penalties = find_average_mm_penalties(
            protospacer=protospacer,
            weight=weight
        )

        return Searcher(
            self.on_target_landscape + ontarget_na_energies[1:],
            self.mismatch_penalties + average_mm_penalties,
            self.internal_rates,
            self.pam_detection
        )

    def bind_guide_rna(self, protospacer: str,
                       weight: Union[float, Tuple[float, float]] = None) \
            -> 'GuidedSearcher':
        """Create a GuidedSearcher object by associating a BareSearcher
        instance with a protospacer, defining the gRNA it is carrying."""
        return GuidedSearcher(
            on_target_landscape=self.on_target_landscape,
            mismatch_penalties=self.mismatch_penalties,
            internal_rates=self.internal_rates,
            pam_detection=self.pam_detection,
            protospacer=protospacer,
            weight=weight,
        )

    def probe_target(self, target_mismatches: MismatchPattern) \
            -> 'SearcherTargetComplex':
        """Disabled, see probe_sequence()."""
        raise ValueError("A Bare/GuidedSearcher object cannot probe a "
                         "target without a defined sequence. Use "
                         "probe_sequence() instead.")

    def probe_sequence(self, protospacer: str, target_seq: str,
                       weight: Union[float, Tuple[float, float]] = None) \
            -> 'SearcherSequenceComplex':
        return SearcherSequenceComplex(self.on_target_landscape,
                                       self.mismatch_penalties,
                                       self.internal_rates,
                                       protospacer=protospacer,
                                       target_seq=target_seq,
                                       weight=weight)


class GuidedSearcher(BareSearcher):
    """Similar to the BareSearcher but with a predefined protospacer/
    gRNA. This way, you only have to provide target sequences to
    simulate binding at different DNA sites."""

    def __init__(self, on_target_landscape: np.ndarray,
                 mismatch_penalties: np.ndarray, internal_rates: dict,
                 protospacer: str,
                 weight: Union[float, Tuple[float, float]] = None,
                 *args, **kwargs):
        super().__init__(on_target_landscape=on_target_landscape,
                         mismatch_penalties=mismatch_penalties,
                         internal_rates=internal_rates,
                         *args, **kwargs)
        self.protospacer = None
        self.guide_rna = None
        self.weight = weight
        self.set_on_target(protospacer)

    def to_searcher(self, *args, **kwargs) -> Searcher:
        return super().to_searcher(self.protospacer, self.weight)

    def to_bare_searcher(self) -> BareSearcher:
        return BareSearcher(
            on_target_landscape=self.on_target_landscape,
            mismatch_penalties=self.mismatch_penalties,
            internal_rates=self.internal_rates
        )

    def set_on_target(self, protospacer: str) -> None:
        self.protospacer = protospacer
        self.guide_rna = protospacer[-23:-3].replace("T", "U")

    def probe_sequence(self, target_seq: str,
                       weight: Union[float, Tuple[float, float]] = None,
                       *args, **kwargs) -> 'SearcherSequenceComplex':
        if weight is None:
            weight = self.weight
        return super().probe_sequence(self.protospacer, target_seq,
                                      weight=weight)


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
                 target_mismatches: MismatchPattern, *args, **kwargs):
        Searcher.__init__(self, on_target_landscape, mismatch_penalties,
                          internal_rates, *args, **kwargs)

        # check dimensions of mismatch position array
        if target_mismatches.length != self.guide_length:
            raise ValueError('Target array should be of same length as guide')
        else:
            self.target_mismatches = target_mismatches

        self.off_target_landscape = self._get_off_target_landscape()
        self.backward_rate_array = self._get_backward_rate_array()

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

    def get_rate_matrix(self, on_rate: float, dead=False) -> np.ndarray:
        """Sets up the rate matrix describing the master equation"""

        # shallow copy to prevent overwriting due to concentration
        forward_rates = self.get_forward_rate_array(k_on=on_rate, dead=dead)
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
                              dead=False, rebinding=True, mode='fast') ->\
            np.ndarray:

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
        dead: bool
            If true, cleavage rate is set to zero to simulate the
            catalytically inactive dCas9 variant.
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
            rate_matrix = self.get_rate_matrix(on_rate, dead=dead)

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
            ref_rate_matrix = self.get_rate_matrix(0., dead=dead)

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

    def get_cleaved_fraction(self, time: Union[float, np.ndarray],
                             on_rate: float) -> np.ndarray:
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

    def get_bound_fraction(self, time: Union[float, np.ndarray],
                           on_rate: Union[float, np.ndarray]) -> np.ndarray:
        """
        Returns the fraction of bound targets after a specified time,
        assuming that searcher is catalytically dead/inactive.

        Parameters
        ----------
        time: Union[float, np.ndarray]
            Time(s) at which the cleaved fraction is calculated
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
        prob_distr = self.solve_master_equation(unbound_state, time,
                                                on_rate, dead=True)
        bound_fraction = 1 - prob_distr.T[0]
        return bound_fraction


class SearcherSequenceComplex(GuidedSearcher, SearcherTargetComplex):
    """The SearcherSequenceComplex is like a SearcherTargetComplex,
    but with sequence-specific nucleic acid contributions. The most
    important difference is how it calculates its "off-target landscape",
    the sum of all energetic contributions: protein landscape, protein
    penalties, DNA opening energy and RNA-DNA hybridization energy.
    """

    def __init__(self, on_target_landscape: np.ndarray,
                 mismatch_penalties: np.ndarray, internal_rates: dict,
                 protospacer: str, target_seq: str,
                 weight: Union[float, Tuple[float, float]] = None):

        self.protospacer = protospacer
        self.target_seq = target_seq
        self.hybrid = GuideTargetHybrid.from_cas9_offtarget(
            protospacer=protospacer,
            offtarget_seq=self.target_seq
        )
        target_mismatches = self.hybrid.get_mismatch_pattern()

        super().__init__(
            on_target_landscape=on_target_landscape,
            mismatch_penalties=mismatch_penalties,
            internal_rates=internal_rates,
            target_mismatches=target_mismatches,
            protospacer=protospacer,
            weight=weight
        )

        # check dimensions of mismatch position array
        if target_mismatches.length != self.guide_length:
            raise ValueError('Target array should be of same length as guide')
        else:
            self.target_mismatches = target_mismatches

        self.off_target_landscape = self._get_off_target_landscape(
            weight=self.weight
        )
        self.backward_rate_array = self._get_backward_rate_array()

    def _get_off_target_landscape(self, weight=None):
        internal_na_energy = get_hybridization_energy(
            protospacer=self.protospacer,
            offtarget_seq=self.target_seq,
            weight=weight
        )[1:]
        protein_na_energy = (
            SearcherTargetComplex._get_off_target_landscape(self)
        )
        return protein_na_energy + internal_na_energy
