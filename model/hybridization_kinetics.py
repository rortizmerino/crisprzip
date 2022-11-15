import numpy as np
import numpy.typing as npt
from scipy import linalg

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import seaborn as sns

import model.aggregate_landscapes


class Searcher:
    """
    Characterizes the hybridization landscape of a nucleic acid guided
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
    _______
    probe_target(target_mismatches)
        Returns a SearcherTargetComplex object
    plot_landscape()
        Creates a line plot of the on- or off-target landscape
    plot_penalties()
        Creates a bar plot of the mismatch penalties
    """

    def __init__(self,
                 on_target_landscape: npt.ArrayLike,
                 mismatch_penalties: npt.ArrayLike,
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

    @classmethod
    def from_param_vector(cls, param_vector,
                          guide_length=20, pam_sensing=True):
        """
        Generates Searcher object on the basis of a parameter vector
        with the following entries:

        0 -  N-1  : on-target hybridization landscape [kBT] - length N
                    (does not include PAM energy)
        N - 2N-1  : mismatch penalties [kBT]                - length N
            2N    : log10( k_off [Hz] )
            2N+1  : log10( k_f [Hz] )
            2N+2  : log10( k_clv [Hz] )

        The remaining entries in the parameter vector (>2N+2) are
        ignored; these should correspond to the context-dependent
        parameters.

        If the searcher is PAM-insensitive, the hybridization landscape
        has length N-1, making the total parameter vector smaller by
        a length of 1.
        """

        # the index between landscape & mismatch penalties
        separator = guide_length + pam_sensing - 1

        return cls(
            on_target_landscape=param_vector[:separator],
            mismatch_penalties=param_vector[separator:separator+guide_length],
            internal_rates={
                'k_off': 10 ** param_vector[separator+guide_length],
                'k_f': 10 ** param_vector[separator+guide_length+1],
                'k_clv': 10 ** param_vector[separator+guide_length+2]
            },
            pam_detection=pam_sensing
        )

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

    def probe_target(self, target_mismatches: np.array):
        """Returns SearcherTargetComplex object"""
        return SearcherTargetComplex(self.on_target_landscape,
                                     self.mismatch_penalties,
                                     self.internal_rates,
                                     target_mismatches)

    def calculate_solution_energy(self, k_on):
        """Given an on-rate, returns the effective free energy of the
        solution state (under the assumption of detailed balance)"""
        return np.log(k_on/self.internal_rates['k_off'])

    def plot_on_target_landscape(self, y_lims=None, color='cornflowerblue',
                                 axs=None, on_rates: list = None,
                                 **plot_kwargs):
        """Creates on-target landscape plot"""
        axs = SearcherPlotter(self).plot_on_target_landscape(y_lims=y_lims,
                                                             color=color,
                                                             axs=axs,
                                                             on_rates=on_rates,
                                                             **plot_kwargs)
        return axs

    def plot_penalties(self, y_lims=None, color='firebrick', axs=None,
                       **plot_kwargs):
        """Creates mismatch penalties landscape plot"""
        axs = SearcherPlotter(self).plot_mismatch_penalties(y_lims=y_lims,
                                                            color=color,
                                                            axs=axs,
                                                            **plot_kwargs)
        return axs

    def plot_internal_rates(self, y_lims=None, color='cornflowerblue',
                            axs=None, extra_rates: dict = None, **plot_kwargs):
        """Creates forward rates plot"""
        axs = SearcherPlotter(self).plot_internal_rates(
            y_lims=y_lims,
            color=color,
            axs=axs,
            extra_rates=extra_rates,
            **plot_kwargs
        )
        return axs


class SearcherTargetComplex(Searcher):
    """
    Characterizes the hybridization landscape of a nucleic acid guided
    searcher on a particular (off-)target sequence. Assumes a reference
    concentration of 1 nM.

    Attributes
    ----------
    guide_length: int
        N, length of the nucleic acid guide (in bp)
    target_mismatches: ndarray
        Positions of mismatches in the guide-target hybrid: has length
        N, with entries 0 (matches) and 1 (mismatches).
    on_target_landscape: ndarray
        Contains the hybridization energies of the intermediate R-loop
        states on an on-target, relative to the PAM energy. In presence
        of a PAM state, it has length N (otherwise N-1).
    off_target_landscape: ndarray
        Contains the hybridization energies of the intermediate R-loop
        states on the current off-target.  In presence
        of a PAM state, it has length N (otherwise N-1).
    mismatch_penalties: ndarray
        Contains the energetic penalties associated with a mismatch
        at a particular R-loop position. Has length N.
    internal_rates: dict
        Specifies the context-independent rates in the model. Should
        contain 'k_off', 'k_f' and 'k_clv'.
    pam_detection: bool
        If true, the landscape includes a PAM recognition state. True
        by default.

    Methods
    _______
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
                 target_mismatches: np.ndarray):
        Searcher.__init__(self, on_target_landscape, mismatch_penalties,
                          internal_rates)

        # check dimensions of mismatch position array
        if target_mismatches.size != self.guide_length:
            raise ValueError('Target array should be of same length as guide')
        else:
            self.target_mismatches = target_mismatches

        self.off_target_landscape = self.__get_off_target_landscape()
        self.backward_rate_array = self.__get_backward_rate_array()

    def generate_dead_clone(self):
        """Returns SearcherTargetComplex object with zero catalytic
        activity"""
        dead_searcher = Searcher.generate_dead_clone(self)
        dead_complex = dead_searcher.probe_target(self.target_mismatches)
        return dead_complex

    def __get_off_target_landscape(self):
        """Adds penalties due to mismatches to landscape"""
        landscape_penalties = np.cumsum(self.target_mismatches *
                                        self.mismatch_penalties)
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

    def __get_landscape_diff(self):
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
                    self.target_mismatches[0] * self.mismatch_penalties[0]
                ]),
                self.off_target_landscape
            ))
        return np.diff(hybrid_landscape, prepend=np.zeros(1))

    def __get_backward_rate_array(self):
        """Obtains backward rates from detailed balance condition"""
        boltzmann_factors = np.exp(self.__get_landscape_diff())
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
                              time: npt.ArrayLike,
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
        exp_matrix = exponentiate_fast(rate_matrix, time)
        if exp_matrix is None:  # this is a safe alternative for e
            exp_matrix = exponentiate_iterative(rate_matrix, time)

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

    def get_cleaved_fraction(self, time: npt.ArrayLike,
                             on_rate: float = 1E-3) -> npt.ArrayLike:
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

    def get_bound_fraction(self, time: npt.ArrayLike,
                           on_rate: float = 1E-3) -> npt.ArrayLike:
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

    def get_all_aggregate_rates(self, intermediate_range):
        aggr_rates, intermediate_id = (
            model.aggregate_landscapes
            .get_all_aggregate_rates(self, intermediate_range)
        )
        return aggr_rates

    def plot_off_target_landscape(self, y_lims=None, color='firebrick',
                                  axs=None, **plot_kwargs):
        """Creates off-target landscape plot"""
        axs = SearcherPlotter(self).plot_off_target_landscape(
            self.target_mismatches,
            y_lims=y_lims, color=color, axs=axs, **plot_kwargs
        )
        return axs


def exponentiate_fast(matrix: np.ndarray, time: np.ndarray):
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


def exponentiate_iterative(matrix: np.ndarray, time: np.ndarray):
    """The safer method to calculate exp(M*t), looping over the values
    in t and using the scipy function for matrix exponentiation."""

    exp_matrix = np.zeros(shape=((matrix.shape[0],
                                  time.shape[0],
                                  matrix.shape[1])))
    for i in range(time.size):
        exp_matrix[:, i, :] = linalg.expm(matrix*time[i])
    return exp_matrix


class SearcherPlotter:
    """Base class that underlies plotting functionality in the Searcher
    and SearcherTargetComplex classes of this module, and the animation
    functionality of the LogAnalyzer and DashboardVideo classes of the
    analysis module. A user should not directly interact with this
    class.

    This class is built up to support 'blitting': redrawing only the
    parts of a figure that are updated. This significantly speeds up
    the video making process in the analysis module. It contains 4
    types of methods:
    1) prepare ... canvas  creates or adapts Figure and Axes objects
                           such that data can later be added
    2) prepare ... lines   adds properly styled lines to the canvas
                           that do not yet contain data (the term line
                           here includes dots or other shapes)
    3) update ...          adds data to the pre-configured line
    4) plot ...            combines the above 3 methods to make a
                           complete plot
    """

    # Some general style definitions
    title_style = {'fontweight': 'bold'}
    label_style = {'fontsize': 10}
    tick_style = {'labelsize': 10}
    line_style = {'linewidth': 2,
                  'marker': '.',
                  'markersize': 12}

    def __init__(self, searcher: Searcher):
        self.searcher = searcher

    def prepare_landscape_canvas(self, y_lims: tuple = None, title: str = '',
                                 axs: Axes = None) -> Axes:
        """Creates or adapts Figure and Axes objects such
        that an on-/off-target/penalties landscape line can later be
        added."""
        searcher = self.searcher
        if axs is None:
            _, axs = plt.subplots(1, 1, figsize=(4, 3))

        # x axis
        axs.set_xlim(-(searcher.pam_detection + 1.2),
                     searcher.guide_length + 2.2)
        axs.set_xticks(np.arange(-searcher.pam_detection,
                                 searcher.guide_length + 2))
        x_tick_labels = (
                ['S'] + searcher.pam_detection * ['P'] + ['1'] +
                ['' + (x % 5 == 0) * str(x) for x in
                 range(2, searcher.guide_length)] +
                [str(searcher.guide_length)] + ['C']
        )
        axs.set_xticklabels(x_tick_labels, rotation=0)
        axs.set_xlabel(r'Targeting progression $b$', **self.label_style)

        # y axis
        if y_lims is None:
            y_lims = (min(searcher.on_target_landscape.min(), 0),
                      searcher.on_target_landscape.max())
        axs.set_ylim(y_lims[0] - .5, y_lims[1] + .5)
        axs.set_ylabel(r'Free energy ($k_BT$)', **self.label_style)

        # title
        axs.set_title(title, **self.title_style)

        # style
        axs.tick_params(axis='both', **self.tick_style)
        axs.grid(True)
        sns.set_style('ticks')
        sns.despine(ax=axs)
        return axs

    def prepare_rates_canvas(self, y_lims: tuple = None,
                             title: str = 'Forward rates',
                             axs: Axes = None,
                             extra_rates: dict = None) -> Axes:
        """Creates or adapts Figure and Axes objects such
        that rates points can later be added."""
        searcher = self.searcher
        if axs is None:
            _, axs = plt.subplots(1, 1, figsize=(3, 3))

        if extra_rates is not None:
            extra_labels = list(extra_rates.keys())
            extra_values = list(extra_rates.values())
        else:
            extra_labels = []
            extra_values = []

        # x axis
        axs.set_xlim(-1, 3 + len(extra_labels))
        axs.set_xlabel(' ')
        axs.set_xticks(np.arange(3 + len(extra_labels)))
        x_tick_labels = (extra_labels +
                         [r'${k_{off}}$', r'${k_{f}}$', r'${k_{clv}}$'])
        axs.set_xticklabels(x_tick_labels, rotation=0)

        # y axis
        axs.set_yscale('log')
        if y_lims is None:
            all_rates = list(searcher.internal_rates.values()) + extra_values
            y_lims = (min(all_rates), max(all_rates))
        axs.set_ylim(y_lims[0] * 10 ** -.5, y_lims[1] * 10 ** .5)
        axs.set_ylabel(r'Rate (${s^{-1}})$', **self.label_style)

        # title
        axs.set_title(title, **self.title_style)

        # background
        axs.tick_params(axis='both', **self.tick_style)
        sns.set_style('ticks')
        sns.despine(ax=axs)
        return axs

    def prepare_landscape_line(self, axs: Axes, color='cornflowerblue',
                               **plot_kwargs) -> Line2D:
        """Adds styled lines to the landscape canvas"""
        line, = axs.plot([], [], color=color,
                         **self.line_style, **plot_kwargs)
        return line

    def prepare_rates_line(self, axs: Axes, color='cornflowerblue',
                           **plot_kwargs) -> Line2D:
        """"Adds styled lines (points) to the rates canvas"""
        plot_kwargs['linestyle'] = ''
        line = self.prepare_landscape_line(axs, color=color, **plot_kwargs)
        return line

    def update_on_target_landscape(self, line: Line2D) -> None:
        """Updates landscape line to represent on-target landscape"""
        searcher = self.searcher
        line.set_data(
            np.arange(1 - searcher.pam_detection, searcher.guide_length + 2),
            np.concatenate(
                (np.zeros(1),  # pam / 1st state
                 searcher.on_target_landscape,
                 np.ones(1) * line.axes.get_ylim()[0])
            )
        )

    def update_solution_energies(self, lines: list, on_rates: list) -> None:
        """Updates the free energy level of the solution state(s)"""
        searcher = self.searcher
        for line, on_rate in zip(lines, on_rates):
            line.set_data(
                np.arange(-searcher.pam_detection, 1),
                np.array([searcher.calculate_solution_energy(on_rate), 0])
            )

    def update_mismatches(self, line: Line2D) -> None:
        """Updates landscape line to represent mismatches"""
        searcher = self.searcher
        line.set_data(
            np.arange(1, searcher.guide_length + 1),
            searcher.mismatch_penalties
        )

    def update_rates(self, line: Line2D, extra_rates: dict = None) -> None:
        """Updates rate points to represent forward rates"""
        searcher = self.searcher

        if extra_rates is not None:
            extra_values = list(extra_rates.values())
        else:
            extra_values = []

        forward_rates = extra_values + list(searcher.internal_rates.values())
        line.set_data(
            list(range(3+len(extra_values))),
            forward_rates
        )

    def plot_on_target_landscape(self, y_lims: tuple = None,
                                 color='cornflowerblue', axs: Axes = None,
                                 on_rates: list = None, **plot_kwargs) -> Axes:
        """Creates complete on-target landscape plot"""

        if y_lims is None and on_rates is not None:
            searcher = self.searcher
            solution_energies = [searcher.calculate_solution_energy(k_on)
                                 for k_on in on_rates]
            y_lims = (min(list(searcher.on_target_landscape) +
                          solution_energies + [0]),
                      max(list(searcher.on_target_landscape) +
                          solution_energies))

        axs = self.prepare_landscape_canvas(
            y_lims,
            title='On-target landscape',
            axs=axs
        )
        line = self.prepare_landscape_line(
            axs,
            color=color,
            **plot_kwargs
        )
        self.update_on_target_landscape(line)

        if on_rates is not None:
            solution_lines = []
            for _ in on_rates:
                solution_lines += [self.prepare_landscape_line(
                    axs, color=color, linestyle='dashed', **plot_kwargs
                )]
            self.update_solution_energies(solution_lines, on_rates)

        return axs

    def plot_off_target_landscape(self, mismatch_positions: np.ndarray,
                                  y_lims: tuple = None,
                                  color='firebrick', axs: Axes = None,
                                  on_rates: list = None,
                                  **plot_kwargs) -> Axes:
        """Creates complete off-target landscape plot, based on the
        mismatch positions array"""

        searcher = self.searcher.probe_target(mismatch_positions)
        if y_lims is None and on_rates is not None:
            if on_rates is None:
                solution_energies = []
            else:
                solution_energies = [searcher.calculate_solution_energy(k_on)
                                     for k_on in on_rates]
            y_lims = (min(list(searcher.on_target_landscape) +
                          solution_energies + [0]),
                      max(list(searcher.off_target_landscape) +
                          solution_energies))

        axs = self.prepare_landscape_canvas(
            y_lims,
            title='Off-target landscape',
            axs=axs
        )
        lines = [
            self.prepare_landscape_line(axs, color='darkgray', **plot_kwargs),
            self.prepare_landscape_line(axs, color=color, **plot_kwargs)
        ]
        self.update_on_target_landscape(lines[0])
        lines[1].set_data(
            np.arange(1 - searcher.pam_detection, searcher.guide_length + 2),
            np.concatenate(
                (np.zeros(1),  # solution state
                 searcher.off_target_landscape,
                 np.ones(1) * lines[1].axes.get_ylim()[0])  # cleaved state
            )
        )

        if on_rates is not None:
            solution_lines = []
            for _ in on_rates:
                solution_lines += [self.prepare_landscape_line(
                    axs, color=color, linestyle='dashed', **plot_kwargs
                )]
            self.update_solution_energies(solution_lines, on_rates)

        return axs

    def plot_mismatch_penalties(self, y_lims: tuple = None,
                                color='firebrick', axs: Axes = None,
                                **plot_kwargs) -> Axes:
        """Creates complete mismatch landscape plot"""
        axs = self.prepare_landscape_canvas(
            y_lims,
            title='Mismatch penalties',
            axs=axs
        )
        line = self.prepare_landscape_line(axs, color=color, **plot_kwargs)
        self.update_mismatches(line)
        return axs

    def plot_internal_rates(self, y_lims: tuple = None,
                            color='cornflowerblue', axs: Axes = None,
                            extra_rates: dict = None, **plot_kwargs) -> Axes:
        """Creates complete forward rates plot"""
        axs = self.prepare_rates_canvas(
            y_lims,
            title='Forward rates',
            axs=axs,
            extra_rates=extra_rates
        )
        line = self.prepare_rates_line(axs, color=color, **plot_kwargs)
        self.update_rates(line, extra_rates=extra_rates)
        return axs
