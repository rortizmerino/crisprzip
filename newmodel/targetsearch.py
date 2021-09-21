import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns


class HybridizationLandscape:
    """
    Characterizes the hybridization landscape of a nucleic acid guided
    searcher. Assumes a reference concentration of 1 nM.

    Attributes
    ----------
    guide_length: int
        N, length of the nucleic acid guide (in bp)
    on_target_landscape: ndarray
        Contains the hybridization energies of the intermediate R-loop
        states. In presence of a PAM state, it has length N+1.
    mismatch_penalties: ndarray
        Contains the energetic penalties associated with a mismatch
        at a particular R-loop position. Has length N.
    forward_rates: dict
        Specifies the forward rates in the model. Should contain 'k_on'
        (at reference concentration 1 nM), 'k_f' and 'k_clv' (zero when
        searcher is catalytically dead).
    pam_detection: bool
        If true, the landscape includes a PAM recognition state.
    catalytic_dead: bool
        If true, the searcher does not cleave.

    Methods
    _______
    solve_master_equation()
        Solves Master equation, giving time evolution of the landscape
        occupancy
    get_cleaved_fraction()
        Returns the fraction of cleaved targets after a specified time
        (for active searchers)
    get_effective_cleavage_rate()
        Returns the effective rate at which the searcher transitions
        from the PAM state (if present, otherwise b=1) to the cleaved
        state (for active searchers)
    get_cleavage_probability()
        Returns the probability that a searcher in the PAM state (if
        present, otherwise b=1) cleaves a target before having left it
        (for active searchers)
    get_bound_fraction()
        Returns the fraction of bound targets after a specified time
        (for dead searchers)
    plot_landscape()
        Creates a line plot of the on- or off-target landscape
    plot_penalties()
        Creates a bar plot of the mismatch penalties
    """

    def __init__(self,
                 on_target_landscape: np.ndarray,
                 mismatch_penalties: np.ndarray,
                 forward_rates: dict,
                 pam_detection=True, catalytic_dead=False):
        """Constructor method"""

        # check whether parameters are 1d arrays
        if on_target_landscape.ndim > 1 or mismatch_penalties.ndim > 1:
            raise ValueError('Landscape parameters must be 1d arrays')

        # check whether landscape dimensions agree with guide length
        guide_length = mismatch_penalties.size
        if on_target_landscape.size != pam_detection + guide_length:
            raise ValueError('Landscape dimensions do not match guide length')

        # check whether forward_rates dictionary contains proper keys
        if not ('k_on' in forward_rates and
                'k_f' in forward_rates and
                'k_clv' in forward_rates):
            raise ValueError('Forward rates dictionary should include k_on, '
                             'k_f and k_clv as keys')

        # check whether cleavage rate agrees with catalytic status
        if (forward_rates['k_clv'] == 0) != catalytic_dead:
            raise ValueError('Cleavage rate does not corresponds to '
                             'catalytic activity')

        # assign values
        self.guide_length = guide_length
        self.pam_detection = pam_detection
        self.catalytic_dead = catalytic_dead

        self.on_target_landscape = on_target_landscape
        self.mismatch_penalties = mismatch_penalties
        self.forward_rate_dict = forward_rates
        self.forward_rate_array = self.__get_forward_rate_array()

    # --- BASIC (PRIVATE) METHODS ---

    def __get_off_target_landscape(self, target_mismatches: np.ndarray):
        """Adds penalties due to mismatches to landscape"""

        # check dimensions of mismatch position array
        if target_mismatches.size != self.guide_length:
            raise ValueError('Target array should be of same length as guide')

        landscape_penalties = np.concatenate(
            (np.zeros(int(self.pam_detection)),  # add preceding zero for PAM
             np.cumsum(target_mismatches * self.mismatch_penalties))
        )
        return self.on_target_landscape + landscape_penalties

    def __get_landscape_diff(self, target_mismatches: np.ndarray):
        """Returns the difference between landscape states"""
        hybrid_landscape = self.__get_off_target_landscape(target_mismatches)
        return np.diff(hybrid_landscape, prepend=np.zeros(1))

    def __get_forward_rate_array(self):
        """Turns the forward rate dictionary into proper array"""
        forward_rate_array = np.concatenate(
            #  solution state
            (self.forward_rate_dict['k_on'] * np.ones(1),
             # PAM and intermediate R-loop states
             self.forward_rate_dict['k_f'] *
             np.ones(self.on_target_landscape.size - 1),
             # final/complete R-loop state
             self.forward_rate_dict['k_clv'] * np.ones(1),
             # cleaved state
             np.zeros(1))
        )
        return forward_rate_array

    def __get_backward_rate_array(self, target_mismatches: np.ndarray):
        """Obtains backward rates from detailed balance condition"""
        boltzmann_factors = np.exp(self.__get_landscape_diff(target_mismatches))
        backward_rate_array = np.concatenate(
            #  solution state
            (np.zeros(1),
             # PAM and R-loop states
             self.forward_rate_array[:-2] * boltzmann_factors,
             # cleaved state
             np.zeros(1))
        )
        return backward_rate_array

    def __get_rate_matrix(self, target_mismatches: np.ndarray,
                          searcher_concentration: float = 1.0) -> np.ndarray:
        """Sets up the rate matrix describing the master equation"""

        backward_rates = self.__get_backward_rate_array(target_mismatches)
        forward_rates = self.forward_rate_array

        # Taking account of non-reference concentration (in units nM)
        forward_rates[0] *= searcher_concentration

        diagonal1 = -(forward_rates + backward_rates)
        diagonal2 = backward_rates[1:]
        diagonal3 = forward_rates[:-1]
        rate_matrix = (np.diag(diagonal1, k=0) +
                       np.diag(diagonal2, k=1) +
                       np.diag(diagonal3, k=-1))

        return rate_matrix

    def solve_master_equation(self, target_mismatches: np.ndarray,
                              initial_condition: np.ndarray,
                              time: float, searcher_concentration: float = 1.0,
                              rebinding=True) -> np.ndarray:
        """
        Calculates how the occupancy of the landscape states evolves by
        evaluating the master equation. Absorbing states (solution and
        cleaved state) are explicitly incorporated.

        Parameters
        ----------
        target_mismatches: ndarray
            Vector, equally long as the guide, where entries 0 and 1
            indicate the positions of matching and mismatching bases
            on an (off-)target
        initial_condition: ndarray
            Vector showing the initial occupancy on the hybridization
            landscape. Has length guide_length+3 (if PAM_detection is
            true), and should sum to 1.
        time: float
            Time at which the master equation is evaluated.
        searcher_concentration: float
            Searcher concentration in solution (units nM). Takes the
            reference value of 1 nM by default.
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
        if initial_condition.size != (2 + self.on_target_landscape.size):
            raise ValueError('Initial condition should be of same length as'
                             'hybridization landscape')

        rate_matrix = self.__get_rate_matrix(target_mismatches,
                                             searcher_concentration)

        # if rebinding is prohibited, on-rate should be zero
        if not rebinding:
            rate_matrix[:, 0] = 0

        # where the magic happens; evaluating the master equation
        matrix_exponent = linalg.expm(rate_matrix * time)
        landscape_occupancy = matrix_exponent.dot(initial_condition)
        return landscape_occupancy

    # --- METHODS FOR ACTIVE SEARCHERS ---

    def get_cleaved_fraction(self, target_mismatches: np.ndarray, time: float,
                             searcher_concentration: float = 1.0) -> float:
        """
        Returns the fraction of cleaved targets after a specified time
        (for active searchers)

        Parameters
        ----------
        target_mismatches: ndarray
            Vector, equally long as the guide, where entries 0 and 1
            indicate the positions of matching and mismatching bases
            on an (off-)target
        time: float
            Time at which the cleaved fraction is calculated
        searcher_concentration: float
            Searcher concentration in solution (units nM). Takes the
            reference value of 1 nM by default.

        Returns
        -------
        cleaved_fraction: float
            Fraction of targets that is expected to be cleaved by time
            t.
        """

        # check if searcher is catalytically active
        if self.catalytic_dead:
            raise ValueError('Cannot obtain cleaved fraction for '
                             'catalytically dead searcher')

        unbound_state = np.concatenate(
            (np.ones(1), np.zeros(self.on_target_landscape.size + 1))
        )
        prob_distr = self.solve_master_equation(target_mismatches,
                                                unbound_state, time,
                                                searcher_concentration)
        cleaved_fraction = prob_distr[-1]
        return cleaved_fraction

    def get_effective_cleavage_rate(self, target_mismatches: np.ndarray)\
            -> float:
        """
        Returns the effective rate at which the searcher transitions
        from the PAM state (if present, otherwise b=1) to the cleaved
        state (for active searchers)

        Parameters
        ----------
        target_mismatches: ndarray
            Vector, equally long as the guide, where entries 0 and 1
            indicate the positions of matching and mismatching bases
            on an (off-)target

        Returns
        -------
        effective_cleavage_rate: float
            effective cleavage rate from PAM state (or b=1)
        """

        # check if searcher is catalytically active
        if self.catalytic_dead:
            raise ValueError('Cannot obtain cleavage probability for '
                             'catalytically dead searcher')

        backward_rates = self.__get_backward_rate_array(target_mismatches)[1:-1]
        forward_rates = self.forward_rate_array[1:-1]
        gamma = backward_rates / forward_rates
        effective_cleavage_rate = backward_rates[0] / gamma.cumprod().sum()
        return effective_cleavage_rate

    def get_cleavage_probability(self, target_mismatches: np.ndarray):
        """
        Returns the probability that a searcher in the PAM state (if
        present, otherwise b=1) cleaves a target before having left it
        (for active searchers)

        Parameters
        ----------
        target_mismatches: ndarray
            Vector, equally long as the guide, where entries 0 and 1
            indicate the positions of matching and mismatching bases
            on an (off-)target

        Returns
        -------
        cleavage_probability: float
            probability of cleavage before leaving from the PAM state
            (or b=1)
        """
        effective_cleavage_rate = \
            self.get_effective_cleavage_rate(target_mismatches)
        off_rate = self.__get_backward_rate_array(target_mismatches)[1]
        return 1 / (1 + off_rate / effective_cleavage_rate)

    # --- METHODS FOR DEAD SEARCHERS ---

    def get_prob_bound(self, target_mismatches: np.ndarray, time: float,
                       searcher_concentration: float = 1.0) -> float:
        """
        Returns the fraction of bound targets after a specified time
        (for dead searchers)

        Parameters
        ----------
        target_mismatches: ndarray
            Vector, equally long as the guide, where entries 0 and 1
            indicate the positions of matching and mismatching bases
            on an (off-)target
        time: float
            Time at which the cleaved fraction is calculated
        searcher_concentration: float
            Searcher concentration in solution (units nM). Takes the
            reference value of 1 nM by default.

        Returns
        -------
        cleaved_fraction: float
            Fraction of targets that is expected to be bound by time
            t.
        """

        # check if searcher is catalytically dead
        if not self.catalytic_dead:
            raise ValueError('Cannot obtain binding probability for '
                             'catalytically active searcher')

        unbound_state = np.concatenate(
            (np.ones(1), np.zeros(self.on_target_landscape.size + 1))
        )
        prob_distr = self.solve_master_equation(target_mismatches,
                                                unbound_state, time,
                                                searcher_concentration)
        bound_fraction = 1 - prob_distr[0]
        return bound_fraction

    # --- PLOTTING METHODS ---

    def plot_landscape(self, target_mismatches: np.ndarray = None, axes=None):
        """
        Creates a line plot of the on- or off-target landscape. If
        target_mismatches is provided, plots both the on- and
        off-target, otherwise just the on-target landscape.

        Parameters
        ----------
        target_mismatches: ndarray
            Vector, equally long as the guide, where entries 0 and 1
            indicate the positions of matching and mismatching bases
            on an (off-)target
        axes: matplotlib.Axes
            (optional) axes object to which plot can be added

        Returns
        -------
        axes: matplotlib.Axes
            axes object with landscape line plots.
        """

        # line plot definition for both landscapes
        def plot_landscape_line(landscape, color):
            axes.plot(
                np.arange(landscape.size) - 1 * self.pam_detection,
                landscape,
                color=color,
                linewidth=2,
                marker="o",
                markersize=8,
                markeredgewidth=2,
                markeredgecolor=color,
                markerfacecolor="white"
            )
            pass

        # obtaining on- and off-target landscapes, adding solution
        # states at zero energy
        on_target_landscape = np.concatenate(
            (np.zeros(1), self.on_target_landscape, np.zeros(1))
        )
        if target_mismatches is not None:
            plot_off_landscape = True
            off_target_landscape = np.concatenate(
                (np.zeros(1),
                 self.__get_off_target_landscape(target_mismatches),
                 np.zeros(1))
            )
        else:
            plot_off_landscape = False
            off_target_landscape = None

        if axes is None:
            axes = plt.subplot()

        # making landscape plots
        if plot_off_landscape:
            plot_landscape_line(on_target_landscape, 'lightgray')
            plot_landscape_line(off_target_landscape, 'firebrick')
        else:
            plot_landscape_line(on_target_landscape, 'cornflowerblue')

        # window dressing
        axes.set_xlabel(r'Targeting progression $b$', fontsize=12)
        axes.set_ylabel(r'Free energy ($k_BT$)', fontsize=12)
        axes.set_xticks(
            np.arange(on_target_landscape.size) - 1 * self.pam_detection
        )
        x_tick_labels = (
                ['S'] + self.pam_detection * ['P'] + ['1'] +
                ['' + (x % 5 == 0) * str(x) for x in
                 range(2, self.guide_length)] +
                [str(self.guide_length)] + (1 - self.catalytic_dead) * ['C']
        )
        axes.set_xticklabels(x_tick_labels, rotation=0)
        axes.tick_params(axis='both', labelsize=10)
        plt.grid('on')
        sns.set_style('ticks')
        sns.despine(ax=axes)

        return axes

    def plot_penalties(self, axes=None):
        """
        Creates a bar plot of the mismatch penalties.

        Parameters
        ----------
        axes: matplotlib.Axes
            (optional) axes object to which plot can be added

        Returns
        -------
        axes: matplotlib.Axes
            axes object with penalties bar plot.
        """

        penalties = self.mismatch_penalties
        if axes is None:
            axes = plt.subplot()

        # making bar plot
        axes.bar(np.arange(self.guide_length) + 1.5,
                 penalties,
                 color='firebrick')

        # window dressing
        axes.set_xlabel(r'Targeting progression $b$', fontsize=12)
        axes.set_ylabel(r'Mismatch penalties ($k_BT$)', fontsize=12)
        axes.set_xticks(np.arange(1, self.guide_length + 1) + 0.5)
        x_tick_labels = (
                ['1'] +
                ['' + (x % 5 == 0) * str(x) for x in
                 range(2, self.guide_length)] +
                [str(self.guide_length)]
        )
        axes.set_xticklabels(x_tick_labels, rotation=0)
        axes.tick_params(axis='both', labelsize=10)
        sns.set_style('ticks')
        sns.despine(ax=axes)

        return axes
