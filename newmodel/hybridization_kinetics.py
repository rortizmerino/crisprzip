import numpy as np
import numpy.typing as npt
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns


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
        states. In presence of a PAM state, it has length N+1.
    mismatch_penalties: array_like
        Contains the energetic penalties associated with a mismatch
        at a particular R-loop position. Has length N.
    forward_rates: dict
        Specifies the forward rates in the model. Should contain 'k_on'
        (at reference concentration 1 nM), 'k_f' and 'k_clv'.
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
                 forward_rates: dict,
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
        if on_target_landscape.size != pam_detection + guide_length:
            raise ValueError('Landscape dimensions do not match guide length')

        # check whether forward_rates dictionary contains proper keys
        if not ('k_on' in forward_rates and
                'k_f' in forward_rates and
                'k_clv' in forward_rates):
            raise ValueError('Forward rates dictionary should include k_on, '
                             'k_f and k_clv as keys')

        # assign values
        self.guide_length = guide_length
        self.pam_detection = pam_detection

        self.on_target_landscape = on_target_landscape
        self.mismatch_penalties = mismatch_penalties
        self.forward_rate_dict = forward_rates
        self.forward_rate_array = self.__get_forward_rate_array()

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

    def generate_dead_clone(self):
        """Returns Searcher object with zero catalytic activity"""
        dead_forward_rate_dict = self.forward_rate_dict.copy()
        dead_forward_rate_dict['k_clv'] = 0
        dead_searcher = Searcher(
            on_target_landscape=self.on_target_landscape,
            mismatch_penalties=self.mismatch_penalties,
            forward_rates=dead_forward_rate_dict
        )
        return dead_searcher

    def probe_target(self, target_mismatches: np.array):
        """Returns SearcherTargetComplex object"""
        return SearcherTargetComplex(self.on_target_landscape,
                                     self.mismatch_penalties,
                                     self.forward_rate_dict,
                                     target_mismatches)

    def plot_on_target_landscape(self, axes=None):
        """
        Creates a line plot of the on- or off-target landscape. If
        target_mismatches is provided, plots both the on- and
        off-target, otherwise just the on-target landscape.

        Parameters
        ----------
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

        if axes is None:
            axes = plt.subplot()

        # obtaining on-target landscape, adding solution
        # states at zero energy
        on_target_landscape = np.concatenate(
            (np.zeros(1), self.on_target_landscape, np.zeros(1))
        )

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
                [str(self.guide_length)] + ['C']
        )
        axes.set_xticklabels(x_tick_labels, rotation=0)
        axes.tick_params(axis='both', labelsize=10)
        axes.grid('on')
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
        states on an on-target. In presence of a PAM state, it has
        length N+1.
    off_target_landscape: ndarray
        Contains the hybridization energies of the intermediate R-loop
        states on the current off-target. In presence of a PAM state,
        it has length N+1.
    mismatch_penalties: ndarray
        Contains the energetic penalties associated with a mismatch
        at a particular R-loop position. Has length N.
    forward_rates: dict
        Specifies the forward rates in the model. Should contain 'k_on'
        (at reference concentration 1 nM), 'k_f' and 'k_clv'.
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
                 mismatch_penalties: np.ndarray, forward_rates: dict,
                 target_mismatches: np.ndarray):
        """Constructor method"""
        Searcher.__init__(self, on_target_landscape, mismatch_penalties,
                          forward_rates)

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
        landscape_penalties = np.concatenate(
            (np.zeros(int(self.pam_detection)),  # add preceding zero for PAM
             np.cumsum(self.target_mismatches * self.mismatch_penalties))
        )
        return self.on_target_landscape + landscape_penalties

    def __get_landscape_diff(self):
        """Returns the difference between landscape states"""
        hybrid_landscape = self.off_target_landscape
        return np.diff(hybrid_landscape, prepend=np.zeros(1))

    def __get_backward_rate_array(self):
        """Obtains backward rates from detailed balance condition"""
        boltzmann_factors = np.exp(self.__get_landscape_diff())
        backward_rate_array = np.concatenate(
            #  solution state
            (np.zeros(1),
             # PAM and R-loop states
             self.forward_rate_array[:-2] * boltzmann_factors,
             # cleaved state
             np.zeros(1))
        )
        return backward_rate_array

    def __get_rate_matrix(self, searcher_concentration: float = 1.0) \
            -> np.ndarray:
        """Sets up the rate matrix describing the master equation"""

        # shallow copy to prevent overwriting due to concentration
        forward_rates = self.forward_rate_array.copy()
        backward_rates = self.backward_rate_array

        # Taking account of non-reference concentration (in units nM)
        forward_rates[0] *= searcher_concentration

        diagonal1 = -(forward_rates + backward_rates)
        diagonal2 = backward_rates[1:]
        diagonal3 = forward_rates[:-1]
        rate_matrix = (np.diag(diagonal1, k=0) +
                       np.diag(diagonal2, k=1) +
                       np.diag(diagonal3, k=-1))

        return rate_matrix

    def solve_master_equation(self, initial_condition: np.ndarray,
                              time: npt.ArrayLike,
                              searcher_concentration: float = 1.0,
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
        rate_matrix = self.__get_rate_matrix(searcher_concentration)

        # if rebinding is prohibited, on-rate should be zero
        if not rebinding:
            rate_matrix[:, 0] = 0

        # trivial case
        if time is int(0):
            return initial_condition

        # making sure that time is a 1d ndarray
        time = np.atleast_1d(time)

        # where the magic happens; evaluating the master equation

        # 1. diagonalize M = U D U_inv
        eigenvals, eigenvecs = linalg.eig(rate_matrix)
        eigenvecs_inv = linalg.inv(eigenvecs)

        # 2. B = exp(Dt)
        b_matrix = np.exp(np.tensordot(eigenvals.real, time, axes=0))
        diag_b_matrix = (b_matrix *  # 2D b_matrix put on 3D diagonal
                         np.repeat(np.eye(b_matrix.shape[0])[:, :, np.newaxis],
                                   b_matrix.shape[1], axis=2))

        # 3. exp(Mt) = U B U_inv = U exp(Dt) U_inv
        exp_matrix = np.tensordot(eigenvecs.dot(diag_b_matrix),
                                  eigenvecs_inv,
                                  axes=((1,), (0,)))

        # 4. P(t) = exp(Mt) P0
        landscape_occupancy = exp_matrix.dot(initial_condition)

        # normalizing P(t) to correct for rounding errors
        total_occupancy = np.sum(landscape_occupancy, axis=0)
        landscape_occupancy = landscape_occupancy / total_occupancy

        # correct for failed diagonalization (>0.1% normalization error)
        sloppy_solutions = np.abs(1 - total_occupancy) > 1e-3
        for i in np.arange(len(time))[sloppy_solutions]:
            landscape_occupancy[:, i] =\
                linalg.expm(rate_matrix * time[i]).dot(initial_condition)
            error_tuple = (np.abs(1 - total_occupancy[i]),
                           np.abs(1 - np.sum(landscape_occupancy[:, i])))

            if error_tuple[0] > error_tuple[1]:
                print(f'Succesful!   Error from {error_tuple[0]:.4f} to '
                      f'{error_tuple[1]:.4f}')
            else:
                print(f'Unsuccesful! Error from {error_tuple[0]:.4f} to '
                      f'{error_tuple[1]:.4f}')

        return np.squeeze(landscape_occupancy.T)

    def get_cleavage_probability(self) -> float:
        """Returns the probability that a searcher in the PAM state (if
        present, otherwise b=1) cleaves a target before having left
        it"""

        forward_rates = self.forward_rate_array[1:-1]
        backward_rates = self.backward_rate_array[1:-1]
        gamma = backward_rates / forward_rates
        cleavage_probability = 1 / (1 + gamma.cumprod().sum())
        return cleavage_probability

    def get_cleaved_fraction(self, time: npt.ArrayLike,
                             searcher_concentration: float = 1.0) \
            -> npt.ArrayLike:
        """
        Returns the fraction of cleaved targets after a specified time

        Parameters
        ----------
        time: array_like
            Times at which the cleaved fraction is calculated
        searcher_concentration: float
            Searcher concentration in solution (units nM). Takes the
            reference value of 1 nM by default.

        Returns
        -------
        cleaved_fraction: array_like
            Fraction of targets that is expected to be cleaved by time
            t.
        """

        unbound_state = np.concatenate(
            (np.ones(1), np.zeros(self.on_target_landscape.size + 1))
        )
        prob_distr = self.solve_master_equation(unbound_state, time,
                                                searcher_concentration)
        cleaved_fraction = prob_distr.T[-1]
        return cleaved_fraction

    def get_bound_fraction(self, time: npt.ArrayLike,
                           searcher_concentration: float = 1.0) \
            -> npt.ArrayLike:
        """
        Returns the fraction of bound targets after a specified time,
        assuming that searcher is catalytically dead/inactive.

        Parameters
        ----------
        time: array_like
            Times at which the cleaved fraction is calculated
        searcher_concentration: float
            Searcher concentration in solution (units nM). Takes the
            reference value of 1 nM by default.

        Returns
        -------
        cleaved_fraction: array_like
            Fraction of targets that is expected to be bound by time
            t.
        """

        unbound_state = np.concatenate(
            (np.ones(1), np.zeros(self.on_target_landscape.size + 1))
        )
        # setting up clone SearcherTargetComplex object with zero
        # catalytic activity, k_clv=0
        dead_forward_rate_dict = self.forward_rate_dict.copy()
        dead_forward_rate_dict['k_clv'] = 0
        dead_searcher_complex = self.generate_dead_clone()

        prob_distr = \
            dead_searcher_complex.solve_master_equation(unbound_state, time,
                                                        searcher_concentration)
        bound_fraction = 1 - prob_distr.T[0]
        return bound_fraction

    def plot_off_target_landscape(self, axes=None):
        """
        Creates a line plot of the off-target landscape on top of the
        on-target landscape.

        Parameters
        ----------
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

        if axes is None:
            axes = plt.subplot()

        # obtaining on- and off-target landscapes, adding solution
        # states at zero energy
        on_target_landscape = np.concatenate(
            (np.zeros(1), self.on_target_landscape, np.zeros(1))
        )
        off_target_landscape = np.concatenate(
            (np.zeros(1), self.off_target_landscape, np.zeros(1))
        )

        # making landscape plots
        plot_landscape_line(on_target_landscape, 'lightgray')
        plot_landscape_line(off_target_landscape, 'firebrick')

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
                [str(self.guide_length)] + ['C']
        )
        axes.set_xticklabels(x_tick_labels, rotation=0)
        axes.tick_params(axis='both', labelsize=10)
        axes.grid('on')
        sns.set_style('ticks')
        sns.despine(ax=axes)

        return axes


def main():
    mm_positions = np.array(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
    p1 = Searcher(
        on_target_landscape=np.array(
            [4, 2, 5, 5, 2, 5, 1, 4, 2, 4, 6, 3, 4, 3, 3, 5, 6, 2, 2, 1, 3]),
        mismatch_penalties=np.array(
            [2, 2, 3, 3, 4, 5, 5, 4, 4, 4, 5, 4, 5, 4, 4, 2, 4, 2, 2, 3]),
        forward_rates={'k_on': .02, 'k_f': 1, 'k_clv': .1},
        pam_detection=True,
    )
    p2 = p1.probe_target(mm_positions)

    p2.get_cleaved_fraction(1)
    p2.get_cleaved_fraction(1E4)
    p2.get_cleaved_fraction(0)
    p2.get_cleaved_fraction(np.arange(1, 5))

    pass


if __name__ == '__main__':
    main()
