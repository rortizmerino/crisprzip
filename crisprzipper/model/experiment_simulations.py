"""
This module simulates the experiments to produce model data that can
be compared to the experimental data.

Classes:
    ExperimentSimulation(ABC)
    NucleaseqSimulation(ExperimentSimulation)
    ChampSimulation(ExperimentSimulation)
"""

import warnings
from abc import ABC, abstractmethod

from scipy import optimize

from crisprzipper.model.data import ExperimentType


class ExperimentSimulation(ABC):
    """
    Represents an experiment being carried out. Contains some standard
    methods for the more specific experiment subclasses (NucleaSeq/
    CHAMP/HitsFlip). This is an Abstract Base Class, so should not be
    instantiated.

    Attributes
    ----------
    searcher: Searcher
        The searcher whose dynamics will be simulated
    k_bind: float
        The rate at which the searcher binds in a particular
        experimental context.
    """

    experiment = None

    def __init__(self, searcher: Searcher, k_bind: float):
        self.searcher = searcher
        self.k_bind = k_bind

    @abstractmethod
    def __call__(self, mm_pattern: MismatchPattern):
        pass


class NucleaseqSimulation(ExperimentSimulation):
    """
    NucleaSeq experiment, as carried out in the Finkelstein group.
    """

    experiment = ExperimentType.NUCLEASEQ

    def __call__(self, mm_pattern: MismatchPattern, try_lin_ls=False):
        """Simulation of the NucleaSeq experiment"""

        # Experimental parameters
        time_points = np.array([12, 60, 180, 600, 1800, 6000, 18000, 60000],
                               dtype=float)  # seconds
        concentration = 100.  # nM

        # Collecting cleavage data
        st_complex = self.searcher.probe_target(mm_pattern)
        cleaved_fraction = st_complex.get_cleaved_fraction(
            time=time_points,
            on_rate=self.k_bind * concentration
        )

        # Prevents issues with NaN values, replace with 0
        cleaved_fraction = np.nan_to_num(cleaved_fraction)

        # fit_data too close to 1 is stripped of to prevent log issues
        fit_data = 1 - cleaved_fraction
        valid_values = fit_data > 1E-9

        # Linear fit if at least 5/8 fit values are acceptable
        if try_lin_ls and np.sum(valid_values) > 4:
            optim_res = optimize.lsq_linear(
                -np.array([time_points[valid_values]]).T,
                np.log(fit_data[valid_values]))
            k_clv_fit = optim_res.x

        # (Slower) exponential fit if too few fit values are acceptable
        else:

            def exp_func(t, k):
                return 1 - np.exp(-k * t)

            try:
                k_clv_fit, _ = optimize.curve_fit(
                    exp_func,
                    time_points,
                    cleaved_fraction,
                    bounds=(0, np.inf)
                )
            except RuntimeError:
                k_clv_fit = [0.]

        return k_clv_fit[0]


class ChampSimulation(ExperimentSimulation):
    """
    CHAMP experiment, as carried out in the Finkelstein group.
    """

    experiment = ExperimentType.CHAMP

    def __call__(self, mm_pattern: MismatchPattern):
        """Simulation of the Champ experiment"""

        # Run parameters
        c_points = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])  # nM
        time = 600  # seconds, = 10 minutes

        # Collecting binding data
        st_complex = self.searcher.probe_target(mm_pattern)
        bound_fraction = np.zeros_like(c_points)
        for i in range(len(c_points)):
            bound_fraction[i] = st_complex.get_bound_fraction(
                time=time,
                on_rate=self.k_bind * c_points[i],
            )

        # Prevents issues with NaN values, replace with 0
        bound_fraction = np.nan_to_num(bound_fraction)

        # If no binding takes place , K_A is zero
        # This is the default for only-nan bound fraction variables
        if np.all(bound_fraction == 0.):
            return 0.

        # Performing a fit on the Hill equation
        # I tried a linearized version, but it gives worse results
        def hill_equation(c, const):
            return c / (c + 1 / const)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", optimize.OptimizeWarning)
            try:
                const_fit, const_var = optimize.curve_fit(
                    hill_equation,
                    c_points,
                    bound_fraction
                )
            except RuntimeError:
                const_fit = [0.]

        return const_fit[0]
