from typing import Union
import numpy as np
import numpy.typing as npt
from scipy import optimize

from hybridization_kinetics import Searcher


class Experiment:
    """
    Represents an experiment being carried out. Contains some standard
    methods for the more specific experiment subclasses (NucleaSeq/
    CHAMP/HitsFlip)
    """

    # TODO: make constructor accept protospacer-target content too
    def __init__(self,
                 mismatch_array: Union[str, list, npt.ArrayLike],
                 concentration: float = None,
                 pam_sensing: bool = True):
        self.mismatch_array = self.check_mismatch_array_format(mismatch_array)
        self.concentration = concentration
        self.pam_sensing = pam_sensing

    @staticmethod
    def check_mismatch_array_format(mismatch_array):
        if type(mismatch_array) is np.ndarray:
            return mismatch_array
        if type(mismatch_array) in (str, list):
            return np.array([int(nt) for nt in mismatch_array])

    def make_searcher_target_complex(self, param_vector):
        """Generates SearcherTargetComplex object on the basis of a
        parameter vector with the following entries:
          0 -    N  : on-target hybridization landscape (kBT) - length N+1
        N+1 - 2N+1  : mismatch penalties (kBT)                - length N
              2N+2  : log10( k_on  )
              2N+3  : log10( k_f   )
              2N+4  : log10( k_clv )
        """

        guide_length = int((len(param_vector) - 3 - self.pam_sensing) / 2)
        searcher = Searcher(
            on_target_landscape=param_vector[
                                0:(guide_length + self.pam_sensing)],
            mismatch_penalties=param_vector[
                               (guide_length + self.pam_sensing):-3],
            forward_rates={
                'k_on': 10 ** param_vector[-3],
                'k_f': 10 ** param_vector[-2],
                'k_clv': 10 ** param_vector[-1]
            },
            pam_detection=self.pam_sensing
        )
        searcher_target_complex = searcher.probe_target(self.mismatch_array)
        return searcher_target_complex


class NucleaSeq(Experiment):
    """
    NucleaSeq experiment, as carried out in the Finkelstein group.
    """

    def __init__(self, mismatch_array: npt.ArrayLike,
                 concentration: float = 1E6, pam_sensing: bool = True):
        Experiment.__init__(self, mismatch_array, concentration, pam_sensing)

    def simulate(self, param_vector):
        """
        Simulates the NucleaSeq experiment for a single guide-target
        combination. Outputs the effective cleavage rate as an
        observable.
        """
        # Run parameters
        time_points = np.array([12, 60, 180, 600, 1800, 6000, 18000, 60000],
                               dtype=float)

        # guarantee saturating Cas9 concentrations
        if self.concentration is None:
            self.concentration = 1E6

        # Collecting cleavage data
        st_complex = self.make_searcher_target_complex(param_vector)
        cleaved_fraction = st_complex.get_cleaved_fraction(
            time=time_points,
            searcher_concentration=self.concentration
        )

        # fit_data too close to 1 is stripped of to prevent log issues
        fit_data = 1 - cleaved_fraction
        valid_values = fit_data > 1E-9

        # Linear fit if at least 5/8 fit values are acceptable
        if np.sum(valid_values) > 4:

            def linear_func(t, k):
                return -k * t

            k_clv_fit, k_clv_var = optimize.curve_fit(
                linear_func,
                time_points[valid_values],
                np.log(fit_data[valid_values])
            )

        # (Slower) exponential fit if too few fit values are acceptable
        else:

            def exp_func(t, k):
                return 1 - np.exp(-k * t)

            k_clv_fit, _ = optimize.curve_fit(
                exp_func,
                time_points,
                cleaved_fraction,
                bounds=(0, np.inf)
            )

        return k_clv_fit[0]


# TODO: everything below will need to be updated
'''
class Champ(Experiment):
    """
    CHAMP experiment, as carried out in the Finkelstein group.
    """

    def simulate_association_constant(self, target_mismatches):
        """
        Simulates the CHAMP experiment for a single guide-target
        combination. Outputs the effective association constant as an
        observable.
        """

        # Run parameters
        c_points = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])  # units nM
        time = 600  # 10 minutes

        # Collecting binding data
        searcher_target_complex = self.searcher.probe_target(target_mismatches)
        bound_fraction = np.zeros_like(c_points)
        for i in range(len(c_points)):
            bound_fraction[i] = searcher_target_complex.get_cleaved_fraction(
                time=time,
                searcher_concentration=c_points[i],
            )

        # Performing a fit on the Hill equation
        def hill_equation(c, const): return c / (c + 1 / const)

        const_fit, const_var = optimize.curve_fit(hill_equation, c_points,
                                                  bound_fraction)
        return const_fit[0], const_var[0, 0]


class HitsFlip(Experiment):
    """
    HiTS-FLIP experiment, as carried out in the Greenleaf group.
    """

    def simulate_association_rate(self, target_mismatches):
        """
        Simulates the HiTS-FLIP experiment for a single guide-target
        combination. Outputs the effective association rate as an
        observable.
        """

        # Run parameters
        time_points = np.array([500, 1000, 1500], dtype=float)
        searcher_concentration = 10  # [dCas9] = 10 nM ?

        # Collecting binding data
        searcher_target_complex = self.searcher.probe_target(target_mismatches)
        bound_fraction = searcher_target_complex.get_bound_fraction(
            time=time_points,
            searcher_concentration=searcher_concentration
        )

        # TODO: I have doubts whether the below is a good method
        # Performing a linear fit
        def linear_func(t, k): return -k * t

        k_on_fit, k_on_var = optimize.curve_fit(linear_func, time_points,
                                                bound_fraction)
        return k_on_fit[0], k_on_var[0, 0]

    def simulate_dissociation_rate(self, target_mismatches):
        """
        Simulates the HiTS-FLIP experiment for a single guide-target
        combination. Outputs the effective dissociation rate as an
        observable.
        """

        # 1) First, let system equilibrate over dCas9 landscape
        equilibrium_time = 12 * 3600  # 12 hours
        saturating_concentration = 1E6  # saturating concentrations

        dead_searcher = self.searcher.generate_dead_clone()
        dead_complex = dead_searcher.probe_target(target_mismatches)

        # Obtaining and normalizing equilibrium distribution
        equilibrium_distr = dead_complex.solve_master_equation(
            initial_condition=np.concatenate(
                (np.ones(1),
                 np.zeros(dead_complex.on_target_landscape.size + 1))
            ),
            time=equilibrium_time,
            searcher_concentration=saturating_concentration,
            rebinding=True)
        equilibrium_distr[0] = 0
        equilibrium_distr = equilibrium_distr / equilibrium_distr.sum()

        # 2) Having obtained the equilibrium distribution, we now allow
        #    the searcher to unbind (without rebinding).

        # Run parameters
        time_points = np.array([500, 1000, 1500], dtype=float)

        # Collecting unbinding data
        unbound_fraction = dead_complex.solve_master_equation(
            initial_condition=equilibrium_distr,
            time=time_points,
            searcher_concentration=0,  # doesn't affect unbinding rate
            rebinding=False  # searcher cannot bind from solution
        )[:, 0]

        # Performing a linear fit in log space (different method than
        # reported in Eslami's 2021 manuscript)
        fit_data = np.log(1 - unbound_fraction)

        def linear_func(t, k): return -k * t

        k_off_fit, k_off_var = optimize.curve_fit(linear_func, time_points,
                                                  fit_data)
        return k_off_fit[0], k_off_var[0, 0]
'''