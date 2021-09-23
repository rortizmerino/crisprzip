import numpy as np
import numpy.typing as npt
from scipy import optimize
from hybridization_kinetics import Searcher, SearcherTargetComplex


class Experiment:
    def __init__(self, searcher: Searcher):
        self.searcher = searcher


class NucleaSeq(Experiment):

    def simulate_cleavage_rate(self, target_mismatches):
        # Run parameters
        time_points = np.array([0, 12, 60, 180, 600, 1800, 6000, 18000, 60000],
                               dtype=float)
        searcher_concentration = 1E6  # saturating Cas9 concentrations

        # Collecting cleavage data
        searcher_target_complex = self.searcher.probe_target(target_mismatches)
        cleaved_fraction = searcher_target_complex.get_cleaved_fraction(
            time=time_points,
            searcher_concentration=searcher_concentration
        )

        # Performing a linear fit in log space
        fit_data = np.log(1 - cleaved_fraction)
        def linear_func(t, k): return -k * t
        k_clv_fit, k_clv_var = optimize.curve_fit(linear_func, time_points,
                                                  fit_data)

        return k_clv_fit, k_clv_var


class Champ(Experiment):

    def simulate_association_constant(self, target_mismatches):
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

        # Performing a linear fit in log space
        def hill_equation(c, const): return c / (c + 1 / const)
        const_fit, const_var = optimize.curve_fit(hill_equation, c_points,
                                                  bound_fraction)
        return const_fit, const_var


class HitsFlip(Experiment):

    def simulate_association_rate(self, target_mismatches):
        # TODO: build this simulation
        pass
