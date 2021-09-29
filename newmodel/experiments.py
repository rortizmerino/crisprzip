import numpy as np
from scipy import optimize
from hybridization_kinetics import Searcher


class Experiment:
    """
    Represents an experiment being carried out on a Searcher or
    SearcherTargetComplex. Each experiment contains a simulation method
    that outputs some observable.
    """
    def __init__(self, searcher: Searcher):
        self.searcher = searcher


class NucleaSeq(Experiment):
    """
    NucleaSeq experiment, as carried out in the Finkelstein group.
    """

    def simulate_cleavage_rate(self, target_mismatches: np.array):
        """
        Simulates the NucleaSeq experiment for a single guide-target
        combination. Outputs the effective cleavage rate as an
        observable.
        """
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

        return k_clv_fit[0], k_clv_var[0, 0]


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
