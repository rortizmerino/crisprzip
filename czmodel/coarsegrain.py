"""
Simplyfing full landscape to a coarse-grained, 3-state model.

Classes:
    CoarseGrainedComplex()

Functions:
    coarse_grain_landscape(searcher_target_complex, intermediate_range)
"""

from typing import Tuple

import numpy as np
from scipy.linalg import inv

from .kinetics import SearcherTargetComplex, SearcherSequenceComplex


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

    if isinstance(searcher_target_complex, SearcherSequenceComplex):
        print("ok")
        return CoarseGrainedComplex(
            searcher_target_complex.on_target_landscape,
            searcher_target_complex.mismatch_penalties,
            searcher_target_complex.internal_rates,
            searcher_target_complex.target_mismatches
        ).get_coarse_grained_rates(intermediate_range)
    else:
        print("ok2")
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


class CoarseGrainedSequenceComplex(SearcherSequenceComplex,
                                   CoarseGrainedComplex):
    pass
