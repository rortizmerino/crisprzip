"""Simplify a full landscape to a coarse-grained, 3-state model."""

from typing import Tuple, Union, Dict

import numpy as np
from scipy.linalg import inv

from .kinetics import SearcherTargetComplex, SearcherSequenceComplex


def coarse_grain_landscape(
    searcher_target_complex: Union[SearcherTargetComplex,
                                   SearcherSequenceComplex],
    intermediate_range: Tuple[int, int] = (7, 14)
) -> Tuple[Dict[str, float], int]:
    """Calculate the coarse-grained rates over two barrier regions.

    A typical off-target landscape for CRISPR-Cas9 has 2 barriers,
    corresponding to the formation of the seed and the PAM-distal
    R-loop. By coarse-graining the full landscape, one can obtain
    the effective kinetics between the semi-stable states (PAM,
    intermediate, full R-loop) that can be experimentally discriminated.

    Parameters
    ----------
    searcher_target_complex : `crisprzip.kinetics.SearcherTargetComplex` or
    `crisprzip.kinetics.SearcherSequenceComplex`
        The off-target hybridization landscape from this instance is
        used to calculate the coarse-grained rates
    intermediate_range : `tuple` [`int`], optional
        The states a and b between which intermediate state is looked
        for. The lowest-energy state in the range [a, b) is the
        intermediate state. Default is [7, 14).

    Returns
    -------
    coarse_grained_rates : `dict` [`str`, `float`]
        Dictionary containing the rates k_OI, k_IC, k_IO, k_CI between
        the states Open (=PAM), Intermediate, Closed (=full R-loop).
    intermediate_id: `int`
        Location of the intermediate state.

    Notes
    -----
    Coarse-grained landscapes are obtained by first obtaining the
    intermediate state (see ``intermediate_range``). Then, for the state
    pairs open-intermediate and intermediate-closed, a partial rate
    matrix is obtained. By solving the Master Equation, we obtain
    the average arrival time between the state pairs. The inverse of
    the average arrival time gives the rate. The coarse-graining
    approach is explained in detail in [12345]_.

    Note that state labelling is different in some sources, where open and
    closed refer to the state of the R-loop, not the protein structure.
    Also, the open state "O" can be labelled as PAM state "P".

    References
    ----------
    .. [12345] Eslami-Mossallam B et al. (2022) "A kinetic model predicts SpCas9
       activity, improves off-target classification, and reveals the
       physical basis of targeting fidelity." Nature Communications.
       [10.1038/s41467-022-28994-2](https://doi.org/10.1038/s41467-022-28994-2)
    """

    if isinstance(searcher_target_complex, SearcherSequenceComplex):
        return CoarseGrainedSequenceComplex(
            searcher_target_complex.on_target_landscape,
            searcher_target_complex.mismatch_penalties,
            searcher_target_complex.internal_rates,
            searcher_target_complex.protospacer,
            searcher_target_complex.target_seq,
            searcher_target_complex.weight
        ).get_coarse_grained_rates(intermediate_range)
    else:
        return CoarseGrainedComplex(
            searcher_target_complex.on_target_landscape,
            searcher_target_complex.mismatch_penalties,
            searcher_target_complex.internal_rates,
            searcher_target_complex.target_mismatches
        ).get_coarse_grained_rates(intermediate_range)


class CoarseGrainedComplex(SearcherTargetComplex):
    """Extends the SearchTargetComplex class with coarse-graining
    functionality."""

    def get_coarse_grained_rates(
        self,
        intermediate_range: Tuple[int, int] = (7, 14)
    ) -> Tuple[Dict[str, float], int]:
        """Calculate the coarse-grained rates over two barrier regions.

        Parameters
        ----------
        intermediate_range : `tuple` [`int`], optional
            The states a and b between which intermediate state is looked
            for. The lowest-energy state in the range [a, b) is the
            intermediate state. Default is [7, 14).

        Returns
        -------
        coarse_grained_rates : `dict` [`str`, `float`]
            Dictionary containing the rates k_OI, k_IC, k_IO, k_CI between
            the states Open (=PAM), Intermediate, Closed (=full R-loop).
        intermediate_id: `int`
            Location of the intermediate state.
        """

        # Because the target landscape is defined to have length N,
        # instead of N+1, we add one to the intermediate id
        intermediate_id = 1 + (
                np.argmin(self.off_target_landscape[intermediate_range[0]:
                                                    intermediate_range[1]],
                          axis=0) +
                intermediate_range[0]
        )
        # making sure that intermediate_id is int
        intermediate_id = np.atleast_1d(intermediate_id)[0]

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
        """Calculate the effective rate from state ``start`` to state ``stop``.

        Parameters
        ----------
        forward_rate_array : `numpy.ndarray`, (N+3,)
            Forward rates of a Searcher object
        backward_rate_array: `numpy.ndarray`, (N+3,)
            Backward rates of a Searcher object
        start : `int`, optional
            Starting position, default is 0 (=solution state).
        stop : `int`, optional
            Starting position, default is None (=cleaved state).

        Returns
        -------
        eff_rate: `float`
            Effective rate from ``start`` to state ``stop``.
        """

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
        """Make a rate matrix of the states between ``start`` and ``stop``.

        Parameters
        ----------
        forward_rate_array : `numpy.ndarray`, (N+3,)
            Forward rates of a Searcher object
        backward_rate_array: `numpy.ndarray`, (N+3,)
            Backward rates of a Searcher object
        start : `int`, optional
            Starting position, default is 0 (=solution state).
        stop : `int`, optional
            Starting position, default is None (=cleaved state).
        final_state_absorbs: `bool`, optional
            If `True`, final state is absorbing. Default is `False`.

        Returns
        -------
        rate_matrix : `numpy.ndarray`, (M, M)
            If ``start`` and/or ``stop`` are provided, M is number
            of transitions between the two states. Otherwise, M = N + 3.
        """

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
    """Like CoarseGrainedComplex but works for SearcherSequenceComplexes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
