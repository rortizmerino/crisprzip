from scipy.linalg import inv

from hybridization_kinetics import *


def setup_partial_rate_matrix(forward_rate_array: np.ndarray,
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
        partial_kf = forward_rate_array.copy()[start:stop+1]
        partial_kb = backward_rate_array.copy()[start:stop+1]

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


def calculate_effective_rate(forward_rate_array, backward_rate_array,
                             start=0, stop=None):
    """Calculates the effective rate from state start to state stop.
    This calculation is from the old supplementary materials of
    Eslami et al."""

    partial_k = setup_partial_rate_matrix(forward_rate_array,
                                          backward_rate_array,
                                          start=start, stop=stop,
                                          final_state_absorbs=True)

    initial_state = np.zeros(partial_k.shape[0] - 1)
    initial_state[0] = 1.

    # Eslami et al.
    eff_rate = np.matmul(-inv(partial_k[:-1, :-1]),
                         initial_state).sum() ** -1

    return eff_rate


def get_all_aggregate_rates(searcher_target_complex,
                            intermediate_range=(7, 14)):
    """
    Calculates the aggregate rates between the open (=PAM),
    intermediate, and closed (=bound) state. Assumes PAM sensing.

    Parameters
    ----------
    searcher_target_complex: SearcherTargetComplex
        The searcher-target complex for which the aggregate rates are
        calculated
    intermediate_range: tuple
        The states a and b between which intermediate state is looked
        for. The lowest-energy state in the range [a, b) is the
        intermediate state. Default is [7, 14).

    Returns
    _______
    aggr_rates: dict
        Dictionary containging the rates k_OI, k_IC, k_IO, k_CI
        (Open, Intermediate, Closed). Confusingly, these have later
        been renamed (O -> P for PAM and C -> O).
    intermediate_id: int
        Location of the intermediate state.

    """

    intermediate_id = (
            np.argmin(searcher_target_complex
                      .off_target_landscape[intermediate_range[0]:
                                            intermediate_range[1]]) +
            intermediate_range[0]
    )

    kf = searcher_target_complex.forward_rate_array
    kb = searcher_target_complex.backward_rate_array

    # From open (=PAM) to intermediate
    k_oi = calculate_effective_rate(kf, kb,
                                    start=1,
                                    stop=1+intermediate_id)

    # From intermediate to closed (=bound)
    k_ic = calculate_effective_rate(kf, kb,
                                    start=1+intermediate_id,
                                    stop=len(kf) - 2)

    # From intermediate to closed (=bound)
    k_io = calculate_effective_rate(kb[::-1], kf[::-1],
                                    start=(len(kf)-1) - (1+intermediate_id),
                                    stop=(len(kf)-1) - 1)

    # From intermediate to closed (=bound)
    k_ci = calculate_effective_rate(kb[::-1], kf[::-1],
                                    start=1,
                                    stop=(len(kf)-1) - (1+intermediate_id))

    aggr_rates = {'k_OI': k_oi, 'k_IC': k_ic, 'k_IO': k_io, 'k_CI': k_ci}
    return aggr_rates, intermediate_id
