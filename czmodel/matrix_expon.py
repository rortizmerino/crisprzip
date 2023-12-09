import numpy as np
from numba import njit
from scipy import linalg


@njit(cache=True)
def exponentiate_fast(matrix: np.ndarray, time: np.ndarray):
    """
    Fast method to calculate exp(M*t), by diagonalizing matrix M.
    Uses to the Numba package (@njit) to compile "just-in-time",
    significantly speeding up the operations.

    Returns None if diagnolization is problematic:
    - singular rate matrix
    - rate matrix with complex eigenvals
    - overflow in the exponentiatial
    - negative terms in exp_matrix
    """

    # preallocate result matrix
    exp_matrix = np.zeros(shape=((matrix.shape[0],
                                  time.shape[0],
                                  matrix.shape[1])))

    # 1. diagonalize M = U D U_inv

    # noinspection PyBroadException
    try:
        eigenvals, eigenvecs = np.linalg.eig(matrix)
        eigenvecs_inv = np.linalg.inv(eigenvecs)
    except Exception:
        return None  # handles singular matrices
    if np.any(np.iscomplex(eigenvals)):
        return None  # skip rate matrices with complex eigenvalues

    for i in range(time.size):
        t = time[i]

        # 2. B = exp(Dt)
        exponent_matrix = eigenvals.real * t
        if np.any(exponent_matrix > 700.):
            return None  # prevents np.exp overflow
        b_matrix = np.diag(np.exp(exponent_matrix))

        # 3. exp(Mt) = U B U_inv = U exp(Dt) U_inv
        distr = eigenvecs @ b_matrix @ eigenvecs_inv

        # Previously, strong negative terms ( <-1E-3 ) were ignored to
        # fall back on the iterative method. Now, we set all negative
        # terms to zero , as that seems to be more stable than the iterative
        # method.

        if np.any(distr < 0.):
            distr = np.maximum(distr, 0)

        exp_matrix[:, i, :] = distr

    return exp_matrix


@njit(cache=True)
def update_rate_matrix(ref_rate_matrix: np.ndarray, on_rate: float) \
        -> np.ndarray:
    """Takes a reference rate matrix and updates only the on_rate.
    This function supports the compilation of the function below."""

    rate_matrix = ref_rate_matrix.copy()
    rate_matrix[0, 0] = -on_rate
    rate_matrix[1, 0] = on_rate
    return rate_matrix


@njit(cache=True)
def exponentiate_fast_var_onrate(ref_matrix: np.ndarray, time: float,
                                 on_rates: np.ndarray):
    """
    Fast method to calculate exp(M*t), by diagonalizing matrix M.
    Uses to the Numba package (@njit) to compile "just-in-time",
    significantly speeding up the operations.

    Returns None if diagnolization is problematic:
    - singular rate matrix
    - rate matrix with complex eigenvals
    - overflow in the exponentiatial
    - negative terms in exp_matrix
    """

    # preallocate result matrix
    exp_matrix = np.zeros(shape=((ref_matrix.shape[0],
                                  on_rates.shape[0],
                                  ref_matrix.shape[1])))

    for i, k_on in enumerate(on_rates):
        rate_matrix = update_rate_matrix(ref_matrix, k_on)

        partial_result = exponentiate_fast(rate_matrix, np.array([time]))
        if partial_result is None:
            return None
        else:
            exp_matrix[:, i, :] = partial_result[:, 0, :]

    return exp_matrix


def exponentiate_iterative(matrix: np.ndarray, time: np.ndarray):
    """The safer method to calculate exp(M*t), looping over the values
    in t and using the scipy function for matrix exponentiation."""

    exp_matrix = np.zeros(shape=((matrix.shape[0],
                                  time.shape[0],
                                  matrix.shape[1])))
    for i in range(time.size):
        exp_matrix[:, i, :] = linalg.expm(matrix * time[i])
    return exp_matrix


def exponentiate_iterative_var_onrate(ref_matrix: np.ndarray,
                                      time: float, on_rates: np.ndarray):
    """The safer method to calculate exp(M*t), looping over the values
    in on_rate and using the scipy function for matrix exponentiation."""
    exp_matrix = np.zeros(shape=((ref_matrix.shape[0],
                                  on_rates.shape[0],
                                  ref_matrix.shape[1])))
    for i, k_on in enumerate(on_rates):
        rate_matrix = update_rate_matrix(
            ref_matrix, k_on
        )
        exp_matrix[:, i, :] = linalg.expm(rate_matrix * time)
    return exp_matrix
