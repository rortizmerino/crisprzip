import numpy as np
import pandas as pd

from scipy.special import factorial
from scipy.optimize import basinhopping
from joblib import Parallel, delayed

from hybridization_kinetics import Searcher
from experiments import NucleaSeq
from data_preprocessing import get_sample_aggregate_data

from codetiming import Timer


def msd_lin_cost_function(simulated_values: np.ndarray,
                          data: np.ndarray,
                          weights: np.ndarray):
    """Calculates the cost as a weighted sum over MSD between model and
    data"""
    result = np.sum(weights * (simulated_values - data) ** 2)
    return result


def msd_log_cost_function(simulated_values: np.ndarray,
                          data: np.ndarray,
                          weights: np.ndarray):
    """Calculates the cost as a weighted sum over logarithmic MSD
    between model and data"""

    if (np.any(simulated_values / data < 1e-300) or
            np.any(simulated_values / data > 1e300)):
        print('oops... that\'s an infinite penalty')
        return float('inf')

    result = np.sum(weights * np.log10(simulated_values / data) ** 2)
    return result


def weigh_error_multiplicity(data_df: pd.DataFrame) -> pd.Series:
    """Produces weights for data based on the multiplicity of the
    mismatch array and their relative error"""
    mismatch_array = data_df['mismatch_positions']
    array_length = mismatch_array.str.len().to_numpy()
    mismatch_number = mismatch_array.str.replace('0', '').str.len().to_numpy()
    # Multiplicity is given by the binomial factor N! / n! (N-n)!
    multiplicity = (factorial(array_length) *
                    1 / factorial(array_length - mismatch_number) *
                    1 / factorial(mismatch_number)).astype(int)

    relative_error = data_df['agg_error'] / data_df['agg_value']

    weights = pd.Series(
        index=data_df.index,
        data=(1 / multiplicity * 1 / relative_error ** 2)
    )
    return weights


def make_searcher_from_parameters(param_vector, pam_sensing=True):
    """Generates Searcher object on the basis of a parameter vector
    with the following entries:

      0 -    N  : on-target hybridization landscape (kBT) - length N+1
    N+1 - 2N+1  : mismatch penalties (kBT)                - length N
          2N+2  : log10( k_on  )
          2N+3  : log10( k_f   )
          2N+4  : log10( k_clv )

    if pam_sensing=False, the landscape definitions have length N
    """

    guide_length = int((len(param_vector) - 3 - pam_sensing) / 2)
    searcher = Searcher(
        on_target_landscape=param_vector[0:(guide_length + pam_sensing)],
        mismatch_penalties=param_vector[(guide_length + pam_sensing):-3],
        forward_rates={
            'k_on': 10 ** param_vector[-3],
            'k_f': 10 ** param_vector[-2],
            'k_clv': 10 ** param_vector[-1]
        },
        pam_detection=pam_sensing
    )
    return searcher


def cost_wrt_nucleaseq(param_vector: np.ndarray,
                       data: pd.DataFrame,
                       log_fit=True):
    """Calculates the linear/logarithimic MSD cost of a searcher wrt
    nucleaseq data"""

    searcher = make_searcher_from_parameters(param_vector)

    def simulate_cleavage_rate(df_index):
        target_mismatches = np.array(
            list(
                data.iloc[df_index]['mismatch_positions']
            ), dtype=int
        )
        k_clv_eff, _ = NucleaSeq(searcher).simulate_cleavage_rate(
            target_mismatches)
        return k_clv_eff

    # Parallelization of targets
    job_number = 2
    # TODO: auto-assign core number on the basis of machine

    simulated_values = np.asarray(
        Parallel(n_jobs=job_number)(
            delayed(simulate_cleavage_rate)(i) for i in range(len(data.index))
        )
    )

    if log_fit:  # logarithmic MSD
        cost_func = msd_log_cost_function
    else:
        cost_func = msd_lin_cost_function

    cost = cost_func(
        simulated_values=simulated_values,
        data=data['agg_value'].to_numpy(),
        weights=weigh_error_multiplicity(data).to_numpy()
    )

    return cost


def main():
    N = 20
    aggregate_data = get_sample_aggregate_data()

    round_no = 21
    time_list = [0, ] * round_no
    for i in range(round_no):
        print(f'ROUND {i + 1}')

        random_param_vector = np.append(
            np.random.rand(2 * N + 1) * 1E1,
            np.random.rand(3) * 10 - 5
        )
        opt_func = lambda param_vector: cost_wrt_nucleaseq(
            param_vector,
            aggregate_data,
            log_fit=True
        )

        with Timer(text='Run time: {}') as t:
            _ = opt_func(random_param_vector)
        time_list[i] = t.last

    print(f'Total time: {np.mean(time_list[1:]):.2f} \u00B1 '
          f'{np.std(time_list[1:]):.2f} sec')

    # out = basinhopping(func=opt_func,
    #                   x0=initial_param_vector,
    #                   niter=1,
    #                   disp=True,
    #                   niter_success=True)
    # return out

    pass


if __name__ == '__main__':
    main()
