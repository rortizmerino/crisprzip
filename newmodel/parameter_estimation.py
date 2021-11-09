import numpy as np
import pandas as pd
from scipy.special import factorial
from joblib import Parallel, delayed
from codetiming import Timer

from experiment_simulations import NucleaSeq
from data_preprocessing import get_sample_aggregate_data


class TrainingSet:
    """
    A collection of measurements on which the Cas model will be
    trained.

    Attributes
    ----------
    data: pd.DataFrame
        Should contain at least the following columns:
        - experiment_name: str
        - searcher_name           # for later use
        - protospacer_seq         # for later use
        - target_seq              # for later use
        - mismatch_positions: str
        - value: float
        - error: float
    weigh_error: bool
        Determines whether entries should be weighed according to the
        relative measurement error
    weigh_multiplicity: bool
        Determines whether entries should be weighed according to the
        multiplicity of their mismatch array (True by default)

    Methods
    -------
    prepare_experiments()
        Determines simulation functions on the basis of data
    weigh_multiplicity()
        Calculates weight due to multiplicity of the mismatch array
    weigh_error()
        Calculates weight due to the relative error
    msd_lin_cost_function()
        Calculates the (linear) MSD of the model compared to the data
    msd_log_cost_function(
        Calculates the logarithmic MSD of the model compared to the
        data
    """

    def __init__(self, data: pd.DataFrame,
                 weigh_error=True, weigh_multiplicity=True):

        if not all([col in data.columns for col in
                    ['experiment_name',
                     'mismatch_positions',
                     'value',
                     'error']]):
            raise ValueError('Dataframe does not contain all columns required '
                             'for a training set.')

        self.data = data
        self.weights = self.set_weights(weigh_error, weigh_multiplicity)
        self.simulations = self.prepare_experiments()
        self.simulated_values = pd.Series(index=data.index, dtype=float)

    def run_all_simulations(self, param_vector, multiprocessing=True,
                            job_number=-1):
        """Evaluates all the prepared simulations of experiments and
         saves the results in the simulated_values attribute. Uses
         multiprocessing (all CPUs) by default."""

        if not multiprocessing:
            job_number = 1

        # the 'sim' objects from the simulations Series are the
        # functions to be evaluated, all of which take in
        # param_vector as an argument.

        simulated_values = (
            Parallel(n_jobs=job_number)
            (delayed(sim)(param_vector) for sim in self.simulations)
        )
        self.simulated_values = pd.Series(index=self.data.index,
                                          data=simulated_values)

    def prepare_experiments(self) -> pd.Series:
        """For all rows in the training set, determines the function
        that simulates the proper experiment on the proper mismatch
        array. The entries of the returned simulations Series are
        functions that accept a param_vector argument and produce a
        simulated value that should correspond to the data."""

        simulations = pd.Series(index=self.data.index, dtype=object)

        # experiments Series contains proper Experiment objects
        experiment_map = {'NucleaSeq': NucleaSeq,
                          'CHAMP': Champ}
        experiments = self.data['experiment_name'].map(experiment_map)

        # Check if some experiment types have not been recognized
        undefined_experiments = experiments.isna()
        if undefined_experiments.any():
            print(f'{undefined_experiments.sum()} undefined experiment types '
                  f'in the training set will not contribute to the cost '
                  f'function.')

            def return_nothing(_):
                return 0

            simulations[undefined_experiments] = return_nothing

        # setting up simulation functions for well-defined experiments
        for i in simulations[~undefined_experiments].index:
            simulations[i] = (
                experiments[i](self.data.loc[i, 'mismatch_positions']).simulate
            )
        return simulations

    def set_weights(self, weigh_error=True, weigh_multiplicity=True):
        weights = pd.Series(index=self.data.index, data=1)
        if weigh_error:
            weights = weights * self.weigh_error()
        if weigh_multiplicity:
            weights = weights * self.weigh_multiplicity()
        normalized_weights = weights / weights.sum()
        return normalized_weights

    def weigh_error(self) -> pd.Series:
        """Calculates weights for data based on the multiplicity of the
        mismatch array"""
        relative_error = self.data['error'] / self.data['value']

        weights = pd.Series(
           index=self.data.index,
           data=(1 / relative_error)**2
        )
        return weights

    def weigh_multiplicity(self) -> pd.Series:
        """Calculates weights for data based on the multiplicity of the
        mismatch array"""
        mismatch_array = self.data['mismatch_positions']
        array_length = mismatch_array.str.len().to_numpy()
        mismatch_number = mismatch_array.str.replace('0',
                                                     '').str.len().to_numpy()
        # Multiplicity is given by the binomial factor N! / n! (N-n)!
        multiplicity = (factorial(array_length) *
                        1 / factorial(array_length - mismatch_number) *
                        1 / factorial(mismatch_number)).astype(int)

        weights = pd.Series(
            index=self.data.index,
            data=(1 / multiplicity)
        )
        return weights

    def msd_lin_cost_function(self):
        """Calculates the cost as a weighted sum over MSD between model and
        data"""
        result = np.sum(self.weights * (self.simulated_values -
                                        self.data['value']) ** 2)
        return result

    def msd_log_cost_function(self):
        """Calculates the cost as a weighted sum over logarithmic MSD
        between model and data"""

        # First check if logarithm can safely be taken, otherwise
        # infinite penalty
        if (np.any(self.simulated_values / self.data['value'] < 1e-300) or
                np.any(self.simulated_values / self.data['value'] > 1e300)):
            print('Cannot handle simulated values or data values; infinite '
                  'penalty to cost function.')
            return float('inf')

        result = np.sum(self.weights * np.log10(self.simulated_values /
                                                self.data['value']) ** 2)
        return result


def main(mp=True):
    aggregate_data = get_sample_aggregate_data()
    aggregate_data.columns = ['mismatch_positions', 'value', 'error']
    aggregate_data['experiment_name'] = 'NucleaSeq'
    training_set = TrainingSet(aggregate_data)

    guide_length = 20
    round_no = 25
    time_list = [0., ] * round_no

    for i in range(round_no):
        print(f'ROUND {i}')
        random_param_vector = np.concatenate(
            (np.random.rand(2 * guide_length + 1) * 10,
             np.random.rand(3) * 10 - 5),
            axis=0
        )

        with Timer(text='Run time: {}') as t:
            training_set.run_all_simulations(random_param_vector,
                                             multiprocessing=mp)
            cost = training_set.msd_log_cost_function()
        time_list[i] = t.last
        print(f'Cost: {cost:.3f}')

    print(f'Total time: {np.mean(time_list[1:]):.2f} \u00B1 '
          f'{np.std(time_list[1:]):.2f} sec')


if __name__ == '__main__':
    main(mp=True)
