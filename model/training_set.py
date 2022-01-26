import numpy as np
import pandas as pd
from scipy.special import factorial
from joblib import Parallel, delayed

from experiment_simulations import NucleaSeq, Champ


class TrainingSet:
    """
    A collection of measurements on which the Cas model will be
    trained.

    Attributes
    ----------
    data: pd.DataFrame
        Should contain at least the following columns:
        - experiment_name: str
        - mismatch_positions: str
        - value: float
        - error: float
        - searcher_name           # for later use, to read PAM
        - protospacer_seq         # for later use, to identify mm?
        - target_seq              # for later use, to identify mm?

    weigh_error: bool
        Determines whether entries should be weighed according to the
        relative measurement error
    weigh_multiplicity: bool
        Determines whether entries should be weighed according to the
        multiplicity of their mismatch array (True by default)
    experiment_weights: dict
        Relative weights of the various experiments in the dataset. By
        default, all experiments are weighed equally.

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
                 weigh_error=True, weigh_multiplicity=True,
                 experiment_weights=None):

        if not all([col in data.columns for col in
                    ['experiment_name',
                     'mismatch_positions',
                     'value',
                     'error']]):
            raise ValueError('Dataframe does not contain all columns required '
                             'for a training set.')

        self.data = data
        self.weights = self.set_weights(weigh_error, weigh_multiplicity,
                                        experiment_weights)
        self.simulations = self.prepare_experiments()
        self.simulated_values = pd.Series(index=data.index, dtype=float)

    def get_cost(self, param_vector, multiprocessing=True):
        """Returns the log cost of a particular param_vector. This
        function should be called in the SimulatedAnnealer class."""
        self.run_all_simulations(param_vector, multiprocessing)
        return self.msd_log_cost_function()

    def run_all_simulations(self, param_vector, multiprocessing=True,
                            job_number=-1):
        """Evaluates all the prepared simulations of experiments and
         saves the results in the simulated_values attribute. Uses
         multiprocessing (all CPUs) by default."""

        if not multiprocessing:
            job_number = 1

        # Below, the 'sim' objects from the simulations Series are the
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
        experiment_map = {'nucleaseq': NucleaSeq,
                          'champ': Champ}
        experiments = (self.data['experiment_name']
                       .str.lower()
                       .map(experiment_map))

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

    def set_weights(self, weigh_error=True, weigh_multiplicity=True,
                    experiment_weights=None):
        weights = pd.Series(index=self.data.index, data=1)
        if weigh_error:
            weights = weights * self.weigh_error()
        if weigh_multiplicity:
            weights = weights * self.weigh_multiplicity()

        # Check if an experiment_weights dictionary was given, otherwise
        # make one.
        if experiment_weights is None:
            experiment_keys = self.data.experiment_name.unique()
            experiment_weights = dict(zip(experiment_keys,
                                          len(experiment_keys) * [1, ]))
        else:
            experiment_keys = experiment_weights.keys()
            if not all(self.data.experiment_name.isin(experiment_keys)):
                raise ValueError('Experiment weight dictionary must contain '
                                 'all the experiments of the dataset')

        for exp in experiment_keys:
            selection = (self.data.experiment_name == exp)
            weights[selection] = (weights[selection]
                                  / weights[selection].sum()
                                  * experiment_weights[exp])

        normalized_weights = weights / weights.sum()
        return normalized_weights

    def weigh_error(self) -> pd.Series:
        """Calculates weights for data based on the multiplicity of the
        mismatch array"""
        relative_error = self.data['error'] / self.data['value']

        weights = pd.Series(
            index=self.data.index,
            data=(1 / relative_error) ** 2
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
        if (np.any(self.simulated_values == 0.) or
                np.any(self.simulated_values / self.data['value'] < 1e-300) or
                np.any(self.simulated_values / self.data['value'] > 1e300)):
            print('Cannot handle simulated values or data values; infinite '
                  'penalty to cost function.')
            return float('inf')

        result = np.sum(self.weights * np.log10(self.simulated_values /
                                                self.data['value']) ** 2)
        print('ok')
        return result
