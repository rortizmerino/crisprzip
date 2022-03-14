import numpy as np
import pandas as pd
from scipy.special import factorial
from joblib import Parallel, delayed

from model.experiment_simulations import NucleaSeq, Champ


def read_dataset(file):
    """This function helps to properly import the data files. It handles
    the index column and makes sure that the mismatch_array is
    recognized as a string, not an integer."""
    dataset = pd.read_csv(file, index_col=0, dtype={'mismatch_array': str})
    return dataset


class TrainingSet:
    """
    A collection of measurements on which the Cas model will be
    trained.

    Attributes
    ----------
    data: pd.DataFrame
        A dataframe containing all the measurements that together make
        up this training set. Columns are:
        - mismatch_array
        - mismatch_number
        - value
        - error
        - experiment_name
        - weight
    simulated_values: pd.Series
        Simulated values of the experiments described in the data
        attribute. Gets updated each time that the method
        run_all_simulations() is called.
    simulations: pd.Series
        A Series object that references to the proper functions to call
        in order to simulate the experiments. The philosphy behind
        having all these functions put together in a Series is that
        it might save time by not having to set up the functions over
        and over again.

    Methods
    -------
    get_cost()
        Calculates the log cost of the training set. Allows for
        multiprocessing. This function should be called during
        optimization.
    run_all_simulations()
        Simulates all the experiments that are specified in the dataset.
        Allows for multiprocessing.
    prepare_experiments()
        Determines simulation functions on the basis of data
    get_weights()
        Calculates the proper weights for each dataset
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

    experiment_map = {'nucleaseq': NucleaSeq,
                      'champ': Champ}

    def __init__(self, datasets: list, experiment_names: list,
                 experiment_weights=None,
                 weigh_error=True, rel_error=True,
                 weigh_multiplicity=True,
                 normalize_weights=True):
        """
        Constructor method

        Parameters
        ----------
        datasets: list [pd.DataFrame]
            All the datasets to train the model on. These can best be
            imported with the read_dataset function of this module.
            Datasets must contain at least the following columns:
            - mismatch_array: a string of zeros and ones describing the
                              mismatch positions
            - value: the experiment outcome
            - error: the (absolute) error in measurement
        experiment_names: list [str]
            The names of the experiments. The length must agree with the
            length of the datasets list. Have a look at the class variable
            experiment_map to see which experiment names are associated to
            a simulation.
        experiment_weights: list [float]
            The weights of the experiments. The length must agree with the
            length of the datasets list. Default is a list of ones,
            indicating that each dataset contributes equally to the cost
            function.
        weigh_error: bool
            Determines whether entries should be weighed according to the
            relative measurement error (True by default)
        rel_error: bool
            Determines whether the error weight is taken on the basis
            of relative (True) or absolute (False) errors (True by
            default).
        weigh_multiplicity: bool
            Determines whether entries should be weighed according to the
            multiplicity of their mismatch array (True by default)
        normalize_weights: bool
            Determines whether the weights within each dataset should
            be normalized to 1. After having applied the experiment
            weights, all weights are normalized again.
        """

        # check if dimensions are ok
        dataset_no = len(datasets)
        if len(experiment_names) != dataset_no:
            raise ValueError('Length of data and experiment_names lists do'
                             'not agree')

        # check if experiment names are familiar
        if not all([exp.lower() in self.experiment_map.keys()
                    for exp in experiment_names]):
            raise ValueError('Not all experiment names are known')

        # create experiment_weights if not provided. Default is equal
        # weights for all datasets (corrected for size)
        if experiment_weights is None:
            experiment_weights = [1 / df.shape[0] for df in datasets]

        # create data field, combining all datasets
        self.data = pd.DataFrame()
        for i in range(dataset_no):
            dataframe = datasets[i]

            # check columns
            if not all([col in dataframe.columns for col in
                        ['mismatch_array', 'value', 'error']]):
                raise ValueError('Dataframe does not contain all columns '
                                 'required for a training set.')
            # add experiment name, find weights
            dataframe['experiment_name'] = experiment_names[i]
            dataframe['weight'] = (experiment_weights[i] *
                                   self.get_weights(dataframe, weigh_error,
                                                    rel_error,
                                                    weigh_multiplicity,
                                                    normalize_weights))

            self.data = self.data.append(dataframe)
        self.data.reset_index(drop=True, inplace=True)

        # normalize weights
        self.data['weight'] = self.data['weight'] / self.data['weight'].sum()

        self.simulations = self.prepare_experiments()
        self.simulated_values = pd.Series(index=self.data.index, dtype=float)

    def get_cost(self, param_vector, multiprocessing=True, log_msd=True):
        """Returns the log cost of a particular param_vector. This
        function should be called in the SimulatedAnnealer class."""
        self.run_all_simulations(param_vector, multiprocessing)
        if log_msd:
            return self.msd_log_cost_function()
        else:
            return self.msd_lin_cost_function()

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
        experiments = (self.data['experiment_name']
                       .str.lower()
                       .map(self.experiment_map))

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
                experiments[i](self.data.loc[i, 'mismatch_array']).simulate
            )
        return simulations

    @classmethod
    def get_weights(cls, dataset: pd.DataFrame, weigh_error=True,
                    rel_error=True, weigh_multiplicity=True,
                    normalize=True) -> pd.Series:
        """Calculates weight on the basis of error and multiplicity"""
        weights = pd.Series(data=1, index=dataset.index)
        if weigh_error:
            weights = weights * cls.weigh_error(dataset, rel_error)
        if weigh_multiplicity:
            weights = weights * cls.weigh_multiplicity(dataset)
        if normalize:
            weights = weights / weights.sum()
        return weights

    @staticmethod
    def weigh_error(dataset: pd.DataFrame, relative=True) -> pd.Series:
        """Calculates weights for data based on the multiplicity of the
        mismatch array"""

        weights = pd.Series(
            index=dataset.index,
            data=(1 / dataset['error']) ** 2
        )

        if relative:
            weights = weights * (dataset['value'] ** 2)

        return weights

    @staticmethod
    def weigh_multiplicity(dataset: pd.DataFrame) -> pd.Series:
        """Calculates weights for data based on the multiplicity of the
        mismatch array"""
        mismatch_array = dataset['mismatch_array']
        array_length = mismatch_array.str.len().to_numpy()
        mismatch_number = mismatch_array.str.replace('0',
                                                     '').str.len().to_numpy()
        # Multiplicity is given by the binomial factor N! / n! (N-n)!
        multiplicity = (factorial(array_length) *
                        1 / factorial(array_length - mismatch_number) *
                        1 / factorial(mismatch_number)).astype(int)

        weights = pd.Series(
            index=dataset.index,
            data=(1 / multiplicity)
        )
        return weights

    def msd_lin_cost_function(self):
        """Calculates the cost as a weighted sum over MSD between model and
        data"""
        result = np.sum(self.data['weight'] *
                        (self.simulated_values - self.data['value']) ** 2)
        return result

    def msd_log_cost_function(self):
        """Calculates the cost as a weighted sum over logarithmic MSD
        between model and data"""

        # First check if logarithm can safely be taken, otherwise
        # infinite penalty
        if (np.any(self.simulated_values == 0.) or
                np.any(self.simulated_values / self.data['value'] < 1e-300) or
                np.any(self.simulated_values / self.data['value'] > 1e300)):
            return float('inf')

        result = np.sum(self.data['weight'] *
                        np.log10(self.simulated_values /
                                 self.data['value']) ** 2)
        return result
