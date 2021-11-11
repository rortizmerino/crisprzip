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

    def get_cost(self, param_vector):
        self.run_all_simulations(param_vector)
        return self.msd_log_cost_function()

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
        experiment_map = {'NucleaSeq': NucleaSeq}
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
        if (np.any(self.simulated_values / self.data['value'] < 1e-300) or
                np.any(self.simulated_values / self.data['value'] > 1e300)):
            print('Cannot handle simulated values or data values; infinite '
                  'penalty to cost function.')
            return float('inf')

        result = np.sum(self.weights * np.log10(self.simulated_values /
                                                self.data['value']) ** 2)
        return result


class SimulatedAnnealer:
    """
    WRITE DOCSTRING

    Stops when:
    - the change in the cost function drops below tolerance
    - niter steps have been taken, without success
    - final_temperature has been reached, without success

    """

    def __init__(self,
                 function: callable,
                 initial_param_vector: np.ndarray,

                 # optimization args
                 initial_temperature: float,
                 step_size: float,
                 cost_tolerance: float,

                 # optional arguments
                 parameter_bounds: tuple = None,
                 cooling_rate: float = 1.,  # cooling_rate < 1
                 check_cycle: int = 1,
                 acceptance_bounds: tuple = (0., 1.),
                 adjust_factor: float = 1.,  # adjust_factor > 1
                 final_temperature: float = 0.,
                 max_trial_no: int = float('inf')):

        # function to be optimized
        self.function = function
        self.param_vector = initial_param_vector

        # TODO: temporary timer
        with Timer(name='cost calc', logger=None):
            self.cost = function(initial_param_vector)

        if parameter_bounds is None:
            parameter_bounds = (-np.inf * np.ones(initial_param_vector.shape),
                                np.inf * np.ones(initial_param_vector.shape))
        self.parameter_bounds = parameter_bounds

        # optimization counters
        self.stop_condition = False
        self.trial_no = 0
        self.accept_no = 0
        self.avg_cost = self.cost
        self.cost_gain = 0.

        # temperature parameters
        self.initial_temperature = initial_temperature
        self.temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate

        # step taking parameters
        self.check_cycle = check_cycle
        self.max_trial_no = max_trial_no
        self.cost_tolerance = cost_tolerance
        self.param_no = self.param_vector.size
        self.step_size = step_size
        self.acceptance_bounds = acceptance_bounds  # (min, max)
        self.adjust_factor = adjust_factor

    def take_step(self) -> np.ndarray:
        """Returns a small step away from the given parameter vector"""
        # random step
        step = np.random.uniform(low=-self.step_size,
                                 high=self.step_size,
                                 size=self.param_no)
        # guarantees that trial_param_vector is within parameter_bounds
        trial_param_vector = np.clip(self.param_vector + step,
                                     *self.parameter_bounds)
        return trial_param_vector

    def apply_metropolis_criterion(self, trial_cost) -> bool:
        """Based on the Metropolis criterion, accepts (returns True) or
        rejects (returns False) a trial parameter vector."""

        acceptance_prob = np.exp(-(trial_cost - self.cost) / self.temperature)
        # Note that acceptance_prob > 1 when trial_cost < self.cost,
        # such that it will always be accepted

        return np.random.uniform() < acceptance_prob

    def do_step_cycle(self) -> None:
        """Makes number of steps given by the check_cycle attribute,
        and accepts/rejects them according to the Metropolis
        criterion."""

        self.accept_no = 0
        avg_cycle_cost = 0.

        for _ in range(self.check_cycle):
            self.trial_no += 1
            trial_param_vector = self.take_step()

            # TODO: temporary timer
            with Timer(name='cost calc', logger=None):
                trial_cost = self.function(trial_param_vector)

            # If accepted, updates param_vector and cost
            if self.apply_metropolis_criterion(trial_cost):
                self.param_vector = trial_param_vector
                self.cost = trial_cost
                self.accept_no += 1

            # Calculates the average cost of this check cycle. Used
            # to check the stop condition
            avg_cycle_cost += self.cost / self.check_cycle

        self.cost_gain = avg_cycle_cost - self.avg_cost
        self.avg_cost = avg_cycle_cost

    def equilibrate(self) -> None:
        """Repeats step cycles until the system has equilibrated, i.e.
        the acceptance ratio is within the given bounds."""

        while not self.stop_condition:
            self.do_step_cycle()

            # Checks stop condition: maximum trial number
            if self.trial_no > self.max_trial_no:
                self.stop_condition = True
                print('Maximum number of trials reached!')

            acceptance_ratio = self.accept_no / self.check_cycle
            # Too many accepted trials, increase step size
            if acceptance_ratio > self.acceptance_bounds[1]:
                self.step_size *= self.adjust_factor
                continue
            # Too few accepted trials, decrease step size
            elif acceptance_ratio < self.acceptance_bounds[0]:
                self.step_size /= self.adjust_factor
                continue
            # Equilibrated when acceptance ratio is within bounds
            else:
                break

    def check_stop_condition(self) -> None:
        """Checks whether solution has been found or final temperature
        has been reached"""

        # Second condition (temperature < 1% of initial temperature)
        # guarantees that optimization will not be stopped too early
        if (abs(self.cost_gain) < self.cost_tolerance and
                self.temperature < 0.01 * self.initial_temperature):
            self.stop_condition = True
            print('Solution found!')

        if self.temperature < self.final_temperature:
            self.stop_condition = True
            print('Final temperature reached!')

    def run(self) -> None:
        """Main function"""

        while not self.stop_condition:
            # First, equilibrate at current temperature
            self.equilibrate()

            # Then, check whether param_vec is accepted as solution or
            # final temperature has been reached
            self.check_stop_condition()

            # Finally, update the temperature for next loop
            self.temperature *= self.cooling_rate


def main():

    optimization_kwargs = {
        'check_cycle': 10,  # 1000?
        'step_size': 2.,
        'cost_tolerance': 1E-3,

        'initial_temperature': 0.1,
        'final_temperature': 0.0,
        'cooling_rate': 0.99,

        'acceptance_bounds': (0.4, 0.6),
        'adjust_factor': 1.1,
    }

    aggregate_data = get_sample_aggregate_data()
    aggregate_data.columns = ['mismatch_positions', 'value', 'error']
    aggregate_data['experiment_name'] = 'NucleaSeq'
    training_set = TrainingSet(aggregate_data)

    guide_length = 20
    param_vector_ones = np.ones(2*guide_length + 4)

    with Timer(name='overall calc', logger=None):
        SimulatedAnnealer(
            function=training_set.get_cost,
            initial_param_vector=param_vector_ones,
            parameter_bounds=(
                np.array((2*guide_length + 1) * [0] + 3 * [-4]),
                np.array((2*guide_length + 1) * [10] + 3 * [4])
            ),
            max_trial_no=500,
            **optimization_kwargs
        ).run()

    print('Overall time: %.2f' % Timer.timers['overall calc'])
    print('Cost function time: %.2f' % Timer.timers['cost calc'])
    print('Cost function fraction: %.2f' %
          Timer.timers['cost calc']/Timer.timers['overall calc'])


# TODO: get Monitor working
class Monitor:
    def __init__(self, file):
        self.file = file

    def __enter__(self):
        self.file.open('w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def report(self):
        pass


def old_main(mp=True):
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
    main()
