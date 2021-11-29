import numpy as np
import pandas as pd
from codetiming import Timer

from training_set import TrainingSet
import optimization_logging  # this notation avoids circular import error


class SimulatedAnnealer:
    """
    SimulatedAnnealer looks for the global minimum of a function via
    simulated annealing. It stops when:
    - The change in the function drops below cost_tolerance
    - max_trial_number steps have been taken, without success
    - final_temperature has been reached, without success

    Attributes
    ----------
    function: callable
        The function to be optimized. Commonly, this is the get_cost()-
        method from a TrainingSet object
    initial_param_vector: np.ndarray
        Initial guess for the parameter vector

    initial_temperature: float
        Initial temperature. By default, when the optimization starts,
        the initial temperature is updated until the step acceptance ratio
        lies between its bounds.
    step_size: float
        Initial step size.
    cost_tolerance: float
        Determines when the optimization run is successful.

    final_temperature: float
        When temperature hits this value, optimization run is stopped.
        Default is 0, which is never reached.
    max_trial_no: int
        Sets the maximum number of trials to be made (not check
        cycles). Default is infinity.
    parameter_bounds: np.ndarray
        Limits the space of trial parameter vectors. Default is None,
        setting the limits to infinity, but it is highly recommended to
        provide bounds.
    cooling_rate: float
        Sets the new temperature after the system has 'equilibrated' at
        its current temperature. Should be between 0 and 1. Default is
        1, corresponding to no cooling at all.
    check_cycle: int
        Number of trials that make up a 'check cycle', after which the
        acceptance ratio is evaluated and the temperature can be
        updated (and the optimization status is reported in the log
        file). Default is 1.
    acceptance_bounds: tuple(float)
        Lower and upper bound of the acceptance ratio, determining
        when the system has 'equilibrated' for some temperature.
        Default is (0, 1), such that the step size is never updated.
    adjust_factor: float
        When the system has not equilibrated at some temperature
        (acceptance ratio falls outside of bounds), the step size is
        updated according to adjust_factor. Should be larger than 1.
        Default is 1.

    log_file: str
        Path of the log file that should be written. If None (default),
        the outcome of the optimization run will not be reported.

    Methods
    -------
    take_step()
        Returns a small step away from the given parameter vector
    apply_metropolis_criterion()
        Based on the Metropolis criterion, accepts (returns True) or
        rejects (returns False) a trial parameter vector.
    do_step_cycle()
        Makes number of steps given by the check_cycle attribute,
        and accepts/rejects them according to the Metropolis
        criterion.
    find_start_temperature()
        Performs initial step cycles and updates initial temperature
        until the acceptance ratio is within bounds.
    equilibrate()
        Repeats step cycles until the system has equilibrated, i.e.
        the acceptance ratio is within the given bounds.
    check_stop_condition()
        Checks whether solution has been found or final temperature
        has been reached
    run()
        Main function to be called, combining all the above.
    """

    def __init__(self,
                 function: callable,
                 initial_param_vector: np.ndarray,

                 # optimization args
                 initial_temperature: float,
                 step_size: float,
                 cost_tolerance: float,

                 # optional arguments
                 final_temperature: float = 0.,
                 max_trial_no: int = float('inf'),
                 parameter_bounds: tuple = None,
                 cooling_rate: float = 1.,  # cooling_rate < 1
                 check_cycle: int = 1,
                 acceptance_bounds: tuple = (0., 1.),
                 adjust_factor: float = 1.,  # adjust_factor > 1

                 # location of log file
                 log_file: str = None):
        """Constructor method"""

        # function to be optimized
        self.function = function
        self.param_vector = initial_param_vector

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
        self.init_cycles = 0

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

        # logger
        if log_file is not None:
            self.logger = optimization_logging\
                .SimulatedAnnealingLogger(self, log_file)

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

    def find_start_temperature(self) -> None:
        """Performs initial step cycles and updates initial temperature
        until the acceptance ratio is within bounds."""

        while True:
            self.do_step_cycle()
            self.logger.report_status()

            acceptance_ratio = self.accept_no / self.check_cycle
            # Too many accepted trials, decrease temperature
            if acceptance_ratio > self.acceptance_bounds[1]:
                self.temperature /= self.adjust_factor
                continue
            # Too few accepted trials, increase temperature
            elif acceptance_ratio < self.acceptance_bounds[0]:
                self.temperature *= self.adjust_factor
                continue
            # Stop initialization when acceptance ratio is within bounds
            else:
                self.init_cycles = int(self.trial_no / self.check_cycle)
                break

    def equilibrate(self) -> None:
        """Repeats step cycles until the system has equilibrated, i.e.
        the acceptance ratio is within the given bounds."""

        while not self.stop_condition:
            self.do_step_cycle()
            self.logger.report_status()

            # Checks stop condition: maximum trial number
            if self.trial_no >= self.max_trial_no:
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

        # context manager handles opening/closing of the log file
        # and writes a header upon finishing
        with self.logger:
            # initial status
            self.logger.report_status()

            # Initialize by tuning the starting temperature
            self.find_start_temperature()

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

    # aggregate_data = get_sample_aggregate_data()
    # aggregate_data.columns = ['mismatch_positions', 'value', 'error']
    # aggregate_data['experiment_name'] = 'NucleaSeq'
    aggregate_data = pd.read_csv(
        '../newdata/sample_aggregate_data.csv',
        dtype={'mismatch_positions': str,
               'value': float,
               'error': float,
               'experiment_name': str}
    ).iloc[:, 1:]
    training_set = TrainingSet(aggregate_data)

    guide_length = 20
    param_vector_ones = np.ones(2 * guide_length + 4)

    trial_no = 500

    with Timer(name='overall calc', logger=None):
        SimulatedAnnealer(
            function=training_set.get_cost,
            initial_param_vector=param_vector_ones,
            parameter_bounds=(
                np.array((2 * guide_length + 1) * [0] + 3 * [-4]),
                np.array((2 * guide_length + 1) * [10] + 3 * [4])
            ),
            log_file='SimAnnealTestReport.txt',
            max_trial_no=trial_no,
            **optimization_kwargs
        ).run()


if __name__ == '__main__':
    main()
