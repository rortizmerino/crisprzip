import os
import traceback
from time import time
from datetime import datetime

import numpy as np


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
                 adjust_factor: float = 1.,  # adjust_factor > 1,
                 initialize_temperature: bool = True,

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
        self.initialize_temperature = initialize_temperature

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
            self.logger = SimulatedAnnealingLogger(self, log_file)

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

    # FIXME: stop condition does not immediately stop opt run
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
            if self.initialize_temperature:
                self.find_start_temperature()

            while not self.stop_condition:
                # First, equilibrate at current temperature
                self.equilibrate()

                # Then, check whether param_vec is accepted as solution or
                # final temperature has been reached
                self.check_stop_condition()

                # Finally, update the temperature for next loop
                self.temperature *= self.cooling_rate


class SimulatedAnnealingLogger:
    """
    Generates txt-files containing all relevant info of an optimization run.

    Attributes
    ---------
    optimizer: SimulatedAnnealer
        the optimization process that the logger reports on
    log_file: str
        full path of the log file to be written
    """

    def __init__(self, optimizer: SimulatedAnnealer,
                 log_file: str):
        self.optimizer = optimizer
        self.filename = log_file
        self.temp_filename = log_file[:-4] + '_temp.txt'

    def __enter__(self):
        """The __enter__ and __exit__ method allow the logger to be
        used as a context manager (with ...)"""
        self.original_attributes = self.optimizer.__dict__.copy()
        self.start_time = time()
        self.check_filename()

        # By creating a temporary file and updating that, all content
        # can be copied to the bottom of the final log file.
        self.temp_editor = open(self.temp_filename, 'a')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """The __enter__ and __exit__ method allow the logger to be
        used as a context manager (with ...)"""
        self.temp_editor.close()
        self.make_logfile(exc_type, exc_val, exc_tb)
        os.remove(self.temp_filename)

    def check_filename(self):
        """Checks whether filename already exists, and if so, updates
        filename to make sure nothing gets overwritten"""
        if os.path.exists(self.filename):

            if self.filename[-6] == '_':
                path_extension = int(self.filename[-5])
            else:
                path_extension = 1
                self.filename = self.filename[:-4] + f'_{path_extension}.txt'

            while os.path.exists(self.filename):
                path_extension += 1
                self.filename = self.filename[:-6] + f'_{path_extension}.txt'

            self.temp_filename = self.filename[:-4] + '_temp.txt'

    def report_status(self):
        """Writes a new line in the temporary log file. This method is called
        after each step cycle of the simulated annealing process."""

        current_time = time() - self.start_time
        optimizer = self.optimizer

        newline = '\t'.join([
            # cycle number
            '{:>10.0f}'.format(optimizer.trial_no / optimizer.check_cycle),
            # passed time
            '{:>10.1f}'.format(current_time),
            # average cost
            '{:>10.2e}'.format(optimizer.avg_cost),
            # average cost gain
            '{:+.2e}'.format(optimizer.cost_gain).rjust(10),
            # current temperature (%)
            '{:>8.3f}'.format(100 * optimizer.temperature /
                              optimizer.initial_temperature),
            # parameter vector values
            '\t'.join(
                ['{:>8.4f}'.format(val) for val in optimizer.param_vector]
            )
        ])
        self.temp_editor.write('\n' + newline)

    def make_logfile(self, exc_type, exc_val, exc_tb):
        """Writes the final log file, consisting of three elements:
        1. results (stop status; param_vector summary; run info)
        2. input parameters (optimizer input)
        3. optimization log (copied from the temporary file)"""

        # get optimizer info
        optimizer = self.optimizer
        guide_length = int(.5 * (optimizer.param_no - 3))
        pam_sensing = optimizer.param_no - 3 - 2 * guide_length

        # read log content
        with open(self.temp_filename, 'r') as temp_reader:
            log_content = temp_reader.readlines()

        # error handling
        if exc_type is not None:
            traceback_file = self.filename[:-4] + '_tb.txt'
            stop_status = ('An error occurred: ' + '\n' +
                           str(exc_type)[8:-2] + ': ' +
                           str(exc_val) + '\n' +
                           f'Traceback printed at {traceback_file}')
            # write traceback file
            with open(traceback_file, 'w') as tb_file:
                traceback.print_exception(exc_type, exc_val, exc_tb,
                                          file=tb_file)

        # determine stop status
        elif optimizer.trial_no >= optimizer.max_trial_no:
            stop_status = 'Maximum trial number reached'
        elif optimizer.temperature < optimizer.final_temperature:
            stop_status = 'Final temperature reached'
        elif optimizer.stop_condition:
            stop_status = 'Solution found'
        else:
            stop_status = 'Unknown stop status'

        # optimization summary
        param_vector_summary = (
                10 * ' ' + '\t' +
                '\t'.join(
                    label.rjust(8) for label in (
                            ['U_PAM'] * pam_sensing +
                            [f'U_{i + 1}' for i in range(guide_length)] +
                            [f'Q_{i + 1}' for i in range(guide_length)] +
                            ['log(k_o)', 'log(k_f)', 'log(k_c)']
                    )
                ) + '\n'
        )
        param_vector_summary += '\n'.join([
            '\t'.join(
                ['result'.ljust(10)] + log_content[-1].split('\t')[5:]
            ),
            '\t'.join(
                ['initial'.ljust(10)] + log_content[1].split('\t')[5:]
            )[:-1],
            '\t'.join(
                ['lower bnd'.ljust(10)] +
                ['{:>8.4f}'.format(val)
                 for val in optimizer.parameter_bounds[0]]
            ),
            '\t'.join(
                ['upper bnd'.ljust(10)] +
                ['{:>8.4f}'.format(val)
                 for val in optimizer.parameter_bounds[1]]
            )
        ])

        # run info
        l_pad, r_pad = 30, 20
        start_dt = datetime.fromtimestamp(self.start_time)
        end_dt = datetime.now()
        elapsed_dt = end_dt - start_dt
        start_cycle_info = log_content[1 + optimizer.init_cycles].split('\t')

        run_info = '\n'.join(
            "{0:<{2}}{1:>{3}}".format(label, value, l_pad, r_pad)
            for (label, value) in [
                ('Started', start_dt.strftime("%a %d-%m-%Y %H:%M")),
                ('Stopped', end_dt.strftime("%a %d-%m-%Y %H:%M")),
                ('Time elapsed',
                 ("%02dd %02dh %02dm %02ds" %
                  (elapsed_dt.days, elapsed_dt.seconds // 3600,
                   elapsed_dt.seconds // 60 % 60, elapsed_dt.seconds % 60)
                  )
                 ),
                ('Total step number', optimizer.trial_no),
                ('Final temperature', '%.3e' % optimizer.temperature),
                ('Final cost', '%.3e' % optimizer.cost),
                ('Starting phase: cycle number', int(start_cycle_info[0])),
                ('              : temperature',
                 '%.3e' % (.01 * float(start_cycle_info[4]) *
                           optimizer.initial_temperature)
                 )
            ]
        )

        # optimization input
        l_pad, r_pad = 30, 8
        optimizer_input = '\n'.join(
            "{0:<{2}}{1:>{3}}".format(label, value, l_pad, r_pad)
            for (label, value) in [
                ('Steps per check cycle', optimizer.check_cycle),
                ('Max step number', optimizer.max_trial_no),
                ('Cost tolerance', optimizer.cost_tolerance),
                ('Initial step size', self.original_attributes['step_size']),
                ('Acceptance ratio: min', optimizer.acceptance_bounds[0]),
                ('                  max', optimizer.acceptance_bounds[1]),
                ('Adjust factor', optimizer.adjust_factor),
                ('Initial temperature', optimizer.initial_temperature),
                ('Cooling rate', optimizer.cooling_rate),
                ('Stop temperature', optimizer.final_temperature),
            ]
        )

        # header for optimization log
        column_names = '\t'.join([
            'Cycle no'.rjust(10),
            'Time (s)'.rjust(10),
            'Avg cost'.rjust(10),
            'Cost gain'.rjust(10),
            'Temp (%)'.rjust(8),
            '\t'.join(
                label.rjust(8) for label in (
                        ['U_PAM'] * pam_sensing +
                        [f'U_{i + 1}' for i in range(guide_length)] +
                        [f'Q_{i + 1}' for i in range(guide_length)] +
                        ['log(k_o)', 'log(k_f)', 'log(k_c)']
                )
            )
        ])

        # putting everything together
        with open(self.filename, 'w') as final_editor:
            final_editor.write(
                '\n'.join([

                    ' ------------------',
                    '      results      ',
                    ' ------------------',
                    stop_status + '\n',
                    param_vector_summary + '\n',
                    run_info + '\n',

                    ' ------------------',
                    '  input parameters ',
                    ' ------------------',
                    optimizer_input + '\n',

                    ' ------------------',
                    '  optimization log ',
                    ' ------------------',
                    column_names
                ])
            )
            final_editor.writelines(log_content)
