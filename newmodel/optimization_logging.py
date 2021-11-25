import os
import traceback
from time import time
from datetime import datetime

# FIXME: circular import
from parameter_estimation import SimulatedAnnealer


class SimulatedAnnealingLogger:

    def __init__(self, optimizer: SimulatedAnnealer, log_file):
        self.optimizer = optimizer
        self.filename = log_file
        self.temp_filename = log_file[:-4] + '_temp.txt'

    def __enter__(self):
        self.original_attributes = self.optimizer.__dict__.copy()
        self.start_time = time()
        self.check_filename()
        self.temp_editor = open(self.temp_filename, 'a')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        elif optimizer.check_stop_condition():
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
                ('Started', start_dt.strftime("%a %d-%m-%Y %H:%S")),
                ('Stopped', start_dt.strftime("%a %d-%m-%Y %H:%S")),
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

        # print initial conditions
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
            ]
        )

        # print header for optimization log
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
