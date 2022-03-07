import os
from time import process_time
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.optimize import dual_annealing


def track_dual_annealing(func, x0, bounds, out_path,
                         maxfun=1e7, no_local_search=True,
                         **opt_kwargs):
    """
    Enveloping function for the dual_annealing function from the
    scipy.optimize package. Keeps track of all function evaluations
    and saves them to .csv, with the use of the CallTracker class. The
    details of the function calls are stored in a csv-file, which is
    updated every 5 minutes while the optimization is running.

    Consult scipy documentation for the full functionality of the
    function. Additional parameters can be offered as opt_kwargs.

    Parameters
    ----------
    func: callable
        Objective function, in the form f(x, *args), where x is a
        ndarray of shape (n, )
    x0: ndarray, shape (n, )
        Coordinates of a single N-D starting point
    bounds: ndarray, shape (n, 2)
        Bounds for variables. (min, max) paris for each element in x,
        defining bounds for the objective function parameter
    out_path: string
        Path to csv-file where tracking results will be stored.
    maxfun: int, optional
        Soft limit for the number of objective function calls. If the
        algorithm is in the middle of a local search, this number will
        be exceeded, the algorithm will stop just after the local
        search is done. Default value is 1e7.
    no_local_search: bool, optional
        If no_local_search is set to False, a local search strategy will
        be applied. Default value is True.

    Returns
    -------
    res: OptimizeResult
        The optimization result represented as a OptimizeResult object.
        Important attributes are: x the solution array, fun the value
        of the function at the solution, and message which describes
        the cause of the termination. See scipy documentation of
        the OptimizeResult class for a description of other attributes.

    """

    # save (almost) all dual_annealing keywords in a dict
    opt_kwargs.update({
        'x0': x0,
        'bounds': bounds,
        'maxfun': int(maxfun),
        'no_local_search': no_local_search,
    })

    # run optimization inside CallTracker context manager
    with CallTracker(eval_func=func,
                     max_evals=int(maxfun) + 100,
                     out_path=out_path,
                     opt_kwargs=opt_kwargs) as tracker:

        # running the dual annealing algorithm
        result = dual_annealing(func=tracker.evaluate_and_track,
                                **opt_kwargs)
        # processing results
        tracker.get_result(result)
        print(result.message[0])

    return result


class CallTracker:
    """
    This class keeps track of the calls of the evaluation function
    made by any optimization algorithm. It creates two files,
    'evals.pkl', which contains only the call info and is updated while
    the optimization is running, and 'log.txt', which is more
    informative and created after finishing.

    Attributes
    ----------
    eval_func: callable
        The function to keep track of
    max_evals: int
        Maximum evaluation number, used from preallocation
    out_path: str
        Location where to store the output files
    opt_kwargs: dict
        Contains all the arguments with which the optimization function
        was called. Is necessary to create log.txt-file.
    export_period: int
        Number of seconds after which the evals.csv-file should be
        updated. Default is 300 s (5 min).
    is_cas9_like: bool
        Determines how the parameters are labeled in the log file.
        Default is True.

    Methods
    -------
    evaluate_and_track()
        Like the original evaluation function, but storing the
        time and result as a class attribute (to be exported later)
    export_csv()
        Unused. Exports the details of the function calls to a csv-file.
        Contains the time at which the function was called, the
        return value, and the input parameters.
    export_csv()
        Exports the details of the function calls to a pickle file.
        Contains the time at which the function was called, the
        return value, and the input parameters.
    get_result()
        Collects the OptimizeResult object that is created by the
        optimization function. This can then be integrated in the
        log.txt-file.
    make_log()
        Creates an extensive log.txt file on the basis of the evals.pkl
        file and the optimization function input & output.
    """

    # default arguments for scipy.optimize.dual_annealing
    default_opt_kwargs = {
        'args': (),
        'maxiter': 1000,
        'minimizer_kwargs': None,
        'initial_temp': 5230.0,
        'restart_temp_ratio': 2e-05,
        'visit': 2.62,
        'accept': -5.0,
        'maxfun': 10000000.0,
        'seed': None,
        'no_local_search': False,
        'call_back': None,
        'local_search_options': None
    }

    def __init__(self, eval_func: callable, max_evals: int, out_path: str,
                 opt_kwargs: dict = None, export_period: int = 300,
                 is_cas9_like: bool = True):

        self.eval_func = eval_func

        # pre-allocating arrays saves time
        self.max_evals = max_evals
        self.eval_time = np.zeros(shape=max_evals, dtype=float)
        self.eval_vals = np.zeros(shape=max_evals, dtype=float)
        self.param_vec = None  # param_vec is initialized at first evaluation
        self.head = 0

        self.out_path = os.path.abspath(out_path)

        # optimizer input & output (to print in log)
        self.optimize_kwargs = self.default_opt_kwargs.copy()
        if opt_kwargs is not None:
            self.optimize_kwargs.update(opt_kwargs)
        self.optimize_result = None

        self.export_period = export_period
        self.is_cas9_like = is_cas9_like

        self.start_time = 0
        self.start_datetime = datetime.now()

    def __enter__(self):
        """Run as context manager to handle exceptions"""

        # evaluate the starting point (x0)
        self.evaluate_and_track(self.optimize_kwargs['x0'])

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Run as context manager to handle exceptions"""
        self.eval_time = self.eval_time[:self.head]
        self.eval_vals = self.eval_vals[:self.head]
        self.param_vec = self.param_vec[:self.head, :]

        self.export_pickle(os.path.join(self.out_path, 'evals.pkl'))
        self.make_log(os.path.join(self.out_path, 'log.txt'),
                      cas9=self.is_cas9_like)

    def evaluate_and_track(self, param_vector, *args):
        """Like the original evaluation function, but storing the
        time and result as a class attribute  (to be exported later)"""

        # get value
        result = self.eval_func(param_vector, *args)

        # set first time and initialize param_vec array
        if self.head == 0:
            self.start_time = process_time()
            self.param_vec = np.zeros(shape=(self.max_evals, param_vector.size),
                                      dtype=float)

        # check array length, update if necessary
        if self.head >= self.eval_time.size:
            self.extend_arrays()

        # update arrays
        self.eval_time[self.head] = process_time() - self.start_time
        self.eval_vals[self.head] = result
        self.param_vec[self.head, :] = param_vector

        # export if 5 minutes have passed
        if ((self.eval_time[self.head] // self.export_period) >
                (self.eval_time[self.head-1] // self.export_period)):
            self.export_pickle(os.path.join(self.out_path, 'evals.pkl'))

        # move head
        self.head += 1

        return result

    def extend_arrays(self) -> None:
        """Adds zeros to the end of all arrays, allocating memory when
        an optimization run takes longer than expected"""
        self.eval_time = np.append(self.eval_time,
                                   np.zeros_like(self.eval_time),
                                   axis=0)
        self.eval_vals = np.append(self.eval_vals,
                                   np.zeros_like(self.eval_vals),
                                   axis=0)
        self.param_vec = np.append(self.param_vec,
                                   np.zeros_like(self.param_vec),
                                   axis=0)

    def get_min_evals(self):
        """Returns an array of the lowest costs"""
        return np.minimum.accumulate(self.eval_vals)

    def export_csv(self, path):
        """Writes arrays to csv"""
        pd.DataFrame(
            data=np.concatenate((np.array([self.eval_time, self.eval_vals]).T,
                                 self.param_vec),
                                axis=1),
            columns=(['time', 'cost'] +
                     [f'p{i:02d}' for i in range(self.param_vec.shape[1])])
        ).to_csv(path)

    def export_pickle(self, path):
        """Writes arrays to pickle"""
        pd.DataFrame(
            data=np.concatenate((np.array([self.eval_time, self.eval_vals]).T,
                                 self.param_vec),
                                axis=1),
            columns=(['time', 'cost'] +
                     [f'p{i:02d}' for i in range(self.param_vec.shape[1])])
        ).to_pickle(path)

    def get_result(self, res):
        self.optimize_result = res

    def make_log(self, path, cas9=False):

        # check if optimize_kwargs are given
        if self.optimize_kwargs is None:
            raise ValueError('Cannot make log without optimization kwargs')

        # 1. stop status
        if self.optimize_result is not None:
            stop_status = self.optimize_result.message[0]
        else:
            stop_status = 'An error occurred while optimizing'

        # 2. optimization summary
        # 2.1a header (Cas9 parameters)
        if cas9:
            param_labels = '\t'.join(
                        label.rjust(8) for label in (
                            ['U_PAM'] +
                            [f'U_{i + 1}' for i in range(20)] +
                            [f'Q_{i + 1}' for i in range(20)] +
                            ['log(k_o)', 'log(k_f)', 'log(k_c)']
                        )
                    )
        else:
            param_labels = '\t'.join([f'p{i:02d}'.rjust(8) for i in
                                      range(self.param_vec.shape[1])])

        param_vector_summary = 10 * ' ' + '\t' + param_labels + '\n'

        # 2.2 values
        if self.optimize_result is not None:
            solution = self.optimize_result.x
        else:
            solution = self.param_vec[np.argmin(self.eval_vals), :]

        param_vector_summary += '\n'.join([
            # result
            '\t'.join(
                ['result'.ljust(10)] +
                ['{:>8.4f}'.format(val) for val in solution]
            ),
            # initial
            '\t'.join(
                ['initial'.ljust(10)] +
                ['{:>8.4f}'.format(val) for val in
                 self.optimize_kwargs['x0']]
            ),
            # lower bound
            '\t'.join(
                ['lower bnd'.ljust(10)] +
                ['{:>8.4f}'.format(val) for val in
                 self.optimize_kwargs['bounds'][:, 0]]
            ),
            # upper bound
            '\t'.join(
                ['upper bnd'.ljust(10)] +
                ['{:>8.4f}'.format(val) for val in
                 self.optimize_kwargs['bounds'][:, 1]]
            )
        ])

        # 3. run info
        if self.optimize_result is not None:
            stop_cost = self.optimize_result['fun']
        else:
            stop_cost = np.min(self.eval_vals)

        l_pad, r_pad = 30, 20
        start_dt = self.start_datetime
        end_dt = datetime.now()
        elapsed_dt = end_dt - start_dt

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
                ('Avg evaluation time',
                 '%.0f ms' % (1000 * self.eval_time[-1] /
                              self.eval_time.size)),
                ('Evals per hour',
                 '%.0f' % (3600 * self.eval_time.size /
                           self.eval_time[-1])),
                ('Total step number', self.eval_time.size),
                ('Final cost', '%.3e' % stop_cost),
            ]
        )

        # 4. optimization input
        l_pad, r_pad = 30, 20
        optimizer_input = '\n'.join(
            "{0:<{2}}{1:>{3}}".format(kwarg, str(self.optimize_kwargs[kwarg]),
                                      l_pad, r_pad)
            for kwarg in self.optimize_kwargs.keys()
            if (kwarg not in ['x0', 'bounds'])
        )

        # 5. optimization log
        # 5.1 header
        column_names = '\t'.join([
            'Cycle no'.rjust(10),
            'Time (s)'.rjust(10),
            'Cost'.rjust(10),
            'Cost gain'.rjust(10),
            param_labels  # for cas9 or general
        ])

        # 5.2 contents
        def write_eval_line(i):
            # get cost gain
            if i == 0:
                cost_gain = 0
            else:
                cost_gain = (self.eval_vals[i] -
                             np.minimum.accumulate(self.eval_vals)[i-1])

            # compact log: store only when improvement
            if i > 0 and cost_gain >= 0:
                return None

            newline = '\t'.join([
                # cycle number
                '{:>10d}'.format(i),
                # passed time
                '{:>10.1f}'.format(self.eval_time[i]),
                # cost
                '{:>10.3e}'.format(self.eval_vals[i]),
                # cost gain
                '{:+.2e}'.format(cost_gain).rjust(10),
                # parameter vector values
                '\t'.join(
                    ['{:>8.4f}'.format(val) for val in self.param_vec[i, :]]
                )
            ])
            return newline

        eval_lines = [write_eval_line(i) for i in range(self.eval_time.size)]
        eval_lines = [line for line in eval_lines if (line is not None)]
        log_content = '\n'.join(eval_lines)

        # 6. putting everything together
        with open(path, 'w') as log_editor:
            log_editor.write(
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
                    column_names,
                    log_content
                ])
            )
