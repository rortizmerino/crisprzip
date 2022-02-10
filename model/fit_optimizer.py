import os
from time import process_time

import pandas as pd
import numpy as np
from scipy.optimize import dual_annealing


def track_dual_annealing(func, x0, bounds, out_path,
                         maxfun=1e7, local_search=False,
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
    local_search: bool, optional
        If local_search is set to False, no local search strategy will
        be applied. Note that this variable is opposite to the regular
        scipy variable no_local_search

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
        'no_local_search': ~local_search,
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
    made by any optimization algorithm.

    Attributes
    ----------
    eval_func: callable
        The function to keep track of
    max_evals: int
        Maximum evaluation number, used from preallocation
    out_path: str
        Location where to store the csv-file containing info about the
        function calls
    opt_kwargs: dict
        Contains all the arguments with which the optimization function
        was called. Becomes useful later, when this model will generate
        log.txt-files with all relevant info.
    export_period: int
        Number of seconds after which the csv-file should be updated.
        Default is 300 s (5 min).

    Methods
    -------
    evaluate_and_track()
        Like the original evaluation function, but storing the
        time and result as a class attribute (to be exported later)
    export_csv()
        Exports the details of the function calls to a csv-file.
        Contains the time at which the function was called, the
        return value, and the input parameters.
    get_result()
        Becomes useful later, when this model will generate log.txt-
        files with all relevant info.
    """

    def __init__(self, eval_func: callable, max_evals: int, out_path: str,
                 opt_kwargs: dict = None, export_period: int = 300):

        # pre-allocating arrays saves time
        self.max_evals = max_evals
        self.eval_time = np.zeros(shape=max_evals, dtype=float)
        self.eval_vals = np.zeros(shape=max_evals, dtype=float)
        self.param_vec = None  # param_vec is initialized at first evaluation
        self.head = 0

        self.start_time = 0
        self.export_period = export_period
        self.out_path = out_path

        self.eval_func = eval_func

        # optimizer input & output (to print)
        self.optimize_kwargs = opt_kwargs
        self.optimize_result = None

    def __enter__(self):
        """Run as context manager to handle exceptions"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Run as context manager to handle exceptions"""
        self.eval_time = self.eval_time[:self.head]
        self.eval_vals = self.eval_vals[:self.head]
        self.param_vec = self.param_vec[:self.head, :]

        self.export_csv(self.out_path)

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
            self.export_csv(self.out_path)

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
        ).to_csv(os.path.abspath(path))

    def get_result(self, res):
        self.optimize_result = res
