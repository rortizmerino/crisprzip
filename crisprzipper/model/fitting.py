"""
This module takes care of the optimization of the parameter vector to
a training set. It is based on the dual_annealing algorithm from the
scipy.optimization package. There are several objects to keep track
of the optimization algorithm and save the process and the metadata.

Classes:
    SearcherOptimizer
    CallTracker
    OptRunLog
    EvaluationLog
    SearcherScorer

"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from traceback import format_exception
from typing import List

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pathlib import Path
from scipy.optimize import dual_annealing, OptimizeResult

from crisprzipper.model.data import AggregateData
from crisprzipper.model.hybridization_kinetics import Searcher
from crisprzipper.model.tools import path_handling


class SearcherOptimizer:
    """Optimizes the parameter vector such that it best fits the
    training data

    Attributes
    ----------
    pvec_class: str
        The ParameterVector daughter class to be trained
    training_set: List[AggregateData]
        The aggregate datasets to which the parameter vector will be fit
    exp_types: List[ExperimentType]
        Indicates the type of data from the training set
    eval_log: EvaluationLog
        Keeps track of all evaluations during the optimization process
    opt_run_log: OptRunLog
        Keeps track of all metadata of the optimization process
    opt_kwargs: dict
        Parameters that are related to the general optimization process.
        Default values are gives as class attributes.
    score_kwargs: dict
        Parameters that are related to the general optimization process,
        which are passed on to the cost_function() method. Default
        values are gives as class attributes.
    anneal_kwargs: dict
        Parameters that are related to the general optimization process,
        which are passed on to scipy's dual_annealing(). Default
        values are gives as class attributes.

    Methods
    -------
    bin(out_path)
        Carries out the optimization process, based on dual_annealing.
        Saves its results and metadata in the folder that out_path
        points to, and returns the OptimizeResult from scipy.optimize.
    set_opt_kwargs()
        Changes the general optimization parameters.
    set_score_kwargs()
        Changes the scoring keyword arguments.
    set_anneal_kwargs()
        Changes the dual_annealing keyword arguments.
    cost_function(x)
        Loops over all datasets to score parameter array x.
    """

    # general optimization parameters
    opt_kwargs = {
        # sets how often evals.pkl gets updated (in seconds)
        "export_period": 600,

        # relative weights of the datasets in training_set
        "dataset_weights": None
    }

    # passed on to SearcherScorer.compare()
    score_kwargs = {
        "multiprocessing": True,
        "job_number": -1,
        "log": True,
        "weigh_errors": True,
        "weigh_multiplicity": True,
    }

    # passed on to scipy.optimize.dual_annealing()
    anneal_kwargs = {
        "maxiter": 2500,
        "initial_temp": 300.0,
        "visit": 2.5,
        "accept": -5.0,
        "no_local_search": True,
        "seed": None
    }

    def __init__(self, pvec_type: str, training_set: List[AggregateData]):
        """Constructor method.

        Parameters
        ----------
        pvec_type: str
            The name of the ParameterVector daughter class to be trained.
            For instance, 'FreeBindingFixedCleavageParams'.
        training_set: list[AggregateData]
            The aggregate datasets to which the parameter vector will
            be fit.
        """

        self.pvec_class = getattr(crisprzipper.model.parameter_vector, pvec_type)

        self.training_set = training_set
        self.exp_types = [dataset.exp_type for dataset in training_set]

        self.eval_log = None
        self.opt_run_log = None

        # dataset weights are taken from score_kwargs when bin starts
        self.__dataset_weights = None

    def set_opt_kwargs(self, **new_opt_kwargs):
        self.opt_kwargs.update(new_opt_kwargs)

    def set_score_kwargs(self, **new_score_kwargs):
        self.score_kwargs.update(new_score_kwargs)

    def set_anneal_kwargs(self, **new_anneal_kwargs):
        self.anneal_kwargs.update(new_anneal_kwargs)

    @path_handling
    def run(self, out_path: Union[Path, str]) -> OptimizeResult:
        """This method uses the dual_annealing function from
        scipy.optimize to find a new parameter vector that fits the
        specified training set.
        Most functionality is broken down into other (private) methods,
        which are documented in a bit more detail below.

        Parameters
        ----------
        out_path: Path or str
            Location of the folder where the results should be saved

        Returns
        -------
        result: OptimizeResult
            Standard object from scipy.optimize, quite similar to a
            dict. Among other things, it includes the key 'x', which
            contains the solution of the optimization algorithm
            (in np.ndarray format).
        """

        # NOTE
        # The use of dual_annealing might only be a temporary solution.
        # It is easy to work with, but has its shortcomings and cannot
        # be controlled or tracked without breaking open the code.
        # Moreover, it only serves the goal of finding solutions 'from
        # scratch'; if one would want to finetune a specific solution or
        # a large collection of solutions, other optimization algorithms
        # might be far more suitable (local optimization, or genetic
        # algorithms, for instance). Such functionality probably require
        # different methods than this one, and might also be better off
        # in a class of their own. This is something to be taken up in
        # the future.

        if not out_path.exists():
            os.mkdir(out_path)
        self.__prepare_dataset_weights()
        self.__make_new_evaluation_log(out_path)
        self.__make_new_optrun_log(out_path)
        result = self.__run_annealing(out_path)

        # TODO: make a summary evals log in .txt-format, for quick import

        return result

    def __set_random_seed(self) -> None:
        if self.anneal_kwargs["seed"] is None:
            rng = np.random.default_rng()
            random_seed = rng.integers(low=0, high=2**32 - 1)
            # noinspection PyTypedDict
            self.anneal_kwargs["seed"] = random_seed

    def __prepare_dataset_weights(self) -> None:
        """Takes dataset_weights from opt_kwargs dict and sets it as
        a (private) attribute"""
        self.__dataset_weights = self.opt_kwargs["dataset_weights"]
        if self.__dataset_weights is None:
            self.__dataset_weights = len(self.training_set) * [1]
        if len(self.__dataset_weights) != len(self.training_set):
            raise ValueError("The number of dataset weights does not "
                             "agree with the number of datasets")

    def __make_new_evaluation_log(self, out_path) -> None:
        """Reads off the parameter names from the ParamaterVector and
        uses it to prepare an EvaluationLog object"""
        self.eval_log = EvaluationLog(
            out_path.joinpath('evals.pkl'),
            pd.DataFrame(
                index=[0],
                columns=(['time', 'cost'] + self.pvec_class.param_names)
            )
        )
        self.eval_log.to_pickle()

    def __make_new_optrun_log(self, out_path):
        """Takes all the bin settings/metadata before starting the
        optimization and saves it in a OptRunLog object, which is
        exported to a JSON-file."""
        self.opt_run_log = OptRunLog(
            pvec_type=self.pvec_class.__name__,
            pvec_info=self.pvec_class.get_info(),
            evals=self.eval_log,
            datasets=self.training_set,
            annealing_kwargs=self.anneal_kwargs,
            score_kwargs=self.score_kwargs,
            start_dt=datetime.now(),
            stop_dt=datetime.now()
        )
        self.opt_run_log.to_json(out_path.joinpath("params.json"))

    def __run_annealing(self, out_path) -> OptimizeResult:
        """Runs dual_annealing, but inside a CallTracker object, such
        that we keep track of what is going on during the optimization.
        If an error occurs, everything is saved and the traceback is
        added to the OptRunLog object."""
        try:
            total_eval_no = 1 + (2 * len(self.pvec_class.x0) *
                                 self.anneal_kwargs['maxiter'])

            # Thanks to this context manager, results will be saved
            # whenever an error occurs
            with CallTracker(self.cost_function, total_eval_no, self.eval_log,
                             self.opt_kwargs['export_period']) as tracker:

                x0 = np.array(self.pvec_class.x0)
                bounds = np.stack([self.pvec_class.lb, self.pvec_class.ub],
                                  axis=1)

                opt_result = dual_annealing(func=tracker.evaluate_and_track,
                                            x0=x0, bounds=bounds,
                                            **self.anneal_kwargs)
                self.opt_run_log.result = opt_result

                self.opt_run_log.stop_dt = datetime.now()
                self.opt_run_log.to_json(out_path.joinpath('params.json'))

            return opt_result

        # The except clause will save the error message to the OptRunLog
        except BaseException as exc:
            self.opt_run_log.exceptions = format_exception(*sys.exc_info())
            self.opt_run_log.stop_dt = datetime.now()
            self.opt_run_log.to_json(out_path.joinpath('params.json'))
            raise exc

    def cost_function(self, x: np.ndarray) -> float:
        """Loops over all datasets to score parameter vector x."""

        pvec = self.pvec_class(x)
        scorer = SearcherScorer(pvec.to_searcher())

        cost = 0
        for i, etype in enumerate(self.exp_types):
            score = scorer.compare(self.training_set[i],
                                   pvec.to_binding_rate(etype),
                                   **self.score_kwargs)
            cost += self.__dataset_weights[i] * score
        return cost


class CallTracker:
    """This class keeps track of the calls of the evaluation function
    made by any optimization algorithm. At regular times and after
    finishing, the call info is sent to an EvaluationLog object, which
    gets exported, to keep you up-to-date as the optimization is
    running.

    Attributes
    ----------
    costfunc: callable
        The function to keep track of
    max_evals: int
        Maximum evaluation number, used from preallocation
    evals: EvaluationLog
        Object that contains the evaluation information.
    export_period: int
        Number of seconds after which the evals.csv-file should be
        updated. Default is 600 s (10 min).

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

    def __init__(self, costfunc: callable, max_evals: int,
                 evals: 'EvaluationLog', export_period=600):

        self.costfunc = costfunc
        self.max_evals = max_evals
        self.evals = evals
        self.export_period = export_period

        # pre-allocating arrays saves time
        self.__eval_time = np.zeros(shape=max_evals, dtype=float)
        self.__eval_vals = np.zeros(shape=max_evals, dtype=float)
        self.__param_vec = np.zeros(shape=(max_evals, 1), dtype=float)
        self.__head = 0
        self.__start_dt = 0

    def __enter__(self):
        """Run as context manager to save results whenever bin stops"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Run as context manager to save results whenever bin stops"""
        self.__eval_time = self.__eval_time[:self.__head]
        self.__eval_vals = self.__eval_vals[:self.__head]
        self.__param_vec = self.__param_vec[:self.__head, :]
        self.export_log()

    def evaluate_and_track(self, param_vector):
        """Like the original evaluation function, but storing the
        time and result as a class attribute  (to be exported later)"""

        # get value
        result = self.costfunc(param_vector)

        # set first time and initialize param_vec array
        if self.__head == 0:
            self.__start_dt = datetime.now()
            self.__param_vec = np.zeros(shape=(self.max_evals,
                                               param_vector.size),
                                        dtype=float)

        # check array length, update if necessary
        if self.__head >= self.__eval_time.size:
            self.extend_arrays()

        # update arrays
        self.__eval_time[self.__head] = ((datetime.now() - self.__start_dt)
                                         .total_seconds())
        self.__eval_vals[self.__head] = result
        self.__param_vec[self.__head, :] = param_vector

        # export if export_period has passed
        if ((self.__eval_time[self.__head] // self.export_period) >
                (self.__eval_time[self.__head - 1] // self.export_period)):
            self.export_log()

        # move head
        self.__head += 1

        return result

    def extend_arrays(self) -> None:
        """Adds zeros to the end of all arrays, allocating memory when
        an optimization bin takes longer than expected"""
        self.__eval_time = np.append(self.__eval_time,
                                     np.zeros_like(self.__eval_time),
                                     axis=0)
        self.__eval_vals = np.append(self.__eval_vals,
                                     np.zeros_like(self.__eval_vals),
                                     axis=0)
        self.__param_vec = np.append(self.__param_vec,
                                     np.zeros_like(self.__param_vec),
                                     axis=0)

    def export_log(self) -> None:
        """Sends data to associated EvaluationLog object, and directs
        it to export to pickle file."""

        if self.__head == 0:
            pass

        self.evals.update(
            self.__eval_time[:self.__head + 1],
            self.__eval_vals[:self.__head + 1],
            self.__param_vec[:self.__head + 1, :]
        )
        self.evals.to_pickle()


@dataclass
class OptRunLog:
    """The optimization bin log is a simple placeholder for all the
    data associated with an optimization bin. It is being updated before
    and after the bin (by the Searcher Optimizer), and can be used to
    export and import optimization results and metadata.

    Attributes
    ----------
    pvec_type: str
        The name of the ParameterVector daughter object
    pvec_info: dict
        Contains the names, initial conditions and boundaries of the
        parameters
    evals: EvaluationLog
        The associated object that tracks all costfunc evaluations
    datasets: List[AggregateData]
        The training set that is being trained on
    annealing_kwargs: dict
        Keyword arguments for scipy.optimize's dual_annealing
    score_kwargs: dict
        Keyword arguments for the SearcherScorer
    start_dt: datetime
        Datetime at which optimization started
    stop_dt: datetime
        Datetime at which optimization stopped
    result: OptimizeResult
        Results of the optimization bin
    exceptions: list
        List of the exceptions occuring during optimization (if any).

    """

    pvec_type: str = None
    pvec_info: dict = None
    evals: 'EvaluationLog' = None
    datasets: list = None
    annealing_kwargs: dict = None
    score_kwargs: dict = None
    start_dt: datetime = None
    stop_dt: datetime = None
    result: OptimizeResult = None
    exceptions: list = None

    default_dual_annealing_kwargs = {
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

    def __post_init__(self):
        """Runs after the default dataclass __init__ method. Includes
        the default dual_annealing parameters for the undefined
        kwargs"""
        new_anneal_kwargs = self.annealing_kwargs.copy()
        self.annealing_kwargs = self.default_dual_annealing_kwargs.copy()
        self.annealing_kwargs.update(new_anneal_kwargs)

    def print_exception(self) -> None:
        if self.exceptions:
            for line in self.exceptions:
                print(line)

    @path_handling
    def to_json(self, path: Union[Path, str]):
        """Takes  care of a few attributes that are not JSON-
        serializable. These changes are reversed when loading from
        a JSON file again."""

        contents = self.__dict__.copy()

        # Handle non-serializable attributes
        contents['evals'] = self.evals.path.as_posix()
        contents['datasets'] = [(ds.path.as_posix(), ds.exp_type.value)
                                for ds in self.datasets]
        contents['start_dt'] = self.start_dt.isoformat(sep=' ')
        contents['stop_dt'] = self.stop_dt.isoformat(sep=' ')
        if self.result is not None:
            contents['result']['x'] = contents['result']['x'].tolist()

        with open(path, 'w') as file:
            json.dump(contents, file, indent=4)


class EvaluationLog:
    """A class to save the evaluation information in. Its attributes
    and methods are kept as simple as possible, and in theory, it should
    be possible to store a bit more data in here as well (for instance,
    the temperature or cycle during dual_annealing)."""

    @path_handling
    def __init__(self, path: Union[Path, str], log: pd.DataFrame):
        self.path = path
        self.log = log  # required columns: time, cost, parameter values

    def update(self, times: np.ndarray, values: np.ndarray,
               pvecs: np.ndarray) -> None:
        """Calling this method updates the contents of the log
        dataframe. This is the way in which optimization trackers
        regularly flush information."""

        dim = pvecs.shape[1]
        self.log = pd.DataFrame(
            index=range(times.size),
            columns=self.log.columns
        )
        self.log['time'] = times
        self.log['cost'] = values
        self.log.iloc[:, -dim:] = pvecs

    def to_pickle(self) -> None:
        self.log.dropna(subset=['time']).to_pickle(self.path.as_posix())


class SearcherScorer:
    """
    Object that scores Searcher objects on how similar their
    cleavage/binding dynamics are to experimental data.

    Attributes
    ----------
    searcher: Searcher
        The searcher object to be scored

    Methods
    -------
    compare(dataset, k_bind, multiprocessing=False, job_number=-1,
            log=True, weigh_errors=True, weigh_multiplicity=True)
        Scores the searcher on the basis of the dataset.
    """

    # this dictionary couples ExperimentType to the proper simulations
    _experiment_map = {ExperimentType.NUCLEASEQ: NucleaseqSimulation,
                       ExperimentType.CHAMP: ChampSimulation}

    def __init__(self, searcher: Searcher):
        self.searcher = searcher

    def compare(self, dataset: AggregateData, k_bind: float,
                multiprocessing=False, job_number=-1,
                log=True, weigh_errors=True, weigh_multiplicity=True) -> float:
        """
        Scores the searcher on the basis of the dataset.

        Arguments
        ---------
        dataset: AggregateData
            Dataset to compare to
        k_bind: float
            Rate at which the searcher binds targets in the experiment
        multiprocessing: bool
            Option to calculate searcher-target complex dynamics on
            many targets in parallel. False by default.
        job_number: int
            Number of processes for multiprocessing. Default is -1. Can
            be overridden by the multiprocessing argument.
        log: bool
            Indicates whether the residues between data and experiment
            should contribute logarithmically to the score. Default is
            True.
        weigh_errors: bool
            Indicates whether the experimental errors should be used to
            weigh the relative importance of the data points. Default
            is True.
        weigh_multiplicity: bool
            Indicates whether the multiplicity of the mismatch patterns
            should be used to weigh the relative importance of the data
            points. Default is True.

        Returns
        -------
        score: float
            A sum of squared errors between experimental data and
            simulated values. The score depends on whether
            contributions are logarithmic of linear and to how
            datapoints are weighted (see arguments)
        """

        sim_values = self.run_experiments(dataset, k_bind,
                                          multiprocessing, job_number)
        sqrd_error = self.calculate_sqrd_error(dataset, sim_values,
                                               log, weigh_errors,
                                               weigh_multiplicity)
        return sqrd_error

    def run_experiments(self, dataset: AggregateData, k_bind: float,
                        multiprocessing=False, job_number=-1) -> np.ndarray:
        """
        Runs all the experiment simulations associated to the dataset.

        Arguments
        ---------
        dataset: AggregateData
            Dataset to compare to
        k_bind: float
            Rate at which the searcher binds targets in the experiment
        multiprocessing: bool
            Option to calculate searcher-target complex dynamics on
            many targets in parallel. False by default.
        job_number: int
            Number of processes for multiprocessing. Default is -1. Can
            be overridden by the multiprocessing argument.

        Returns
        -------
        simulated_values: np.ndarray
            Results of the simulated experiments
        """

        exp_type = dataset.exp_type
        experiment = self._experiment_map[exp_type](
            searcher=self.searcher,
            k_bind=k_bind
        )

        mm_patterns = dataset.get_mm_patterns()

        if not multiprocessing:
            job_number = 1
        simulated_values = np.array(
            Parallel(n_jobs=job_number)
            (delayed(experiment)(pattern) for pattern in mm_patterns)
        )
        return simulated_values

    @classmethod
    def calculate_sqrd_error(cls, dataset: AggregateData,
                             sim_values: np.ndarray,
                             log=True, weigh_errors=True,
                             weigh_multiplicity=True) -> float:
        """
        Calculates the score of the simulated values compared to the
        dataset.

        Arguments
        ---------
        dataset: AggregateData
            Dataset to compare to
        sim_values: np.ndarray
            Results of the simulated experiments
        log: bool
            Indicates whether the residues between data and experiment
            should contribute logarithmically to the score. Default is
            True.
        weigh_errors: bool
            Indicates whether the experimental errors should be used to
            weigh the relative importance of the data points. Default
            is True.
        weigh_multiplicity: bool
            Indicates whether the multiplicity of the mismatch patterns
            should be used to weigh the relative importance of the data
            points. Default is True.

        Returns
        -------
        score: float
            A sum of squared errors between experimental data and
            simulated values. The score depends on whether
            contributions are logarithmic of linear and to how
            datapoints are weighted (see arguments)
        """

        # calculate weights
        exp_values = dataset.data['value']
        weights = np.ones_like(exp_values)
        if weigh_errors:
            weights = weights * dataset.weigh_errors(
                dataset.data[['value', 'error']],
                relative=log,  # absolute if lin, relative if log
                normalize=True
            ).to_numpy()
        if weigh_multiplicity:
            weights = weights * dataset.weigh_multiplicity(
                dataset.data['mismatch_array'],
                normalize=True
            ).to_numpy()

        if log:
            return cls.msd_log_cost_function(sim_values, exp_values, weights)
        else:
            return cls.msd_lin_cost_function(sim_values, exp_values, weights)

    @staticmethod
    def msd_lin_cost_function(sim_vals: np.ndarray, exp_vals: np.ndarray,
                              weights: np.ndarray) -> float:
        """Calculates the cost as a weighted sum over MSD between
        simulation and experimental data"""
        result = (weights * (sim_vals - exp_vals) ** 2).sum()
        return result

    @staticmethod
    def msd_log_cost_function(sim_vals: np.ndarray, exp_vals: np.ndarray,
                              weights: np.ndarray) -> float:
        """Calculates the cost as a weighted sum over logarithmic MSD
        between model and data"""

        # First check if logarithm can safely be taken, otherwise
        # infinite penalty
        if (np.any(sim_vals == 0.) or
                np.any(sim_vals / exp_vals < 1e-300) or
                np.any(sim_vals / exp_vals > 1e300)):
            return float('inf')

        result = (weights * np.log10(sim_vals / exp_vals) ** 2).sum()
        return result
