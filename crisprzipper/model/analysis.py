"""
Classes with which logs of optimization runs can be analyzed, with some
basic analytics functionality.

Classes:
    JobAnalyzer
    EvalsAnalyzer(EvaluationLog)
    OptRunAnalyzer(OptRunLog)

"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from crisprzipper.model.fitting import OptRunLog, EvaluationLog
from crisprzipper.model.tools import path_handling


@dataclass
class JobAnalyzer:
    """
    Analyzes all the results of an optimization job. A JobAnalyzer
    object is essentially a collection of OptRunAnalyzer objects that has
    a number of methods to do quick analyses on them.

    Attributes
    ----------
    analyzers: List[OptRunAnalyzer]
        The OptRunAnalyzer objects based on the output files in the
        job directory.
    job_ids: List[str]
        The corresponding jobs in which the analyzers were created.
    run_ids: List[str]
        The bin ids of the analyzers. In combination with the job_ids,
        these make unique identificators of the various runs.

    Methods
    -------
    summarize()
        Makes a dataframe that summarizes all bin information
    """

    analyzers: List['OptRunAnalyzer'] = None
    job_ids: List[str] = None
    run_ids: List[str] = None

    def __add__(self, other):
        """Allows concatination of two JobAnalyzer objects with
        the + symbol"""
        return JobAnalyzer(self.analyzers + other.analyzers,
                           self.job_ids + other.job_ids,
                           self.run_ids + other.run_ids)

    @classmethod
    @path_handling
    def from_job_dir(cls, job_path: Union[str, Path],
                     param_filename: str = "params.json") -> 'JobAnalyzer':
        """Walks through a job directory and loads all bin data
        according to the newest output layout. (params.json and
        evals.pkl).

        Arguments
        ---------
        job_path: Union[str, Path]
            Path in which to look for bin results
        param_filename: str
            The filename of the param files. Default is params.json.
        """

        if not job_path.is_dir():
            raise FileNotFoundError

        job_ids = []
        run_ids = []
        analyzers = []
        for root, _, files in os.walk(job_path):
            if param_filename in files:
                analyzers += [OptRunAnalyzer.from_json(
                    Path(root).joinpath(param_filename)
                )]
                job_ids += [job_path.name]
                run_ids += [Path(root).name]

        return cls(analyzers, job_ids, run_ids)

    @classmethod
    @path_handling
    def from_job_dir_old(cls, job_path: Union[str, Path], pvec_type: str,
                         log_filename: str = "log.txt",
                         evals_filename: str = "evals.pkl",
                         quick_import: bool = True) -> 'JobAnalyzer':
        """Walks through a job directory and loads all bin data
        according to the previous output layout (log.txt and evals.pkl).

        Arguments
        ---------
        job_path: Union[str, Path]
            Path in which to look for bin results
        pvec_type: str
            Name of the ParameterVector class that corresponds with the
            bin setup. Previously, this was not stored, so it should
            be given as an argument to make all analyses work.
        log_filename: str
            The filename of the log files. Default is log.txt.
        evals_filename: str
            The filename of the evals files. Default is evals.pkl.
        quick_import: bool
            If True, skips the evals file to instead just import the
            (summarized) evaluation log from the log file.
        """

        if not job_path.is_dir():
            raise FileNotFoundError

        job_ids = []
        run_ids = []
        analyzers = []
        for root, _, files in os.walk(job_path):
            if log_filename in files and (quick_import or
                                          evals_filename in files):
                if quick_import:
                    analyzers += [OptRunAnalyzer.from_text_log(
                        Path(root).joinpath(log_filename),
                        None, pvec_type
                    )]
                else:
                    analyzers += [OptRunAnalyzer.from_text_log(
                        Path(root).joinpath(log_filename),
                        Path(root).joinpath(evals_filename),
                        pvec_type
                    )]
                job_ids += [job_path.name]
                run_ids += [Path(root).name]

        return cls(analyzers, job_ids, run_ids)

    def summarize(self) -> pd.DataFrame:
        """Makes a dataframe that summarizes all bin information"""
        summary = pd.DataFrame(
            data={
                'Job id': self.job_ids,
                'Run id': self.run_ids,
                'Final cost': [a.get_lowest_cost() for a in self.analyzers],
                'Total evals': [a.result['nfev'] for a in self.analyzers],
                'Runtime (min)': [a.get_total_runtime() / 60
                                  for a in self.analyzers]
            })
        return summary

    def to_top_selection(self, top: int) -> 'JobAnalyzer':
        summary = self.summarize()
        selection_ids = (summary.sort_values("Final cost", ascending=True)
                         .index).to_list()[:top]
        return JobAnalyzer(
            [self.analyzers[i] for i in selection_ids],
            [self.job_ids[i] for i in selection_ids],
            [self.run_ids[i] for i in selection_ids]
        )


class EvalsAnalyzer(EvaluationLog):
    """An object to analyze the EvaluationLog that was created during
    an optimization bin."""

    @path_handling
    def __init__(self, path: Union[Path, str], log: pd.DataFrame):
        super().__init__(path, log)

        if "gain" not in self.log.columns:
            self.__calculate_cost_gain()

    def __calculate_cost_gain(self) -> None:
        cost = self.log.cost.to_numpy()
        lowest_cost = np.minimum.accumulate(cost)
        cost_gain = np.concatenate((np.zeros(1), cost[1:] - lowest_cost[:-1]))
        self.log.insert(2, "gain", cost_gain)

    @classmethod
    @path_handling
    def from_pickle(cls, path: Union[Path, str]) -> 'EvaluationLog':
        """Constructs an EvalsAnalyzer object from a pickle file."""
        log = pd.read_pickle(path.as_posix())
        return cls(path, log)

    def summarize_log(self):
        """Returns a log DataFrame with only the costfunc evaluations
        that improved upon the existing solution, significantly
        reducing memory usage."""
        return self.log.loc[(self.log.index == 0) |
                            (self.log["gain"] < 0.)]

    def rename_param_columns(self, new_column_names: List[str]) -> None:
        """Rename columns for runs that were performed before different
        ParameterVector objects could be used, such that they had no
        descriptive parameter names."""
        self.log.columns = (
            self.log.columns[:-len(new_column_names)].to_list() +
            new_column_names
        )

    def calculate_cycles(self, dim):
        """Finds the cycle no. on the basis of the step no. This only
        applies to dual_annealing runs with no_local_search=True
        (which is the default)."""
        steps = self.log.index.to_numpy()
        cycles = (steps - 1) // (2 * dim) + 1
        return pd.Series(index=self.log.index, data=cycles)

    def calculate_temperature(self, dim, initial_temp=5340, visit=2.62):
        """Finds the temperature on the basis of the step no. This only
        applies to dual_annealing runs with no_local_search=True
        (which is the default)."""
        def temp(cycle, temp0, q_v):
            return temp0 * ((2 ** (q_v - 1) - 1) /
                            ((1 + cycle) ** (q_v - 1) - 1))

        temperatures = temp(
            cycle=self.calculate_cycles(dim).loc[1:].to_numpy(),
            temp0=initial_temp, q_v=visit)
        return pd.Series(index=self.log.loc[1:].index, data=temperatures)


class OptRunAnalyzer(OptRunLog):
    """An object to analyze the OptRunLog that was created during
    an optimization bin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.evals, EvalsAnalyzer):
            self.evals = EvalsAnalyzer(self.evals.path, self.evals.log)

    @classmethod
    @path_handling
    def from_json(cls, path: Union[Path, str]) -> 'OptRunAnalyzer':
        """Constructs an OptRunAnalyzer object from a JSON file.
        Reverses the changes that were made to non-JSON-serializable
        attributes as an OptRunLog object was exported previously."""

        with open(path, 'r') as file:
            contents = json.load(file)

        # Handle non-serializable attributes
        contents['evals'] = EvalsAnalyzer.from_pickle(contents['evals'])
        # TODO: handle dataset files too (how to find exp_type?)
        contents['start_dt'] = datetime.fromisoformat(contents['start_dt'])
        contents['stop_dt'] = datetime.fromisoformat(contents['stop_dt'])
        if 'result' in contents:
            contents['result']['x'] = np.array(contents['result']['x'])

        return cls(**contents)

    @classmethod
    @path_handling
    def from_text_log(cls, log_path: Union[Path, str],
                      evals_path: Union[Path, str] = None,
                      pvec_type: str = "EslamiParams"):
        """Reads all the metadata & bin parameters from the output
        file format that was used previously. These files do not have
        all data saved that are stored as attributes in the
        OptRunAnalyzer class, which may lead to problems when analyzing
        them.

        Parameters
        ----------
        log_path: Union[Path, str]
            Path to the log file to be read.
        evals_path: Union[Path, str]
            Path to the evals file to be read. Default is None, in which
            case the (summarized) evals data will be retrieved from the
             log file.
        pvec_type: str
            Name of the ParameterVector class that corresponds with the
            bin setup. Previously, this was not stored, so it should
            be given as an argument to make all analyses work.
        """

        params = cls.__read_params_from_log(log_path)

        if evals_path is not None:
            evals = EvalsAnalyzer(evals_path,
                                  pd.read_pickle(evals_path))
        else:
            evals = EvalsAnalyzer(log_path,
                                  cls.__read_evals_from_log(log_path))
        pvec_class = getattr(crisprzipper.model.parameter_vector, pvec_type)

        if params['pvec_info']['param_names'][0] == "p00":
            evals.rename_param_columns(pvec_class.param_names)

        return cls(
            pvec_type=pvec_type,
            pvec_info=params['pvec_info'],
            evals=evals,
            datasets=[],
            annealing_kwargs=params['annealing_kwargs'],
            score_kwargs={},
            start_dt=params['start_dt'],
            stop_dt=params['stop_dt'],
            result=params['result'],
            exceptions=[]
        )

    # noinspection PyUnboundLocalVariable
    @staticmethod
    def __read_params_from_log(log_path: Union[Path, str]) -> dict:
        """Private method to read bin params from old log files."""
        with open(log_path, 'r') as log_file:
            lines = log_file.readlines()

        exit_message = lines[3].strip()
        for i, line in enumerate(lines):

            # Optimization result
            if line[:6] == 'result':
                param_names = [s.strip() for s
                               in lines[i - 1][:-1].split('\t')[1:]]
                final_result = np.array(
                    [float(val) for val in line[:-1].split('\t')[1:]]
                )
            elif line[:7] == 'initial':
                initial = [float(val) for val in line[:-1].split('\t')[1:]]
            elif line[:9] == 'lower bnd':
                lb = [float(val) for val in line[:-1].split('\t')[1:]]
            elif line[:9] == 'upper bnd':
                ub = [float(val) for val in line[:-1].split('\t')[1:]]

            elif line[:29].strip() == 'Started':
                start_dt = datetime.strptime(line[-21:].strip(),
                                             "%a %d-%m-%Y  %H:%M")
            elif line[:29].strip() == 'Stopped':
                stop_dt = datetime.strptime(line[-21:].strip(),
                                            "%a %d-%m-%Y  %H:%M")
            elif line[:29].strip() == 'Total step number':
                nfev = int(line[-21:])
            elif line[:29].strip() == 'Final cost':
                fun = float(line[-21:])

            # annealing parameters
            elif line[:29].strip() == 'input parameters':
                param_range = range(i + 2, i + 14)
                break

        pvec_info = {'param_names': param_names, 'x0': initial,
                     'lb': lb, 'ub': ub}
        result_dict = {'x': final_result, 'message': exit_message,
                       'fun': fun, 'nfev': nfev}

        def autoconvert(string):
            """A bit hacky function to recognize string types"""
            if string.lower() == "true":
                return True
            elif string.lower() == "false":
                return False
            elif string.lower() == 'none':
                return None
            elif string.lower() == '()':
                return ()
            try:
                return int(string)
            except ValueError:
                try:
                    return float(string)
                except ValueError:
                    return string

        annealing_kwargs = dict(zip(
            [lines[i][:29].strip() for i in param_range],
            [autoconvert(lines[i][-21:].strip()) for i in param_range]
        ))

        return {'pvec_info': pvec_info,
                'annealing_kwargs': annealing_kwargs,
                'start_dt': start_dt,
                'stop_dt': stop_dt,
                'result': result_dict}

    @staticmethod
    def __read_evals_from_log(log_path: Union[Path, str]) -> pd.DataFrame:
        """Private method to read evals dataframe from old log files."""
        with open(log_path, 'r') as log_reader:
            log_lines = log_reader.readlines()
            log_start = 0
            for line in log_lines:
                if line.split('\t')[0].strip() == 'Cycle no':
                    break
                else:
                    log_start += 1

        dataframe = pd.read_table(log_path,
                                  index_col="Cycle no",
                                  skiprows=log_start,
                                  delimiter='\t',
                                  skipinitialspace=True)
        dataframe.index.name = None
        dataframe = dataframe.rename(columns={"Time (s)": "time",
                                              "Cost": "cost",
                                              "Cost gain": "gain"})
        return dataframe

    def get_total_runtime(self):
        return (self.stop_dt - self.start_dt).total_seconds()

    def get_evaluation_time(self):
        return self.get_total_runtime() / self.result['nfev']

    # noinspection PyUnresolvedReferences
    def get_summarized_pvecs(self) -> pd.Series:
        """First summarizes the evals data (showing only imroving
        evaluations), the returning the corresponding ParameterVector
        instances as a pandas Series."""
        pvec_class = getattr(crisprzipper.model.parameter_vector, self.pvec_type)
        dim = len(pvec_class.x0)
        evals_summary = self.evals.summarize_log()
        return pd.Series(
            index=evals_summary.index,
            data=[pvec_class(x) for x
                  in evals_summary.iloc[:, -dim:].to_numpy()]
        )

    # noinspection PyUnresolvedReferences
    def get_best_pvec(self) -> ParameterVector:
        """Finds the best solution that was encountered during
        optimization (which is not necessarily the solution of the
        dual_annealing algorithm), returns it as a ParameterVector
        instance."""
        pvec_class = getattr(crisprzipper.model.parameter_vector, self.pvec_type)
        dim = len(pvec_class.x0)
        evals_summary = self.evals.summarize_log()
        return pvec_class(evals_summary.iloc[-1, -dim:])

    def get_lowest_cost(self) -> float:
        """Finds the best solution that was encountered during
        optimization (which is not necessarily the solution of the
        dual_annealing algorithm), returns its cost."""
        return self.evals.log.cost.min()

    # noinspection PyUnresolvedReferences
    def get_run_details(self, summary=True) -> pd.DataFrame:
        pvec_class = getattr(crisprzipper.model.parameter_vector, self.pvec_type)
        dim = len(pvec_class.x0)
        cycles = self.evals.calculate_cycles(dim)
        temper = self.evals.calculate_temperature(
            dim,
            self.annealing_kwargs['initial_temp'],
            self.annealing_kwargs['visit']
        )
        log = self.evals.summarize_log() if summary else self.evals.log
        log = log[['time', 'cost', 'gain']]
        log.insert(1, 'cycle', cycles.loc[log.index])
        log.insert(2, 'temp', temper.loc[log.loc[1:].index])
        log['temp'][0] = self.annealing_kwargs['initial_temp']
        return log
