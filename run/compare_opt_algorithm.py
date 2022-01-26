import sys
import os
from time import process_time

import pandas as pd
import numpy as np
import scipy.optimize

hp_root_dir = 'C:\\Users\\HP\\depkengit\\CRISPR_kinetic_model'
if os.path.exists(hp_root_dir):
    sys.path.append(hp_root_dir)
    sys.path.append(os.path.join(hp_root_dir, 'model'))

from model.training_set import TrainingSet
from model.sim_anneal import SimulatedAnnealer


def main(argv):
    # find root dir
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(argv[0])),  # parent dir (=/run)
            os.pardir  # /.. (up a directory)
        )
    )

    # Collecting arguments
    algorithm = argv[1]
    max_evals = int(argv[2])
    out_file = argv[3]

    # initial vector and bounds
    initial_param_vector = np.ones(shape=(44,))
    param_lower_bounds = np.array(21 * [-10] + 20 * [0] + 3 * [-6])
    param_upper_bounds = np.array(41 * [20] + 3 * [6])
    param_bounds = np.stack([param_lower_bounds,
                             param_upper_bounds], axis=1)

    # Making a training set and analyzer
    training_set = make_orig_training_set(root_dir)

    with AlgorithmAnalyzer(training_set.get_cost, max_evals, out_file) \
            as analyzer:

        # Pick and run algorithm
        if algorithm == 'CustomSA':
            run_custom_sa(analyzer, max_evals)
            out = 'done.'
        elif algorithm == 'FastSA':
            out = scipy.optimize.dual_annealing(
                func=analyzer.evaluate_and_track,
                bounds=param_bounds,
                visit=2,
                accept=1,
                maxfun=max_evals,
                no_local_search=True,
                x0=initial_param_vector
            )
        elif algorithm == 'BasinHopping':  # Not really working yet
            out = scipy.optimize.basinhopping(
                func=analyzer.evaluate_and_track,
                x0=initial_param_vector,
                niter=max_evals,
                T=1.0,
                stepsize=0.5,
                disp=True
            )
        elif algorithm == 'DualAnnealing':
            out = scipy.optimize.dual_annealing(
                func=analyzer.evaluate_and_track,
                bounds=param_bounds,
                maxfun=max_evals,
                x0=initial_param_vector
            )
        elif algorithm == 'DifferentialEvolution':
            out = scipy.optimize.differential_evolution(
                func=analyzer.evaluate_and_track,
                bounds=param_bounds,
                strategy='best1bin',
                maxiter=int(max_evals / (15 * 44) - 1),
                popsize=15,
                tol=1E-5,
                mutation=(.5, 1),
                recombination=.7,
                disp=True,
                polish=True,
                init='latinhypercube',
                workers=1
            )
        elif algorithm == 'DifferentialEvolutionNoPolish':
            out = scipy.optimize.differential_evolution(
                func=analyzer.evaluate_and_track,
                bounds=param_bounds,
                strategy='best1bin',
                maxiter=int(max_evals / (15 * 44) - 1),
                popsize=15,
                tol=1E-5,
                mutation=(.5, 1),
                recombination=.7,
                disp=True,
                polish=False,
                init='latinhypercube',
                workers=1
            )
        else:
            raise ValueError('Unknown optimization method')

        print(out)


def make_orig_training_set(root_dir):

    # original champ and nucleaseq datasets
    orig_nuseq_data = pd.read_csv(
        os.path.join(root_dir, 'data/SpCas9/NucleaSeq2020/orig_data.csv'),
        index_col=0,
        dtype={'mismatch_array': str}
    )
    orig_nuseq_data['experiment_name'] = 'nucleaseq'

    orig_champ_data = pd.read_csv(
        os.path.join(root_dir, 'data/SpCas9/Champ2020/orig_data.csv'),
        index_col=0,
        dtype={'mismatch_array': str}
    )
    orig_champ_data['experiment_name'] = 'champ'

    for df in [orig_nuseq_data, orig_champ_data]:
        df.rename(columns={'mismatch_array': 'mismatch_positions'},
                  inplace=True)

    # making a training set
    orig_all_data = orig_champ_data.append(orig_nuseq_data)
    orig_all_data.reset_index(drop=True, inplace=True)
    orig_training_set = TrainingSet(orig_all_data)

    return orig_training_set


class AlgorithmAnalyzer:
    """
    Keeping track of the calls of the evaluation function. Can be
    used to compare optimization methods.
    """

    def __init__(self, eval_func, max_evals, out_path):

        # pre-allocating saves time
        self.max_evals = max_evals
        self.eval_time = np.zeros(shape=max_evals, dtype=float)
        self.eval_vals = np.zeros(shape=max_evals, dtype=float)
        self.head = 0

        self.start_time = 0

        self.out_path = out_path

        self.eval_func = eval_func

    def __enter__(self):
        """Run as context manager to handle exceptions"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Run as context manager to handle exceptions"""
        self.eval_time = np.trim_zeros(self.eval_time, trim='b')
        self.eval_vals = np.trim_zeros(self.eval_vals, trim='b')

        self.export(self.out_path)

    def evaluate_and_track(self, param_vector):
        """Like the original evaluation function, but storing the
        time and result as a class attribute"""

        # get value
        result = self.eval_func(param_vector)

        # set first time
        if self.head == 0:
            self.start_time = process_time()

        # check array length, update if necessary
        if self.head >= self.eval_time.size:
            self.eval_time = np.append(self.eval_time,
                                       np.zeros(100, dtype=float),
                                       axis=0)
            self.eval_vals = np.append(self.eval_vals,
                                       np.zeros(100, dtype=float),
                                       axis=0)

        # update arrays
        self.eval_time[self.head] = process_time() - self.start_time
        self.eval_vals[self.head] = result
        self.head += 1

        return result

    def get_min_evals(self):
        return np.minimum.accumulate(self.eval_vals)

    def export(self, path):
        """Writes evaluation values to csv"""
        pd.DataFrame({
            'time': self.eval_time,
            'cost': self.eval_vals,
        }).to_csv(os.path.abspath(path))


def run_custom_sa(analyzer, max_evals):

    optimization_kwargs = {
        'check_cycle': 10,  # 1000?
        'step_size': 0.5,
        'cost_tolerance': 1e-5,

        'initial_temperature': 1e-1,  # ?
        'final_temperature': 1e-5,
        'cooling_rate': 0.98,

        'acceptance_bounds': (0.4, 0.6),
        'adjust_factor': 1.1,
    }

    SimulatedAnnealer(
        function=analyzer.evaluate_and_track,
        initial_param_vector=np.ones(44),
        parameter_bounds=(
            np.array((20 + 1) * [-10] + 20 * [0] + 3 * [-6]),
            np.array((2 * 20 + 1) * [20] + 3 * [6])
        ),
        log_file='out/customSA/customSA.txt',
        max_trial_no=max_evals,
        initialize_temperature=False,
        **optimization_kwargs
    ).run()


if __name__ == "__main__":
    main(sys.argv)
