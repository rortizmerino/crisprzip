import sys
import os

# on Hidde's laptop
if os.path.exists('C:/Users/HP/depkengit/CRISPR_kinetic_model'):
    sys.path.append('C:/Users/HP/depkengit/CRISPR_kinetic_model')

import numpy as np

from model.training_set import read_dataset, TrainingSet
from model.fit_optimizer import track_dual_annealing


def get_root_dir(script_path):
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(script_path)),
            # parent dir (=/run)
            os.pardir  # /.. (up a directory)
        )
    )
    return root_dir


def main(target='E', script_path='./fit_data_da.py', out_path='results/',
         array_id=1):
    # FIT SETTINGS
    dual_annealing_kwargs = {
        'no_local_search': True,
        'maxiter': 1500,
        'maxfun': 250000,
        'initial_temp': 20000,
        'restart_temp_ratio': 5E-3,
        'visit': 2.8,
    }

    # initial vector and bounds
    initial_param_vector = np.ones(shape=(43,))
    param_lower_bounds = np.array(20 * [-10] + 20 * [0] + 3 * [-4])
    param_upper_bounds = np.array(40 * [20] + 3 * [4])
    param_bounds = np.stack([param_lower_bounds,
                             param_upper_bounds], axis=1)

    # collecting arguments
    root_dir = get_root_dir(script_path)
    out_dir = os.path.abspath(out_path)

    # preparing champ and nucleaseq data (MY AGGREGATE DATASET)
    experiments = ['NucleaSeq', 'Champ']
    datasets = []
    for exp in experiments:
        path = os.path.join(root_dir, f'data/SpCas9/{exp}2020/target{target}/'
                                      f'aggr_data.csv')
        datasets += [read_dataset(path)]

    # make training set
    training_set = TrainingSet(datasets, experiments)

    # cost function with fixed rates
    k_f = 4.
    k_clv = 3.

    costfunc = lambda param_vec: training_set.get_cost(
        np.concatenate((
            param_vec[:-2],  # OT landscape + mm penalties + k_off
            np.array([k_f, k_clv]),  # fixed rates
            param_vec[-2:]  # k_on for nuseq + champ
        )),
        multiprocessing=True
    )

    # run the optimization
    _ = track_dual_annealing(
        func=costfunc,
        x0=initial_param_vector,
        bounds=param_bounds,
        out_path=out_dir,
        cas9_log=False,
        **dual_annealing_kwargs
    )


if __name__ == "__main__":

    # (cluster) keyword arguments: script_path, array_id and out_path
    kwargs = {'script_path': sys.argv[0]}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, val = arg.split('=')
            if key == 'array_id':
                kwargs[key] = int(val)
            else:
                kwargs[key] = val

    # arguments: anything needed for this script
    args = [arg for arg in sys.argv[1:] if not ('=' in arg)]

    main(*args, **kwargs)
