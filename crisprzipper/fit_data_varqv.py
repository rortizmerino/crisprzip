import sys
import os

# on Hidde's laptop
if os.path.exists('C:/Users/HP/depkengit/CRISPR_kinetic_model'):
    sys.path.append('C:/Users/HP/depkengit/CRISPR_kinetic_model')

import numpy as np

from crisprzipper.model import read_dataset, TrainingSet
from crisprzipper.model import track_dual_annealing


def get_root_dir(script_path):
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(script_path)),
            # parent dir (=/bin)
            os.pardir  # /.. (up a directory)
        )
    )
    return root_dir


def main(target='E', script_path='./fit_data_da.py', out_path='results/',
         array_id=1):

    run_id = (int(array_id) - 1)
    visit_sweep = [2.3, 2.4, 2.5, 2.6, 2.7, 2.8]

    visit = visit_sweep[run_id//10]
    print(f"q_visit: {visit:.2f}")

    maxiter = 2000
    print(f"maxiter: {maxiter:d}")

    initial_temp = 5230.
    print(f"initl temp: {initial_temp:.1f}")

    final_temp = initial_temp * (2.**(visit-1)-1)/((2.+maxiter)**(visit-1)-1)
    print(f"final temp: {final_temp:.2e}")

    # FIT SETTINGS
    dual_annealing_kwargs = {
        'no_local_search': True,
        'maxiter': maxiter,
        'maxfun': 100*(maxiter*1.1),  # never reached
        'initial_temp': initial_temp,
        'restart_temp_ratio': 1E-20,  # never reached
        'visit': visit,
    }

    # initial vector and bounds
    initial_param_vector = np.ones(shape=(45,))
    param_lower_bounds = np.array(20 * [-10] + 20 * [0] + 5 * [-6])
    param_upper_bounds = np.array(40 * [20] + 5 * [6])
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

    # bin the optimization
    _ = track_dual_annealing(
        func=training_set.get_cost,
        x0=initial_param_vector,
        bounds=param_bounds,
        out_path=out_dir,
        cas9_log=False,  # 45 parameters instead of 44
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
