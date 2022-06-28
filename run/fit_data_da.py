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
            os.path.dirname(os.path.abspath(script_path)),  # parent dir (=/run)
            os.pardir  # /.. (up a directory)
        )
    )
    return root_dir


def main(normalized_weights=False, extra_nucleaseq_weight=None,
         script_path='./fit_data_da.py',
         out_path='results/',
         array_id=1):

    # FIT SETTINGS
    dual_annealing_kwargs = {
        # 'no_local_search': not bool(local_search),
        'no_local_search': False,
        'maxiter': 2500,
        'maxfun': 250000,
        # 'maxfun': 25,
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

    # preparing champ and nucleaseq data (ORIGINAL DATASET)
    experiments = ['NucleaSeq', 'Champ']
    datasets = []
    for exp in experiments:
        path = os.path.join(root_dir, f'data/SpCas9/{exp}2020/orig_data.csv')
        datasets += [read_dataset(path)]

    # make training set
    if extra_nucleaseq_weight is None:
        experiment_weights = None
    else:
        experiment_weights = [float(extra_nucleaseq_weight), 1.]

    training_set = TrainingSet(datasets, experiments,
                               experiment_weights=experiment_weights,
                               normalize_weights=bool(normalized_weights))

    # run the optimization
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
