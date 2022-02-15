import sys
import os

# on Hidde's laptop
if os.path.exists('C:/Users/HP/depkengit/CRISPR_kinetic_model'):
    sys.path.append('C:/Users/HP/depkengit/CRISPR_kinetic_model')

import pandas as pd
import numpy as np

from model.training_set import TrainingSet
from model.fit_optimizer import track_dual_annealing


def get_root_dir(script_path):
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(script_path)),  # parent dir (=/run)
            os.pardir  # /.. (up a directory)
        )
    )
    return root_dir


def main(local_search, script_path='./fit_data_da.py', out_path='results/',
         array_id=1):

    # FIT SETTINGS
    dual_annealing_kwargs = {
        'no_local_search': not bool(local_search),
        'maxiter': 2500,
        'maxfun': 250000,
    }

    # initial vector and bounds
    initial_param_vector = np.ones(shape=(44,))
    param_lower_bounds = np.array(21 * [-10] + 20 * [0] + 3 * [-6])
    param_upper_bounds = np.array(41 * [20] + 3 * [6])
    param_bounds = np.stack([param_lower_bounds,
                             param_upper_bounds], axis=1)

    # collecting arguments
    root_dir = get_root_dir(script_path)
    out_dir = os.path.abspath(out_path)

    # preparing champ and nucleaseq data (ORIGINAL DATASET)
    champ_data = pd.read_csv(
        os.path.join(root_dir, 'data/SpCas9/Champ2020/orig_data.csv'),
        index_col=0, dtype={'mismatch_array': str}
    )
    champ_data.rename(columns={'mismatch_array': 'mismatch_positions'},
                      inplace=True)
    champ_data['experiment_name'] = 'CHAMP'

    nuseq_data = pd.read_csv(
        os.path.join(root_dir, 'data/SpCas9/NucleaSeq2020/orig_data.csv'),
        index_col=0, dtype={'mismatch_array': str}
    )
    nuseq_data.rename(columns={'mismatch_array': 'mismatch_positions'},
                      inplace=True)
    nuseq_data['experiment_name'] = 'NucleaSeq'

    all_data = champ_data.append(nuseq_data)
    all_data.reset_index(drop=True, inplace=True)
    training_set = TrainingSet(all_data)

    # run the optimization
    _ = track_dual_annealing(
        func=training_set.get_cost,
        x0=initial_param_vector,
        bounds=param_bounds,
        out_path=out_dir,
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
