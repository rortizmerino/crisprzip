import sys
import os

# on Hidde's laptop
if os.path.exists('C:/Users/HP/depkengit/CRISPR_kinetic_model'):
    sys.path.append('C:/Users/HP/depkengit/CRISPR_kinetic_model')

import pandas as pd
import numpy as np

from model.training_set import TrainingSet
from model.sim_anneal import SimulatedAnnealer


def get_root_dir(script_path):
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(script_path)),  # parent dir (=/run)
            os.pardir  # /.. (up a directory)
        )
    )
    return root_dir


def main(script_path='./fit_data_custom_sa.py', out_path='results/', array_id=1):

    # collecting arguments
    root_dir = get_root_dir(script_path)
    out_file = os.path.join(out_path, 'custom_simanneal_log.csv')

    # FIT SETTINGS

    # trial no
    trial_no = 25000

    # initial temp
    temp = 10

    optimization_kwargs = {
        'check_cycle': 10,  # 1000?
        'step_size': 2.,
        'cost_tolerance': .1,

        'initial_temperature': temp,
        'final_temperature': temp/1000,
        'cooling_rate': 0.99,

        'acceptance_bounds': (0.4, 0.6),
        'adjust_factor': 1.1,
    }

    # EXECUTES FIT

    # fitting champ and nucleaseq data (ORIGINAL DATASET)
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

    # initial param vector
    guide_length = 20
    param_vector_ones = np.ones(2 * guide_length + 4)

    # run the optimization
    SimulatedAnnealer(
        function=training_set.get_cost,
        initial_param_vector=param_vector_ones,
        parameter_bounds=(
            np.array((guide_length+1) * [-10] + guide_length * [0] + 3 * [-6]),
            np.array((2 * guide_length + 1) * [20] + 3 * [6])
        ),
        log_file=out_file,
        max_trial_no=trial_no,
        **optimization_kwargs
    ).run()


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
