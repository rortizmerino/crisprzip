import sys
import os

# on Hidde's laptop
if os.path.exists('C:/Users/HP/depkengit/CRISPR_kinetic_model'):
    sys.path.append('C:/Users/HP/depkengit/CRISPR_kinetic_model')

import numpy as np

from crisprzipper.model import read_dataset, TrainingSet
from crisprzipper.model import SimulatedAnnealer


def get_root_dir(script_path):
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(script_path)),  # parent dir (=/bin)
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
    experiments = ['NucleaSeq', 'Champ']
    datasets = []
    for exp in experiments:
        path = os.path.join(root_dir, f'data/SpCas9/{exp}2020/orig_data.csv')
        datasets += [read_dataset(path)]

    # make training set
    training_set = TrainingSet(datasets, experiments)

    # initial param vector
    guide_length = 20
    param_vector_ones = np.ones(2 * guide_length + 4)

    # bin the optimization
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
