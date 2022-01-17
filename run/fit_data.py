import sys
import os

import pandas as pd
import numpy as np

from training_set import TrainingSet
from sim_anneal import SimulatedAnnealer


def main(argv):
    # collecting arguments
    root_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(os.path.abspath(argv[0])),  # parent dir (=/run)
            os.pardir  # /.. (up a directory)
        )
    )
    out_file = argv[1]

    optimization_kwargs = {
        'check_cycle': 10,  # 1000?
        'step_size': 2.,
        'cost_tolerance': 1E-3,

        'initial_temperature': 0.1,
        'final_temperature': .0005,
        'cooling_rate': 0.99,

        'acceptance_bounds': (0.4, 0.6),
        'adjust_factor': 1.1,
    }

    # fitting champ and nucleaseq data
    champ_data = pd.read_csv(
        os.path.join(root_dir, 'data/SpCas9/Champ2020/aggr_data.csv'),
        index_col=0, dtype={'mismatch_array': str}
    )
    champ_data.rename(columns={'mismatch_array': 'mismatch_positions'},
                      inplace=True)
    champ_data['experiment_name'] = 'CHAMP'

    nuseq_data = pd.read_csv(
        os.path.join(root_dir, 'data/SpCas9/NucleaSeq2020/aggr_data.csv'),
        index_col=0, dtype={'mismatch_positions': str}
    )

    all_data = champ_data.append(nuseq_data)
    all_data.reset_index(drop=True, inplace=True)
    training_set = TrainingSet(all_data)

    # initial param vector
    guide_length = 20
    param_vector_ones = np.ones(2 * guide_length + 4)

    # trial no
    trial_no = 10000

    # run the optimization
    cost_func = lambda param_vector: training_set.get_cost(param_vector,
                                                           multiprocessing=False)

    SimulatedAnnealer(
        function=training_set.get_cost,
        initial_param_vector=param_vector_ones,
        parameter_bounds=(
            np.array((2 * guide_length + 1) * [0] + 3 * [-4]),
            np.array((2 * guide_length + 1) * [10] + 3 * [4])
        ),
        log_file=out_file,
        max_trial_no=trial_no,
        **optimization_kwargs
    ).run()


if __name__ == "__main__":
    main(sys.argv)
