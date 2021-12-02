import pandas as pd
from Bio.Seq import Seq
from Bio.SeqUtils import nt_search


def generate_empty_searcher_dataframe():
    searcher_df = pd.DataFrame(
        {'searcher_id': pd.Series([], dtype=int),
         'type': pd.Series([], dtype=str),
         'target_direction': pd.Series([], dtype=str),
         'canonical_PAM': pd.Series([], dtype=str),
         'guide_length': pd.Series([], dtype=int)}
    ).set_index('searcher_id')
    return searcher_df


def generate_empty_guide_dataframe():
    guide_df = pd.DataFrame(
        {'guide_id': pd.Series([], dtype=int),
         'guide_seq': pd.Series([], dtype=str),
         'protospacer_seq': pd.Series([], dtype=str)}
    ).set_index('guide_id')
    return guide_df


def generate_empty_complex_dataframe():
    complex_df = pd.DataFrame(
        {'complex_id': pd.Series([], dtype=int),
         'searcher_id': pd.Series([], dtype=int),
         'guide_id': pd.Series([], dtype=int),
         'pam_seq': pd.Series([], dtype=str),
         'target_seq': pd.Series([], dtype=str),
         'mutations': pd.Series([], dtype=object),  # list
         'mismatches': pd.Series([], dtype=object),  # dict
         'is_canonical_pam': pd.Series([], dtype=bool),
         'is_on_target': pd.Series([], dtype=bool)
         }
    ).set_index('complex_id')
    return complex_df


def generate_empty_experiment_dataframe():
    experiment_df = pd.DataFrame(
        {'experiment_id': pd.Series([], dtype=int),
         'name': pd.Series([], dtype=str),
         'observable_name': pd.Series([], dtype=str),
         'observable_symbol': pd.Series([], dtype=str)
         }
    ).set_index('experiment_id')
    return experiment_df


def generate_empty_dataset_dataframe():
    dataset_df = pd.DataFrame(
        {'dataset_id': pd.Series([], dtype=int),
         'experiment_id': pd.Series([], dtype=int),
         'label': pd.Series([], dtype=str),
         'path': pd.Series([], dtype=str)
         }
    ).set_index('dataset_id')
    return dataset_df


def generate_empty_measurement_dataframe():
    measurement_df = pd.DataFrame(
        {'measurement_id': pd.Series([], dtype=int),
         'dataset_id': pd.Series([], dtype=int),
         'complex_id': pd.Series([], dtype=int),
         'value': pd.Series([], dtype=float),
         'error': pd.Series([], dtype=float)
         }
    ).set_index('measurement_id')
    return measurement_df


def generate_joint_dataframe(searcher_df, guide_df, complex_df):
    joint_df = complex_df.join(searcher_df, on='searcher_id')
    joint_df = joint_df.join(guide_df, on='guide_id')
    joint_df = joint_df[
        ['type',
         'protospacer_seq',
         'target_seq',
         'is_canonical_pam',
         'is_on_target',
         'mismatches']
    ]
    return joint_df


def generate_full_searcher_dataframe():
    searcher_df = generate_empty_searcher_dataframe()
    searcher_df.loc[100] = ['WT SpCas9', '3\'-to-5\'', 'GGN', 20]
    searcher_df.loc[201] = ['SpCas9-Hf1', '3\'-to-5\'', 'GGN', 20]
    searcher_df.loc[202] = ['SpCas9-Enh', '3\'-to-5\'', 'GGN', 20]
    searcher_df.loc[203] = ['SpCas9-Hypa', '3\'-to-5\'', 'GGN', 20]
    searcher_df.loc[300] = ['Cas12a', '5\'-to-3\'', 'TTTV', 20]
    searcher_df.loc[600] = ['Cascade', None, None, None]
    searcher_df.loc[800] = ['AGO2', None, None, None]
    return searcher_df


def generate_full_experiment_dataframe():
    experiment_df = generate_empty_experiment_dataframe()
    experiment_df.loc[10] = ['CHAMP', 'association constant', 'K_A']
    experiment_df.loc[20] = ['NucleaSeq', 'cleavage rate', 'k_clv']
    experiment_df.loc[31] = ['HiTS-FLIP', 'association rate', 'k_on']
    experiment_df.loc[32] = ['HiTS-FLIP', 'dissociation rate', 'k_off']
    return experiment_df


def generate_sample_guide_complex_dataframe():
    """
    This is a temporary function that copies some sample data from
    Misha into the general data format. Some elements of it might be
    reused in the functions that should import raw data.
    """

    # Collect empty dataframes for the guide and complex
    guide_df = generate_empty_guide_dataframe()
    complex_df = generate_empty_complex_dataframe()

    # import data in the original data folder
    df_raw = pd.read_csv(
        "/data/prepared_experimental\\NucleaSeq_dataset.csv"
    )

    # This data is for wildtype SpCas9 (I presume)
    wt_sp_cas9_id = 100
    wt_sp_cas9_pam = ['GGA', 'GGC', 'GGG', 'GGT']

    # First, we identify the on-target (protospacer & guide content)
    df_on_target = df_raw.loc[
        (df_raw['On Target'] == True) & (df_raw['Length difference'] == 0)]
    protospacer_content = df_on_target['Sequence'].head(1).iloc[0][-4:-24:-1]
    guide_content = str(Seq(protospacer_content).complement().transcribe())

    # Add the guide to the guide dataframe
    guide_df.loc[0] = [guide_content, protospacer_content]
    # ... and as an off-target to the complex dataframe
    for k in range(len(wt_sp_cas9_pam)):
        complex_df.loc[k] = [wt_sp_cas9_id,
                             0,
                             wt_sp_cas9_pam[k],
                             protospacer_content,
                             [], {}, True, True]

    # Second, we write an off-target dataframe
    df_raw_off = df_raw.copy().loc[df_raw['Mutation ID'] != 'OT']
    off_targets_df = pd.DataFrame(
        {
            'complex_id': list(range(1, 1 + len(df_raw_off))),
            'searcher_id': wt_sp_cas9_id,
            'guide_id': 0,
            'pam_seq': df_raw_off['Sequence'].str[-1:-4:-1],
            # takes last 3 characters
            'target_seq': df_raw_off['Sequence'].str[-4:-24:-1]
            # takes next 20 characters
        }
    ).set_index('complex_id')

    # Then, we loop over the rows to properly format everything
    idx = 1
    for i in df_raw_off.index:

        # Formatting mutation list
        mut_list = []
        for mut_code in df_raw_off.loc[i, 'Mutation ID'].split('|'):
            mut_loc = int(mut_code[2:-2])
            dropped_nt = protospacer_content[mut_loc - 1]
            new_mut_code = (dropped_nt +
                            str(mut_loc).rjust(2, '0') +
                            mut_code[-1])
            mut_list.append(new_mut_code)

        # Formatting mismatch dict
        mm_dict = {}
        # For now, we only include guide mm (not pam mm)
        for j in range(0, 20):
            off_target_nt = df_raw_off.loc[i, 'Sequence'][-(4 + j)]
            protospacer_nt = protospacer_content[j]
            if not nt_search(off_target_nt, protospacer_nt)[1:]:
                mm_dict[j + 1] = 'r' + guide_content[j] + ':d' + off_target_nt

        off_targets_df.loc[idx, 'mutations'] = str(mut_list)
        off_targets_df.loc[idx, 'mismatches'] = str(mm_dict)
        off_targets_df.loc[idx, 'is_canonical_pam'] = \
            (off_targets_df.loc[idx, 'pam_seq'] in wt_sp_cas9_pam)
        # The above does not include ambiguous 'N' base pair
        off_targets_df.loc[idx, 'is_on_target'] = mm_dict == {}
        off_targets_df.loc[idx, 'mm_num'] = len(mm_dict)

        idx += 1

    # finishing off
    off_targets_df = off_targets_df.sort_values(
        ['mm_num', 'target_seq', 'pam_seq']
    )
    off_targets_df = off_targets_df.drop(columns=['mm_num'])
    off_targets_df = off_targets_df.drop_duplicates()
    off_targets_df = off_targets_df.reset_index(drop=True)
    complex_df = complex_df.append(off_targets_df, ignore_index=True)

    # Also, we obtain the dataframes with the database results
    experiment_df = generate_full_experiment_dataframe()
    dataset_df = generate_empty_dataset_dataframe()
    measurement_df = generate_empty_measurement_dataframe()

    # get the id of NucleaSeq in the experiment df
    experiment_id = experiment_df.index[experiment_df['name'] == 'NucleaSeq'] \
        .tolist()[0]

    # make a dataset entry
    dataset_df.loc[0] = [experiment_id, 'Misha\'s prepared NucleaSeq data',
                         "C:\\Users\\HP\\depkengit\\CRISPR_kinetic_model\\"
                         "prepared_experimental\\NucleaSeq_dataset.csv"
                         ]

    # now get the data and store them
    k = 0
    for i in df_raw.index:
        pam = df_raw.loc[i, 'Sequence'][-1:-4:-1]  # off-target pam
        target = df_raw.loc[i, 'Sequence'][-4:-24:-1]  # off-target content
        complex_ids = complex_df.index[
            (complex_df['pam_seq'] == pam) &
            (complex_df['target_seq'] == target)
            ].tolist()
        for j in complex_ids:
            value = df_raw.loc[i, 'cleavage_rate']
            error = max(
                (df_raw.loc[i, 'cleavage_rate'] -
                 df_raw.loc[i, 'cleavage_rate_5th_pctl']),
                (df_raw.loc[i, 'cleavage_rate_95th_pctl'] -
                 df_raw.loc[i, 'cleavage_rate'])
            )  # for error, take the longer side of the confidence int.

            measurement_df.loc[k] = [0, j, value, error]
            k += 1

    measurement_df = measurement_df.astype({'complex_id': int,
                                            'dataset_id': int})

    return guide_df, complex_df, dataset_df, measurement_df


def main():
    guide_df, complex_df, dataset_df, measurement_df =\
        generate_sample_guide_complex_dataframe()
    pass


if __name__ == '__main__':
    main()
