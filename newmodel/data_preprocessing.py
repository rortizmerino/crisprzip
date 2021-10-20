import pandas as pd
import ast


def find_mismatch_positions(complex_df):
    complex_df = complex_df.copy()
    for i in complex_df.index:
        mismatch_dict = complex_df.loc[i, 'mismatches']
        if str(mismatch_dict) == '{}':
            mismatch_keys = []
        else:
            mismatch_keys = list(ast.literal_eval(mismatch_dict).keys())

        guide_length = len(complex_df.loc[i, 'target_seq'])

        mismatch_array = ''
        for b in range(guide_length):
            mismatch_array += str(int(b+1 in mismatch_keys))
        complex_df.loc[i, 'mismatch_array'] = mismatch_array

    return complex_df['mismatch_array']


def get_mismatch_map_dict(complex_df):
    mismatch_positions = find_mismatch_positions(complex_df)
    mismatch_dict = {}
    for array in mismatch_positions.unique():
        mismatch_dict[array] = mismatch_positions[
            mismatch_positions == array].index.to_list()
    return mismatch_dict


def aggregate_measurements(measurement_df, mismatch_map, agg_func):
    agg_measurement_df = pd.DataFrame(
        {'mismatch_positions': pd.Series([], dtype=str),
         'agg_value': pd.Series([], dtype=float),
         'agg_error': pd.Series([], dtype=float)}
    )

    i = 0
    for array in mismatch_map:
        measurement_subset = measurement_df.loc[
            measurement_df['complex_id'].isin(mismatch_map[array])
        ]
        agg_value, agg_error = agg_func(measurement_subset)
        agg_measurement_df.loc[i] = [array, agg_value, agg_error]
        i += 1
    return agg_measurement_df


def weigh_by_error(measurement_df):
    relative_error = measurement_df['error'] / measurement_df['value']
    weights = relative_error**(-2)
    normalized_weights = weights / weights.sum()
    weighted_average_value = (
        normalized_weights * measurement_df['value']
    ).sum()
    weighted_average_error = (
        (normalized_weights * measurement_df['error'])**2
    ).sum()**.5
    return weighted_average_value, weighted_average_error


def main():
    complex_df =\
        pd.read_csv('../newdata/experimental/complex.csv', index_col=0)
    measurement_df =\
        pd.read_csv('../newdata/experimental/measurement.csv', index_col=0)

    mm_dict = get_mismatch_map_dict(complex_df)

    B = aggregate_measurements(
        measurement_df,
        mm_dict,
        weigh_by_error
    )
    pass


if __name__ == '__main__':
    main()
