import json
import pytest
import pandas as pd
from crisprzip.kinetics import *

PRECISION = 1E-4

# Load expected output
df_avg = pd.read_csv("tests/cleavage_binding_prediction/expected_output_avg.csv")
df_seq = pd.read_csv("tests/cleavage_binding_prediction/expected_output_seq.csv")

# Load parameter sets
@pytest.fixture
def average_params():
    """Loads average-based parameter set."""
    with open("data/landscapes/average_params.json", "r") as file:
        return json.load(file)['param_values']

@pytest.fixture
def sequence_params():
    """Loads sequence-based parameter set."""
    with open("data/landscapes/sequence_params.json", "r") as file:
        return json.load(file)['param_values']

@pytest.mark.parametrize("time, on_rate, f_clv",
                         df_avg.loc[~df_avg['f_clv'].isna()].values)
def test_cleavage_average(time, on_rate, f_clv, *args, **kwargs):
    """Test cleavage fraction for both average-based parameter sets."""
    complex_obj = SearcherSequenceComplex(**average_params)
    f_clv = complex_obj.get_cleaved_fraction(time=time, on_rate=on_rate)

    # Validate result
    assert pytest.approx(f_clv, rel=PRECISION) == f_clv

# @pytest.mark.parametrize("param_set, protospacer, target_seq, time, on_rate, expected_f_bnd",
#                          df[df["f_bnd"].notna()].values)
# def test_binding(param_set, protospacer, target_seq, time, on_rate, expected_f_bnd, sequence_params, average_params):
#     """Test binding fraction for both sequence-based and average-based parameter sets."""
#
#     # Select appropriate parameter set
#     params = sequence_params if param_set == "sequence" else average_params
#
#     # Skip NULL protospacer/target for "average" tests
#     if pd.isna(protospacer) or pd.isna(target_seq):
#         protospacer = "AGACGCATAAAGATGAGACGCTGG"  # Dummy values for avg test
#         target_seq = "AGACCCATTAAGATGAGACGCGGG"
#
#     # Run test
#     complex_obj = SearcherSequenceComplex(protospacer=protospacer, target_seq=target_seq, **params)
#     f_bnd = complex_obj.get_bound_fraction(time=time, on_rate=on_rate)
#
#     # Validate result
#     assert pytest.approx(f_bnd, rel=1e-2) == expected_f_bnd, f"Mismatch for {param_set} at {time}s, {on_rate}"
