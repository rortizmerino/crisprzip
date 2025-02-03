import json
import pytest
import pandas as pd
from crisprzip.kinetics import *

PRECISION = 1E-4

# Load expected output
DATA_AVG = pd.read_csv("tests/cleavage_binding_prediction/expected_output_avg.csv",
                     dtype={'mm_pattern': str})
DATA_SEQ = pd.read_csv("tests/cleavage_binding_prediction/expected_output_seq.csv")


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


@pytest.mark.parametrize("mm_pattern, exptype, k_on, time, fraction", DATA_AVG.values)
def test_average_model(mm_pattern, exptype, k_on, time, fraction, average_params):
    mm_pattern = MismatchPattern.from_string(mm_pattern)
    complex_obj = SearcherTargetComplex(
        target_mismatches=mm_pattern,
        **average_params
    )
    if exptype == 'cleave':
        f_clv = complex_obj.get_cleaved_fraction(time=time, on_rate=k_on)
        assert f_clv == pytest.approx(fraction, abs=PRECISION)
    if exptype == 'bind':
        f_clv = complex_obj.get_bound_fraction(time=time, on_rate=k_on)
        assert f_clv == pytest.approx(fraction, abs=PRECISION)


@pytest.mark.parametrize("protospacer, targetseq, exptype, k_on, time, fraction", DATA_SEQ.values)
def test_sequence_model(protospacer, targetseq, exptype, k_on, time, fraction, sequence_params):
    complex_obj = SearcherSequenceComplex(
        protospacer=protospacer,
        target_seq=targetseq,
        **sequence_params
    )
    if exptype == 'cleave':
        f_clv = complex_obj.get_cleaved_fraction(time=time, on_rate=k_on)
        assert f_clv == pytest.approx(fraction, abs=PRECISION)
    if exptype == 'bind':
        f_clv = complex_obj.get_bound_fraction(time=time, on_rate=k_on)
        assert f_clv == pytest.approx(fraction, abs=PRECISION)
