"""
The data module handles experimental data, reading it as well as
processing it.

Classes:
    ExperimentType(Enum)
    ExperimentalData(ABC)
    SequenceData(ExperimentalData)
    AggregateData(ExperimentalData)
"""

from abc import ABC
from enum import Enum, auto
from math import factorial
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
import pandas as pd

from crisprzipper.model.hybridization_kinetics import MismatchPattern
from crisprzipper.model.tools import path_handling


class ExperimentType(Enum):
    """Lists all the types of experimental data that could be used."""
    NUCLEASEQ = auto()
    CHAMP = auto()


class ExperimentalData(ABC):
    """The ExperimentalData Abstract Base Class contains some fields and
    methods that are relevant to all the different Data Classes. As it
    is an ABC, it should never be instanciated.

    Attributes
    ----------
    path: Path
        Path to the datafile
    exp_type: ExperimentType
        Type of experiment to which the data corresponds
    data: pd.DataFrame
        Contains the datavalues. Some daughter data classes require
        particular columns to be included in this dataframe.
    columns: List[str]
        Required column names in the data attribute

    Methods
    -------
    from_csv(path, exp_type)
        Alternative constructor that makes use of pd.read_csv().
    exclude_outliers(exp_type, data, value_limits=None, error_limits=None)
        Can filter the contents of the data to remove values outside a
        designated range
    get_mm_patterns()
        Takes the mismatch patterns that are stored as strings in one
        of the columns of the 'data' dataframe, and returns them as a
        list of MismatchPattern objects
    weigh_errors(df, relative=True, normalize=True)
        Calculates weights for data based on the squared (relative)
        error
    calc_weighted_average(df, weights)
        Averages the data and errors from a dataframe based on a series
        of weights
    """

    columns = []

    def __init__(self, path: Union[str, Path], data: pd.DataFrame,
                 exp_type: ExperimentType):
        self.path = Path(path)
        self.exp_type = exp_type
        self.data = self.exclude_outliers(exp_type, data)
        self.__check_columns()

    @classmethod
    @path_handling
    def from_csv(cls, path: Union[str, Path],
                 exp_type: ExperimentType) -> 'ExperimentalData':
        """Alternative constructor that makes use of pd.read_csv()"""
        data = pd.read_csv(path, dtype={'mismatch_array': str}, index_col=0)
        return cls(path, data, exp_type)

    @staticmethod
    def exclude_outliers(exp_type: ExperimentType, data: pd.DataFrame,
                         value_limits: Tuple[float] = None,
                         error_limits: Tuple[float] = None) -> pd.DataFrame:
        """Can filter the contents of the data to remove values outside
        a designated range. By default, there are some standard values
        that work for the raw 2020 datasets left by Misha."""
        if exp_type.name == "NUCLEASEQ" and value_limits is None:
            value_limits = (1E-10, np.inf)
        if exp_type.name == "NUCLEASEQ" and error_limits is None:
            error_limits = (0, np.inf)
        if exp_type.name == "CHAMP" and value_limits is None:
            value_limits = (0, np.inf)
        if exp_type.name == "CHAMP" and error_limits is None:
            error_limits = (1E-10, np.inf)

        return data.loc[(data.value >= value_limits[0]) &
                        (data.value <= value_limits[1]) &
                        (data.error >= error_limits[0]) &
                        (data.error <= value_limits[1])]

    def __check_columns(self):
        """In a daughter class which has required columns, controls whether
        these are present in the data attribute"""
        return all(col in self.data.columns for col in self.columns)

    def get_mm_patterns(self) -> List[MismatchPattern]:
        """Takes the mismatch patterns that are stored as strings in one
        of the columns of the 'data' dataframe, and returns them as a
        list of MismatchPattern objects"""
        return list(map(MismatchPattern.from_string,
                        self.data.mismatch_array.to_list()))

    @staticmethod
    def weigh_errors(df: pd.DataFrame, relative=True,
                     normalize=True) -> pd.Series:
        """Calculates weights for data based on the squared (relative)
        error"""
        weights = pd.Series(index=df.index,
                            data=(1 / df.error) ** 2)
        if relative:
            weights = weights * (df.value ** 2)
        if normalize:
            weights = weights / weights.sum()
        return weights.rename("weight")

    @staticmethod
    def calc_weighted_average(df: pd.DataFrame,
                              weights: pd.Series) -> Tuple[float, float]:
        """Averages the data and errors from a dataframe based on a
        series of weights"""
        weights = weights / weights.sum()  # normalize weights
        avg_value = (df.value * weights).sum()
        avg_error = (df.error * weights).sum()
        return avg_value, avg_error


class SequenceData(ExperimentalData):
    """Sequence data contains experimental data (and errors) on the level
    of protospacer sequences. These can unique sequences or not.

    Attributes
    ----------
    unique_sequences: bool
        Indicates whether each sequence is associated with a unique
        experimental value

    Methods
    -------
    aggregate_sequences()
        For non-unique datasets, constructs a SequenceData object that
        does have unique entries by calculating (weighted) average
        values and errors for each sequence.
    aggregate_mismatch_patterns()
        Constructs a AggregateData object by calculating weighted
        average values and errors for each mismatch pattern.
    """

    columns = ["canonical PAM", "protospacer_seq", "pam_seq", "target_seq",
               "mismatch_array", "value", "error"]

    def __init__(self, path: Union[str, Path], data: pd.DataFrame,
                 exp_type: ExperimentType):
        super().__init__(path, data, exp_type)
        self.unique_sequences = self.__check_unique_sequences()

    def __check_unique_sequences(self) -> bool:
        """Sets unique_sequences attribute"""
        return (self.data.shape[0] ==
                (self.data.pam_seq.str
                 .cat(self.data.target_seq).unique()).shape[0])

    def aggregate_sequences(self) -> 'SequenceData':
        """For non-unique datasets, constructs a SequenceData object that
        does have unique entries by calculating (weighted) average
        values and errors for each sequence."""
        weights = self.weigh_errors(self.data, relative=True, normalize=True)
        aggr_df = pd.DataFrame(columns=self.data.columns)

        i = 0
        target_seqs = self.data.target_seq.unique()
        pam_seqs = self.data.pam_seq.unique()
        for target in target_seqs:
            for pam in pam_seqs:
                selection = ((self.data.pam_seq == pam) &
                             (self.data.target_seq == target))
                if selection.sum() == 0:
                    continue
                aggr_val, aggr_err = self.calc_weighted_average(
                    self.data.loc[selection], weights[selection]
                )
                aggr_df.loc[i] = self.data.loc[selection].iloc[0]
                aggr_df.loc[i, 'value'] = aggr_val
                aggr_df.loc[i, 'error'] = aggr_err
                i += 1

        return SequenceData(self.path, aggr_df, self.exp_type)

    def aggregate_mismatch_patterns(self) -> 'AggregateData':
        """Constructs a AggregateData object by calculating weighted
        average values and errors for each mismatch pattern."""

        if not self.unique_sequences:
            obj = self.aggregate_sequences()
        else:
            obj = self

        weights = obj.weigh_errors(obj.data, relative=True, normalize=True)
        aggr_df = pd.DataFrame(columns=obj.data.columns)

        i = 0
        mm_patterns = obj.data.mismatch_array.unique()
        for mm_pattern in mm_patterns:
            selection = obj.data.mismatch_array == mm_pattern
            aggr_val, aggr_err = obj.calc_weighted_average(
                obj.data.loc[selection], weights[selection]
            )
            aggr_df.loc[i] = obj.data.loc[selection].iloc[0]
            aggr_df.loc[i, 'value'] = aggr_val
            aggr_df.loc[i, 'error'] = aggr_err
            i += 1

        aggr_df = aggr_df[['canonical PAM', 'protospacer_seq',
                           'mismatch_array', 'mismatch_number',
                           'value', 'error']]
        return AggregateData(self.path, aggr_df, self.exp_type)


class AggregateData(ExperimentalData):
    """Sequence data contains experimental data (and errors) on the level
    of mismatch patterns.

    Methods
    -------
    get_mm_nums()
        Calculates the number of mismatches for each mismatch pattern
        in the data attribute, making use of the MismatchPattern
        methods.
    to_mm_num_subset(mm_num)
        Constructs a AggregateData object containing only the entries
        with a specified number of mismatch numbers.
    weight_multiplicity(mm_patterns, normalize=True)
        Calculates weights for data based on the multiplicity of the
        mismatch array
    """

    columns = ["mismatch_array", "value", "error"]

    def get_mm_nums(self):
        """Calculates the number of mismatches for each mismatch pattern
        in the data attribute, making use of the MismatchPattern
        methods. This way, this object no longer depends on the
        existence of a data column containing mismatch numbers a
        priori."""
        def mm_num_from_str(mm_string):
            return MismatchPattern.from_string(mm_string).mm_num
        return (self.data['mismatch_array'].apply(mm_num_from_str)
                .rename("mismatch_number"))

    def to_mm_num_subset(self, mm_num) -> 'AggregateData':
        return AggregateData(
            self.path,
            self.data.loc[self.get_mm_nums() == mm_num],
            self.exp_type
        )

    @staticmethod
    def _calc_multiplicity(mm_num: int, guide_length: int):
        """Multiplicity, given by the binomial factor N! / n! (N-n)!"""
        return int(factorial(guide_length) *
                   1 / factorial(guide_length - mm_num) *
                   1 / factorial(mm_num))

    @classmethod
    def weigh_multiplicity(cls, mm_patterns: pd.Series,
                           normalize=True) -> pd.Series:
        """Calculates weights for data based on the multiplicity of the
        mismatch array"""

        def pattern_multiplicity(mm_string: str):
            pattern = MismatchPattern.from_string(mm_string)
            return cls._calc_multiplicity(pattern.mm_num, pattern.length)

        weights = pd.Series(
            index=mm_patterns.index,
            data=(1 / mm_patterns.apply(pattern_multiplicity))
        )
        if normalize:
            weights = weights / weights.sum()
        return weights.rename('weights')
