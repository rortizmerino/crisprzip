"""
The parameter vectors in this method are different representations
of searchers, corresponding to different ways to optimize the model.

Classes:
    ParameterVector(ABC)
    EslamiParams(ParameterVector)
    CumulativeParams(EslamiParams)
    FixedRatesParams(EslamiParams)
    FreeBindingParams(ParameterVector)
    FreeBindingFixedRates(ParameterVector)
    FreeBindingFixedCleavageParams(ParameterVector)
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from numpy.random import Generator, default_rng

from crisprzipper.model.data import ExperimentType
from crisprzipper.model.hybridization_kinetics import Searcher


class ParameterVector(ABC):
    """An Abstract Base Class (should not be instantiated) setting a few
    common features among all the different parameter vectors.

    Attributes
    ----------
    x: np.array
        The parameter values
    x0: List[float]
        The initial value for optimization
    lb: List[float]
        The lower bound for the parameters during optimization
    ub: List[float]
        The upper bound for the parameters during optimization
    param_names: List[str]
        Names of the parameters

    Methods
    -------
    make_random()
        Alternative constructor, draws parameter values from uniform
        distribution
    get_info()
        Return param_names, x0, lb and ub as a dictionary
    to_searcher()
        Returns the corresponding Searcher
    to_binding_rate(exptype)
        Returns the binding rate of the specified experiment
    """

    x0 = []  # initial value
    lb = []  # lower bound
    ub = []  # upper bound
    param_names = []

    def __init__(self, x: np.typing.ArrayLike):
        self.x = np.asarray(x)
        self.__check_dimensionality()
        self.__check_bounds()

    @classmethod
    def get_x0(cls):
        """Returns the initial condition as a ParameterVector instance"""
        return cls(cls.x0)

    @classmethod
    def get_lb(cls):
        """Returns the lower bounds as a ParameterVector instance"""
        return cls(cls.lb)

    @classmethod
    def get_ub(cls):
        """Returns the upper bounds as a ParameterVector instance"""
        return cls(cls.ub)

    def __check_dimensionality(self):
        if self.x0 and self.x.shape != (len(self.x0),):
            raise ValueError(f'Parameter vector should be of'
                             f'shape ({len(self.x0)}, )')
        elif self.x.ndim != 1:
            raise ValueError('Parameter vector should be 1-dimensional')

    def __check_bounds(self):
        if self.lb and self.ub:
            if not (np.all(self.x >= self.lb) and
                    np.all(self.x <= self.ub)):
                raise ValueError('Parameter values lie outside of'
                                 'upper and/or lower bounds.')

    @classmethod
    def make_random(cls, rng: Union[int, Generator] = None):
        if type(rng) is int or rng is None:
            rng = default_rng(rng)
        lb = np.asarray(cls.lb)
        ub = np.asarray(cls.ub)
        return cls(lb + (ub - lb) * rng.random(size=lb.size))

    def get_info(self):
        return {'param_names': self.param_names,
                'x0': self.x0,
                'lb': self.lb,
                'ub': self.ub}

    @abstractmethod
    def to_searcher(self):
        pass

    @abstractmethod
    def to_binding_rate(self, exptype: ExperimentType):
        pass


class EslamiParams(ParameterVector):
    """
    Original definition, with fixed k_on.
    -  0-20:  PAM energy + on-target landscape [kBT]
    - 21-40: mismatch penalties [kBT]
    -    41: log10(k_on) [Hz]
    -    42: log10(k_f) [Hz]
    -    43: log10(k_clv) [Hz]
    """

    x0 = 21 * [0.] + 20 * [0.] + 3 * [0.]
    lb = 21 * [-10] + 20 * [0] + 3 * [-6]
    ub = 21 * [20] + 20 * [20] + 3 * [6]
    param_names = (["U_PAM"] + [f"U_{i:02d}" for i in range(1, 21)] +
                   [f"Q_{i:02d}" for i in range(1, 21)] +
                   ["log10(k_on)", "log10(k_f)", "log10(k_clv)"])

    def __init__(self, x, do_nuseq_correction=False):
        # do_nuseq_correction corrects for the fact that previous
        # solutions were trained at 1E6 nM NucleaSeq concentration,
        # whereas they now get simulated at 100 nM.
        self.do_nuseq_correction = do_nuseq_correction
        super().__init__(x)

    def to_searcher(self):
        return Searcher(
            on_target_landscape=self.x[1:21],
            mismatch_penalties=self.x[21:41],
            internal_rates={
                # k_off = k_on * exp(U_pam), detailed balance
                'k_off': 10.**self.x[41] * np.exp(self.x[0]),
                'k_f': 10.**self.x[42],
                'k_clv': 10.**self.x[43]
            }
        )

    def to_binding_rate(self, exptype: ExperimentType,
                        do_nuseq_correction=False):
        # Doing proper enum comparison is problematic in Jupyter
        # Notebook, so the uncommented code is a workaround
        # if not type(exptype) is ExperimentType:
        #     raise ValueError("Experiment type is not recognized")

        if self.do_nuseq_correction and exptype.name == "NUCLEASEQ":
            return 1E4 * 10.**self.x[-3]
        else:
            return 10.**self.x[-3]


class CumulativeParams(EslamiParams):
    """Like the Eslami searcher, with a cumulative on-target landscape"""

    x0 = 21 * [0.] + 20 * [0.] + 3 * [0.]
    lb = 21 * [-10] + 20 * [0] + 3 * [-6]
    ub = 21 * [10] + 20 * [20] + 3 * [6]
    param_names = (["U_PAM"] + [f"dU_{i:02d}" for i in range(1, 21)] +
                   [f"Q_{i:02d}" for i in range(1, 21)] +
                   ["log10(k_on)", "log10(k_f)", "log10(k_clv)"])

    def to_searcher(self):
        return Searcher(
            on_target_landscape=np.cumsum(self.x[1:21]),
            mismatch_penalties=self.x[21:41],
            internal_rates={
                # k_off = k_on * exp(U_pam), detailed balance
                'k_off': 10.**self.x[41] * np.exp(self.x[0]),
                'k_f': 10.**self.x[42],
                'k_clv': 10.**self.x[43]
            }
        )


class FixedRatesParams(EslamiParams):
    """Like EslamiParams, with k_f and k_clv fixed."""

    x0 = 21 * [0.] + 20 * [0.] + [0.]
    lb = 21 * [-10] + 20 * [0] + [-6]
    ub = 21 * [20] + 20 * [20] + [6]
    param_names = (["U_PAM"] + [f"U_{i:02d}" for i in range(1, 21)] +
                   [f"Q_{i:02d}" for i in range(1, 21)] +
                   ["log10(k_on)"])
    log_k_fwd = 4.
    log_k_clv = 3.

    def to_searcher(self):
        return Searcher(
            on_target_landscape=self.x[1:21],
            mismatch_penalties=self.x[21:41],
            internal_rates={
                # k_off = k_on * exp(U_pam), detailed balance
                'k_off': 10.**self.x[41] * np.exp(self.x[0]),
                'k_f': 10.**self.log_k_fwd,
                'k_clv': 10.**self.log_k_clv
            }
        )


class FreeBindingParams(ParameterVector):
    """Experiment-dependent k_on, k_off fixed.
    -  0-19:  on-target landscape [kBT]
    - 20-39: mismatch penalties [kBT]
    -    40: log10(k_off) [Hz]
    -    41: log10(k_f) [Hz]
    -    42: log10(k_clv) [Hz]
    -    43: log10(k_on_Nucleaseq) [Hz]
    -    44: log10(k_on_Champ) [Hz]
    """

    x0 = 20 * [0.] + 20 * [0.] + 5 * [0.]
    lb = 20 * [-10] + 20 * [0] + 5 * [-6]
    ub = 20 * [20] + 20 * [20] + 5 * [6]
    param_names = ([f"U_{i:02d}" for i in range(1, 21)] +
                   [f"Q_{i:02d}" for i in range(1, 21)] +
                   ["log10(k_off)", "log10(k_f)", "log10(k_clv)",
                    "log10(k_on_NuSeq)", "log10(k_on_Champ)"])

    def to_searcher(self):
        return Searcher(
            on_target_landscape=self.x[0:20],
            mismatch_penalties=self.x[20:40],
            internal_rates={
                'k_off': 10. ** self.x[40],
                'k_f': 10. ** self.x[41],
                'k_clv': 10. ** self.x[42]
            }
        )

    def to_binding_rate(self, exptype: ExperimentType):
        # Doing proper enum comparison is problematic in Jupyter
        # Notebook, so the uncommented code is a workaround
        # if not type(exptype) == ExperimentType:
        #     raise ValueError("ExperimentType could not be recognized.")
        if exptype.value == ExperimentType.NUCLEASEQ.value:
            return 10. ** self.x[43]
        elif exptype.value == ExperimentType.CHAMP.value:
            return 10. ** self.x[44]
        else:
            return 1.


class FreeBindingFixedRatesParams(ParameterVector):
    """Like FreeBindingParams, with k_f and k_clv fixed."""

    x0 = 20 * [0.] + 20 * [0.] + 3 * [0.]
    lb = 20 * [-10] + 20 * [0] + 3 * [-6]
    ub = 20 * [20] + 20 * [20] + 3 * [6]
    param_names = ([f"U_{i:02d}" for i in range(1, 21)] +
                   [f"Q_{i:02d}" for i in range(1, 21)] +
                   ["log10(k_off)", "log10(k_on_NuSeq)", "log10(k_on_Champ)"])
    log_k_fwd = 4.
    log_k_clv = 3.

    def to_searcher(self):
        return Searcher(
            on_target_landscape=self.x[0:20],
            mismatch_penalties=self.x[20:40],
            internal_rates={
                'k_off': 10. ** self.x[40],
                'k_f': 10. ** self.log_k_fwd,
                'k_clv': 10. ** self.log_k_clv
            }
        )

    def to_binding_rate(self, exptype: ExperimentType):
        # Doing proper enum comparison is problematic in Jupyter
        # Notebook, so the uncommented code is a workaround
        # if not type(exptype) == ExperimentType:
        #     raise ValueError("ExperimentType could not be recognized.")
        if exptype.value == ExperimentType.NUCLEASEQ.value:
            return 10. ** self.x[41]
        elif exptype.value == ExperimentType.CHAMP.value:
            return 10. ** self.x[42]
        else:
            return 1.


class FreeBindingFixedCleavageParams(ParameterVector):
    """Like FreeBindingParams, with only k_clv fixed."""
    param_names = ([f"U_{i:02d}" for i in range(1, 21)] +
                   [f"Q_{i:02d}" for i in range(1, 21)] +
                   ["log10(k_off)", "log10(k_f)",
                    "log10(k_on_NuSeq)", "log10(k_on_Champ)"])

    x0 = 20 * [0.] + 20 * [0.] + 4 * [0.]
    lb = 20 * [-10] + 20 * [0] + 4 * [-6]
    ub = 20 * [20] + 20 * [20] + 4 * [6]
    log_k_clv = 3.

    def to_searcher(self):
        return Searcher(
            on_target_landscape=self.x[0:20],
            mismatch_penalties=self.x[20:40],
            internal_rates={
                'k_off': 10. ** self.x[40],
                'k_f': 10. ** self.x[41],
                'k_clv': 10. ** self.log_k_clv
            }
        )

    def to_binding_rate(self, exptype: ExperimentType):
        # Doing proper enum comparison is problematic in Jupyter
        # Notebook, so the uncommented code is a workaround
        # if not type(exptype) == ExperimentType:
        #     raise ValueError("ExperimentType could not be recognized.")
        if exptype.value == ExperimentType.NUCLEASEQ.value:
            return 10. ** self.x[42]
        elif exptype.value == ExperimentType.CHAMP.value:
            return 10. ** self.x[43]
        else:
            return 1.
