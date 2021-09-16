import numpy as np
from scipy import linalg
from nucleicacid import NucleicAcid, DnaCode, RnaCode


class NAGSearcher:
    """
    Represents a nucleic acid guided searcher complex, i.e. a searcher
    protein carrying a generic (DNA/RNA) guide.
    """

    def __init__(self, guide: NucleicAcid, guide_direction=None,
                 pam_detection=False):
        if guide_direction is not None:
            if guide.direction == 0:
                guide.direction = guide_direction
            else:
                guide = guide.enforce_direction(guide_direction)
        self.guide = guide
        self.guide_length = guide.length
        self.pam_detection = pam_detection


class HybridizationLandscape:
    def __init__(self,
                 on_target_landscape: np.ndarray,
                 mismatch_penalties: np.ndarray,
                 forward_rates: dict,
                 pam_detection=True, catalytic_dead=False):

        # check whether parameters are 1d arrays
        if on_target_landscape.ndim > 1 or mismatch_penalties.ndim > 1:
            raise ValueError('Landscape parameters must be 1d arrays')

        # check whether landscape dimensions agree with guide length
        guide_length = mismatch_penalties.size
        if on_target_landscape.size != pam_detection + guide_length:
            raise ValueError('Landscape dimensions do not match guide length')

        # check whether forward_rates dictionary contains proper keys
        if not ('k_on' in forward_rates and
                'k_f' in forward_rates and
                'k_clv' in forward_rates):
            raise ValueError('Forward rates dictionary should include k_on, '
                             'k_f and k_clv as keys')

        # check whether cleavage rate agrees with catalytic status
        if (forward_rates['k_clv'] == 0) != catalytic_dead:
            raise ValueError('Cleavage rate does not corresponds to '
                             'catalytic activity')

        # assign values
        self.guide_length = guide_length
        self.pam_detection = pam_detection
        self.catalytic_dead = catalytic_dead

        self.on_target_landscape = on_target_landscape
        self.mismatch_penalties = mismatch_penalties
        self.forward_rate_dict = forward_rates
        self.forward_rate_array = self.get_forward_rate_array()

    def get_off_target_landscape(self, target_mismatches: np.ndarray):

        # check dimensions of mismatch position array
        if target_mismatches.size != self.guide_length:
            raise ValueError('Target array should be of same length as guide')

        landscape_penalties = np.concatenate(
            (np.zeros(int(self.pam_detection)),  # add preceding zero for PAM
             np.cumsum(target_mismatches * self.mismatch_penalties))
        )
        return self.on_target_landscape + landscape_penalties

    def get_landscape_diff(self, target_mismatches: np.ndarray):
        hybrid_landscape = self.get_off_target_landscape(target_mismatches)
        return np.diff(hybrid_landscape, prepend=np.zeros(1))

    def get_forward_rate_array(self):
        forward_rate_array = np.concatenate(
            (self.forward_rate_dict['k_on'] * np.ones(1),
             self.forward_rate_dict['k_f'] *
                np.ones(self.on_target_landscape.size - 1),
             self.forward_rate_dict['k_clv'] * np.ones(1),
             np.zeros(1))
        )
        return forward_rate_array

    def get_backward_rate_array(self, target_mismatches: np.ndarray):
        boltzmann_factors = np.exp(self.get_landscape_diff(target_mismatches))
        backward_rate_array = np.concatenate(
            (np.zeros(1),
             self.forward_rate_array[:-2] * boltzmann_factors,
             np.zeros(1))
        )
        return backward_rate_array

    def get_rate_matrix(self, target_mismatches: np.ndarray) -> np.ndarray:

        backward_rates = self.get_backward_rate_array(target_mismatches)
        forward_rates = self.forward_rate_array

        # overwrite on-rate such that solution state is probability sink
        forward_rates[0] = 0

        diagonal1 = -(forward_rates + backward_rates)
        diagonal2 = backward_rates[1:]
        diagonal3 = forward_rates[:-1]

        rate_matrix = (np.diag(diagonal1, k=0) +
                       np.diag(diagonal2, k=1) +
                       np.diag(diagonal3, k=-1))
        return rate_matrix

    def solve_master_equation(self, target_mismatches: np.ndarray,
                              initial_condition: np.ndarray,
                              time: float) -> np.ndarray:

        # check dimensions initial condition
        if initial_condition.size != (2+self.guide_length+self.pam_detection):
            raise ValueError('Initial condition should be of same length as'
                             'hybridization landscape')

        rate_matrix = self.get_rate_matrix(target_mismatches)
        matrix_exponent = linalg.expm(rate_matrix * time)
        return matrix_exponent.dot(initial_condition)
