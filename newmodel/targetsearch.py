import numpy as np
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
                 forward_rates: np.ndarray,
                 pam_detection=True, catalytic_dead=False):

        # check whether parameters are 1d arrays
        if (on_target_landscape.ndim > 1 or
                mismatch_penalties.ndim > 1 or
                forward_rates.ndim > 1):
            raise ValueError('Landscape parameters must be 1d arrays')

        # check whether landscape dimensions agree with guide length
        guide_length = mismatch_penalties.size
        if (on_target_landscape.size != pam_detection + guide_length or
                forward_rates.size != on_target_landscape.size +
                1 - catalytic_dead):
            raise ValueError('Landscape dimensions do not match guide length')

        self.guide_length = on_target_landscape.size
        self.on_target_landscape = on_target_landscape
        self.mismatch_penalties = mismatch_penalties
        self.forward_rates = forward_rates
        self.pam_detection = pam_detection
        self.catalytic_dead = catalytic_dead

    def get_off_target_landscape(self, target_mismatches: np.ndarray):

        if target_mismatches.size != self.mismatch_penalties.size:
            raise ValueError('Target array should be of same length as guide')

        landscape_penalties = np.concatenate(
            np.zeros(int(self.pam_detection)),  # add preceding zero for PAM
            target_mismatches * self.mismatch_penalties
        )
        return self.on_target_landscape + landscape_penalties

    def get_landscape_diff(self, target_mismatches: np.ndarray):
        hybrid_landscape = self.get_off_target_landscape(target_mismatches)
        landscape_diff = np.zeros(self.on_target_landscape.size)
        landscape_diff[1:] += np.diff( hybrid_landscape )
        return landscape_diff


