import numpy as np

def get_mismatched_target(on_target,mismatch_positions,Cas):
    on_target_seq = np.array(list(on_target))
    off_target = on_target_seq
    for mm in mismatch_positions:
        off_target[Cas.guide_length-mm] = 'X'
    off_target = ''.join(off_target)
    return off_target