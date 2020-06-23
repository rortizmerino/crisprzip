import numpy as np

from CRISPRclass import CRISPR


################################################################################
#  mimicking smFRET experiments (Dagdas et al. / Yang et al. ) that labelled Cas9 to
#  detect movement of HNH-domain during target binding
#
#
################################################################################


def Gillespie_simple(forward_rates, backward_rates, Cas):
    '''
    use parameters from microscopic model to generate a trajectory through landscape
    stops when max_time has been reached or when cleavage occurs
    :param forward_rates:
    :param backward_rates:
    :param Cas:
    :return:
    '''
    max_time = 60.
    time = [0.0]
    state = [-1]  # start in solution
    while time[-1] <= max_time:
        # --- outgoing rates from current state ----
        kf = forward_rates[state[-1] + 1]
        kb = backward_rates[state[-1] + 1]

        # --- add time ----
        lifetime = (kf + kb) ** (-1)
        delta_t = np.random.exponential(scale=lifetime)
        time.append(time[-1] + delta_t)

        # --- splitting prob ---
        prob_fwd = kf / (kf + kb)
        prob_bck = kb / (kf + kb)

        # --- determine events ---
        U = np.random.uniform()
        if U < prob_fwd:
            state.append(state[-1] + 1)
        else:
            state.append(state[-1] - 1)

        # --- stop if you hit post-cleavage state ---
        if state[-1] > Cas.guide_length:
            break
    return state, time


def Gillespie_time_resolution(forward_rates, backward_rates, time_resolution, Cas):
    '''
    use parameters from microscopic model to generate a trajectory through landscape
    stops when max_time has been reached or when cleavage occurs.

    produces a trace sampled at finite time resolution.
    :param forward_rates:
    :param backward_rates:
    :param time_resolution:
    :param Cas:
    :return:
    '''
    max_time = 60.
    time = np.array([n * time_resolution for n in range(0, int(max_time / time_resolution) + 1)])
    state = np.array([Cas.guide_length + 1] * len(time))
    #     state[0] = -1 #start in solution

    start_event = 0
    current_time = 0.0
    current_state = -1
    while current_time <= max_time:
        # --- outgoing rates from current state ----
        kf = forward_rates[current_state + 1]
        kb = backward_rates[current_state + 1]

        # --- add time ----
        lifetime = (kf + kb) ** (-1)
        current_time += np.random.exponential(scale=lifetime)

        # --- determine trace at finite time resultion ----
        end_event = np.minimum(int(np.floor(current_time / time_resolution)), len(time) - 1)
        state[start_event:(end_event + 1)] = current_state
        start_event = end_event + 1

        # --- splitting prob ---
        prob_fwd = kf / (kf + kb)
        prob_bck = kb / (kf + kb)

        # --- determine events ---
        U = np.random.uniform()
        if U < prob_fwd:
            current_state += 1
        else:
            current_state -= 1

        # --- stop if you hit post-cleavage state ---
        if current_state > Cas.guide_length:
            break
    return state, time



def HNH_trace_hard_boundaries(trace, state_dict={"O": 0.5, "I":12, "C":19.}):

    '''
    uses fixed translation of every microscopic state to coarse-grained state
    :param trace:
    :param state_dict:
    :return:
    '''
    # ---- coarse-grained states ---
    # state 0: Open state (RNA-only state)
    # state 1: Intermediate state
    # state 2: Closed state (Displaced state)
    HNHstate = []
    for state in trace:
        if (state <0) or (state>20):
            HNHstate.append(state)
        elif state < 8:
            HNHstate.append(state_dict["O"])
        elif state <  20:
            HNHstate.append(state_dict["I"])
        else:
            HNHstate.append(state_dict["C"])
    return np.array(HNHstate)


def HNH_trace_conditional_boundaries(trace, state_dict={"O": 0.5, "I": 12, "C": 19., "sol": -1, "clv": 21}):
    '''
    Translation between microscopic states and "HNH-domain-configurations".
    Let the readout depend on latests readout, i.e.: Stay in state "I" until you cross barrier to "O", likewise stay in "O"
    until you cross barrier to "I".
    Hence, microscopic states in "interviening barriers" will be classified based on the previously
    recorded coarse-grained state
    :param trace:
    :param state_dict:
    :return:
    '''
    # ---- coarse-grained states ---
    # state 0: Open state (RNA-only state)
    # state 1: Intermediate state
    # state 2: Closed state (Displaced state)
    HNHstate = []
    for i, state in enumerate(trace):
        # --- solution state and post-cleavage state ----
        if state == state_dict['sol']:
            HNHstate.append(state_dict['sol'])
        elif state == state_dict['clv']:
            HNHstate.append(state_dict['clv'])

        # --- currently in state "O" ----
        elif HNHstate[-1] == state_dict["O"]:
            if state != 12:
                HNHstate.append(state_dict["O"])
            else:
                HNHstate.append(state_dict["I"])


        # --- currently in state "I" ---
        elif HNHstate[-1] == state_dict["I"]:
            if (state > 0) and (state < 20):
                HNHstate.append(state_dict["I"])
            elif state == 20:
                HNHstate.append(state_dict["C"])
            else:
                HNHstate.append(state_dict["O"])


        # --- currently in state "C" ---
        elif HNHstate[-1] == state_dict["C"]:
            if state != 12:
                HNHstate.append(state_dict["C"])
            else:
                HNHstate.append(state_dict["I"])

        # -- if non of the above, you must have come from solution ---
        else:
            HNHstate.append(state_dict["O"])
    return np.array(HNHstate)


def analyse_trace(trace, time, state_dict={"O": 0.5, "I": 12, "C": 19., "sol": -1, "clv": 21}):
    '''
    Determine events, their duration and "mock FRET change".
    :param trace: coarse-grained trace
    :param time:
    :param state_dict:
    :return:
    '''

    # --- possible transitions / mock FRET changes ---
    lvl_to_state = {}
    lvl_to_state[state_dict["O"] - state_dict["sol"]] = "bind"
    lvl_to_state[state_dict["I"] - state_dict["O"]] = "O to I"
    lvl_to_state[state_dict["C"] - state_dict["I"]] = "I to C"
    lvl_to_state[state_dict["clv"] - state_dict["C"]] = "cleave"
    lvl_to_state[state_dict["I"] - state_dict["C"]] = "C to I"
    lvl_to_state[state_dict["O"] - state_dict["I"]] = "I to O"
    lvl_to_state[state_dict["sol"] - state_dict["O"]] = "unbind"

    # --- output arrays / lists ---
    event = []
    time_event = []
    lvl_before = []
    lvl_after = []

    # --- previous transition (store index) ---
    start_event = 0
    for n in range(len(trace) - 1):
        lvl_change = trace[n + 1] - trace[n]
        if lvl_change != 0:
            try:
                event.append(lvl_to_state[lvl_change])
                time_event.append(time[n] - time[start_event])
                start_event = n

                lvl_before.append(trace[n])
                lvl_after.append(trace[n + 1])
            except:
                break
    return np.array(event), np.array(time_event), np.array(lvl_before), np.array(lvl_after)











