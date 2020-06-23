import numpy as np
import dead_Cas as dCas
import kinetic_parameters
from CRISPRclass import CRISPR
from scipy import linalg



################################################################################
# Main model calulations for quantities involving cleavage
#  it is assumed one has the model parameters as a list of energy differences (epsilon) and a list of
# forward rates. Possibly extracted using "kinetic_parameters.py" from SA fit
#
#
#    1. Pclv : "Probability to cleave a bound target before rejection"
#       => "Hybridization Kinetics Explains Off-targeting Rules"
#    2.
################################################################################

'''
Time to cleave 
'''

def calc_fraction_cleaved(guide, target, epsilon, forward_rates,Cas,
                          timepoints):
    '''
    solves the Master Equation for a given initial condition and desired time point
    :param rate_matrix: matrix with rates that makes up the Master Equation
    :param initial_condition: vector with initial configuration
    :param T: Evaluate solution at time T
    :return:
    '''
    mismatch_positions = get_mismatch_positions(guide,target,Cas)
    guide_length = Cas.guide_length
    rate_matrix = get_master_equation(epsilon,forward_rates, mismatch_positions, guide_length)

    everything_unbound = np.array([1.0] + [0.0] * (guide_length + 1))
    P0 = everything_unbound
    M = rate_matrix
    fraction_cleaved = []
    for T in timepoints:
        matrix_exponent = linalg.expm(+M*T)

        probs = matrix_exponent.dot(P0)
        fraction_cleaved.append( 1 - np.sum(probs)  )

    return np.array(fraction_cleaved)



def calc_MFPT(rate_matrix, initial_condition):
    '''
    use direct determination of MFPT from system of master equations
    :param rate_matrix:
    :return:
    '''
    Minv = np.linalg.inv(-1 * rate_matrix)
    vec = np.ones(len(rate_matrix))
    MFPT = vec.dot(Minv.dot(initial_condition))  # <vec| M^{-1} | P0>
    return MFPT

def cleavage_rate(guide, target, epsilon, forward_rates, Cas):
    # --- prepare Master Equation ---
    guide_length = Cas.guide_length
    mismatch_positions = get_mismatch_positions(guide, target, Cas)
    rate_matrix = get_master_equation(epsilon,forward_rates, mismatch_positions, guide_length)

    # --- determine inverse average time ---
    everything_unbound = np.array([1.0] + [0.0] * (guide_length + 1))
    avg_time = calc_MFPT(rate_matrix, everything_unbound)
    return avg_time**(-1)


def rate_complete_Rloop(guide, target, epsilon, forward_rates,Cas):
    # --- prepare Master Equation ---
    guide_length = Cas.guide_length
    mismatch_positions = get_mismatch_positions(guide, target,Cas )
    energies = dCas.get_energies(epsilon, mismatch_positions, guide_length=guide_length)
    backward_rates = dCas.get_backward_rates(energies, forward_rates)

    # --- use cleavage competent state into absorber ---
    rates_fwd = forward_rates[:-1]
    rates_bck = backward_rates[:-1]
    rate_matrix = build_rate_matrix(rates_fwd, rates_bck)

    # --- determine inverse average time ---
    everything_unbound = np.array([1.0] + [0.0] * guide_length)
    avg_time = calc_MFPT(rate_matrix, everything_unbound)
    return avg_time**(-1)





'''
coarse-grained system describes fits to bulk data (spCas9)

coincides with picture of moving HNH domain. Used the following terminology:
1. solution state  (sol)
2. Open HNH state (O) at the PAM-bound configuration
3. Intermediate HNH configuration when R-loop has progressed to nt 7-13
4. Closed HNH configuration when complete R-loop has been formed, cleavage competent 
(5. post-cleavage state)

Rates are calulated as follows:
1. rates between solution and Open (PAM) are taken directly from fit based on 20-state model 
2. cleavage rate (kcat) also directly taken from fit 
3. forward rates from O-->I and I-->C are determined using MFPT. Location of state I is found 
    by using minimum energy in selected range
4. (remaining) backward rates are set using detailed balance. For this we need corresponding energies of the coarse-grained states:
    A. solution has energy 0 kT 
    B. Open state has energy of PAM state 
    C. Intermediate state gets assigned effective free-energy of states between 7 and 13 (sum of BM-factors)
    D. Closed state has energy of cleavage competent state 
'''





def cg_HNH(guide, target, epsilon,forward_rates):
    # --- this system is for spCas9 ----
    spCas9 = CRISPR(guide_length=20, PAM_length=3, PAM_sequence='NGG')
    # --- determine energies of 20-state model (using mismatch positions) ----
    mismatch_positions = get_mismatch_positions(guide, target,spCas9 )
    energies = dCas.get_energies(epsilon, mismatch_positions, guide_length=spCas9.guide_length)
    backward_rates = dCas.get_backward_rates(energies, forward_rates)

    # --- energies coarse-grained system ----
    E_sol = 0.0
    E_O = energies[0]
    landscape = np.cumsum(energies)
    E_I = -1 * np.log( np.sum( np.exp( -1*landscape[7:13]   )    )  )
    E_C = landscape[-1]

    # --- location of intermediate state ----
    loc_O = 0
    loc_I = np.argmin( landscape[7:13]  ) + 7
    loc_C = 20

    # -- rate from solution to O from original rates ---
    kon = forward_rates[0]

    # --- solve for rate from O-->I using MFPT ---
    rates_fwd = forward_rates[loc_O+1:loc_I+2]
    rates_bck = backward_rates[loc_O+1:loc_I+2].copy()
    rates_bck[0] = 0.0
    matrix_OI = build_rate_matrix(rates_fwd, rates_bck)
    start_at_O_state = np.array([1.0] + [0.0]*loc_I)
    MFPT = calc_MFPT(matrix_OI, start_at_O_state)
    kOI = 1.0/MFPT

    # --- solve for rate from I-->C using MFPT ---
    rates_fwd = forward_rates[loc_I+1:loc_C+2]
    rates_bck = backward_rates[loc_I+1:loc_C+2].copy()
    rates_bck[0] = 0.0
    matrix_IC = build_rate_matrix(rates_fwd, rates_bck)
    start_at_I_state = np.array([1.0] + [0.0]*(loc_C-loc_I))
    MFPT = calc_MFPT(matrix_IC, start_at_I_state)
    kIC = 1.0/MFPT

    # -- rate to cleave from original rates ---
    kcat = forward_rates[-1]

    # -- determine backward rates in coarse-grained system ---
    cg_energies  = [E_sol,E_O, E_I, E_C]
    cg_fwd_rates = [kon, kOI, kIC, kcat]

    cg_bck_rates = dCas.get_backward_rates(np.diff(cg_energies), cg_fwd_rates)
    return  cg_energies, cg_fwd_rates, cg_bck_rates


def build_rate_matrix(forward_rates, backward_rates):
    '''
    build matrix in Master Equation

    :param forward_rates:
    :param backward_rates:
    :return:
    '''
    diagonal1 = -(forward_rates + backward_rates)
    diagonal2 = backward_rates[1:]
    diagonal3 = forward_rates[:-1]
    rate_matrix = np.diag(diagonal1, k=0) + np.diag(diagonal2, k=1) + np.diag(diagonal3, k=-1)
    return rate_matrix











''' 
Pclv from "Hybr.Kin. Explains Off-targeting Rules"
'''
def Pclv(guide, target, epsilon,forward_rates,Cas):
    guide_length = Cas.guide_length
    mismatch_positions = get_mismatch_positions(target, guide,Cas)
    Delta = get_transition_states(epsilon, forward_rates, mismatch_positions, guide_length)
    DeltaT = get_transition_landscape(Delta)
    exp_of_T = np.sum(np.exp(-DeltaT))
    return 1.0 / (1.0 + exp_of_T)

def get_mismatch_positions(seq1, seq2,Cas):
    PAMlength = Cas.PAM_length
    Target = np.array(list(seq1))[:-PAMlength]   # could make for general PAM length
    Guide = np.array(list(seq2))[:-PAMlength]
    mismatch_positions = np.where(Guide!= Target)[0]
    return 20 - mismatch_positions

def get_transition_states(epsilon, forward_rates, mismatch_positions, guide_length=20):

    # 1) determine free-energy minima
    energies = dCas.get_energies(epsilon, mismatch_positions, guide_length)

    # 2) Use detailed balance to get the backward rates:
    backward_rates = get_backward_rates(energies, forward_rates)

    # 3) Use Kramer's rate to convert to Delta's:
    Delta = np.log(forward_rates[1:] / backward_rates)
    return Delta

def get_transition_landscape(Delta):
    return np.cumsum(Delta)

def get_backward_rates(energies, forward_rates):
    '''
    Apply detailed balance condition to get the backward rates from the energies and forward rates

    :param energies:
    :param forward_rates:
    :param guide_length:
    :return:
    '''
    # 0) Construct array containing backward rates
    backward_rates = np.zeros(len(forward_rates))

    # 1) Apply detailed balance condition:
    backward_rates[1:] = forward_rates[:-1] * np.exp(energies)
    return backward_rates

def get_master_equation(epsilon,forward_rates, mismatch_positions, guide_length):
    '''
    Construct rate matrix from given parameter set
    :param parameters:
    :param mismatch_positions:
    :param guide_length:
    :return:
    '''
    # --- prepare Master equation(s) ---
    energies = dCas.get_energies(epsilon,mismatch_positions, guide_length)
    backward_rates = get_backward_rates(energies, forward_rates )
    rate_matrix = build_rate_matrix(forward_rates, backward_rates)
    return rate_matrix

def get_Probability(rate_matrix, initial_condition,T):
    '''
    solves the Master Equation for a given initial condition and desired time point
    :param rate_matrix: matrix with rates that makes up the Master Equation
    :param initial_condition: vector with initial configuration
    :param T: Evaluate solution at time T
    :return:
    '''
    P0 = initial_condition
    M = rate_matrix
    matrix_exponent = linalg.expm(+M*T)
    return matrix_exponent.dot(P0)