import numpy as np
import numpy as np
from scipy import linalg
import active_Cas as active_Cas
from kinetic_parameters import change_concentration


def calc_fraction_bound(guide, target, epsilon, forward_rates, Cas, timepoints):
    '''
    solves the Master Equation for a given initial condition and desired time point
    :param rate_matrix: matrix with rates that makes up the Master Equation
    :param initial_condition: vector with initial configuration
    :param T: Evaluate solution at time T
    :return:
    '''
    # --- use properties of CRISPR-system ---
    guide_length = Cas.guide_length

    # --- catalytically dead: set kclv to zero ---
    fwrd_rates_dCas = forward_rates.copy()
    fwrd_rates_dCas[-1] = 0.

     # --- use parameters to build rate matrix ----
    mismatch_positions = active_Cas.get_mismatch_positions(guide, target, Cas)
    rate_matrix = get_master_equation(epsilon, fwrd_rates_dCas, mismatch_positions,guide_length)

    # --- use rate matrix to propagate Master Eqn ---
    everything_unbound = np.array([1.0] + [0.0] * (guide_length + 1))

    bound_fraction = []
    for time in timepoints:
        Prob = get_Probability(rate_matrix=rate_matrix, initial_condition=everything_unbound,T=time)
        bound_fraction.append( 1 - Prob[0] )
    return np.array(bound_fraction )

def Boltzmann(guide,target,epsilon,forward_rates, Cas):
    guide_length = Cas.guide_length
    mismatch_positions = active_Cas.get_mismatch_positions(guide, target, Cas)
    energies = get_energies(epsilon,mismatch_positions,guide_length)
    boltzmann_weigths = np.exp( -1*np.cumsum(energies) )
    return np.sum(boltzmann_weigths)


def equillibrium_binding_curve(guide, target,concentrations, epsilon, forward_rates, Cas):
    '''
    determine the "True" binding curve if the system is allowed to equillibrate
    '''
    eq_bnd = []
    for conc in concentrations:
        epsilon_new , _ = change_concentration(epsilon, forward_rates, conc, ref_concentration=1.0)

        boltzmann_weigths = Boltzmann(guide, target, epsilon_new, Cas)
        eq_bnd.append( 1. - 1.0/(1.0 + np.sum(boltzmann_weigths))  )
    return np.array(eq_bnd)



def Pbound(guide, target, epsilon, forward_rates, time, concentration,Cas):
    # --- use properties of CRISPR-system ---
    guide_length = Cas.guide_length

    # --- catalytically dead: set kclv to zero ---
    fwrd_rates_dCas = forward_rates.copy()
    fwrd_rates_dCas[-1] = 0.

    # --- use parameters to build rate matrix at 1nM ----
    mismatch_positions = active_Cas.get_mismatch_positions(guide, target, Cas)
    rate_matrix = get_master_equation(epsilon, fwrd_rates_dCas, mismatch_positions,guide_length)

    # --- use concentration to adjust rate matrix ---
    new_rate_matrix = rate_matrix.copy()
    new_rate_matrix[0][0] *= concentration
    new_rate_matrix[1][0] *= concentration

    # --- use rate matrix to propagate Master Eqn ---
    everything_unbound = np.array([1.0] + [0.0] * (guide_length + 1))
    Prob = get_Probability(rate_matrix=new_rate_matrix, initial_condition=everything_unbound,T=time)

    # --- get fraction of bound molecules (dCas9: so no cleaved molecules) ----
    bound_fraction = 1 - Prob[0]
    return bound_fraction

def get_master_equation(epsilon,forward_rates, mismatch_positions, guide_length):
    '''
    Construct rate matrix from given parameter set
    :param parameters:
    :param mismatch_positions:
    :param guide_length:
    :return:
    '''

    # --- dead Cas9, set kcat to zero ---
    fwrd_rates_dCas = forward_rates.copy()
    fwrd_rates_dCas[-1] = 0.

    # --- prepare Master equation(s) ---
    energies = get_energies(epsilon,mismatch_positions, guide_length)
    backward_rates = get_backward_rates(energies, fwrd_rates_dCas,guide_length )
    rate_matrix = build_rate_matrix(fwrd_rates_dCas, backward_rates)
    return rate_matrix

def get_energies(epsilon,mismatch_positions, guide_length=20):
    '''
    For general (position dependent) model make a list with the energies at every bound state
    At positions with a mismatch incorporated: add mismatch penalty (epsI[mm_pos])

    So this function returns the minima in the energy lanscape (the actual energy at every state)

    :param epsilon: [epsPAM, epsC[state_1],....,epsC[state_N],epsI[state_1],...epsI[state_N] ]
    provide as np.array()
    :param mismatch_positions: each mismach position has a range [1,2, ... , 20]
    :return: vector with the minima in the landscape
    '''
    if type(mismatch_positions)==type([]):
        mismatch_positions = np.array(mismatch_positions)
    new_epsilon = epsilon.copy()
    epsI = new_epsilon[(guide_length+1):]
    energies = -1*new_epsilon[0:(guide_length+1)] # convention: epsC>0 means downward slope
    energies[0] = new_epsilon[0]                 # convention: epsPAM>0 means upward slope
    if len(mismatch_positions)>0:
        energies[mismatch_positions.astype(int)] += epsI[(mismatch_positions.astype(int)-1)]
    return energies


def get_backward_rates(energies, forward_rates,guide_length=20):
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

    # 2) No rate backward from solution state
    backward_rates[0] = 0.0
    return backward_rates


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
    # rate_matrix = np.zeros((len(forward_rates), len(forward_rates)))  # Build the matrix

    rate_matrix = np.diag(diagonal1, k=0) + np.diag(diagonal2, k=1) + np.diag(diagonal3, k=-1)

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