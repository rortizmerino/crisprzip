import numpy as np
import active_Cas
import dead_Cas
import kinetic_parameters as kinpar


################################################################################
# mimic the experiment by Boyle et al. using kinetic model
#
#
################################################################################



def calc_association_rate(guide, target, epsilon, forward_rates,Cas,timepoints=[500., 1000., 1500.],
                          rel_concentration=1.):




    # --- prepare  master Equation ---
    guide_length = Cas.guide_length
    mismatch_positions = active_Cas.get_mismatch_positions(target, guide,Cas)
    rate_matrix = dead_Cas.get_master_equation(epsilon, forward_rates, mismatch_positions, guide_length)



    # ---  Calculate bound fraction for specified time points ---
    # Association experiment starts with all dCas9 being unbound:
    everything_unbound = np.array([1.0] + [0.0] * (guide_length + 1))

    # -- Association rate is taken at 1nM the other data at 10nM --> adjust k_on appropriatly (if needed) --- :
    new_rate_matrix = rate_matrix.copy()
    new_rate_matrix[0][0] *= rel_concentration  #rel_concentration=1 corresponds to 10nM
    new_rate_matrix[1][0] *= rel_concentration

    bound_fraction = []
    for time in timepoints:
        Probabilities = dead_Cas.get_Probability(rate_matrix=new_rate_matrix,initial_condition=everything_unbound,T=time)
        bound_fraction.append( np.sum(Probabilities[1:]))

    #-- Use least-squares to fit straight line with origin forced through zero --- :
    association_rate = least_squares_line_through_origin(x_points=np.array(timepoints),y_points=np.array(bound_fraction))
    return association_rate




def calc_dissociation_rate(guide, target, epsilon, forward_rates,Cas,
                           timepoints=[500.,1000.,1500.]):
    '''
    Dissociation experiment for (effective off-rate):
    method as used in Boyle et al.

    :return:
    '''


    # --- prepare  master Equation ---
    guide_length = Cas.guide_length
    mismatch_positions = active_Cas.get_mismatch_positions(target, guide,Cas)
    rate_matrix = dead_Cas.get_master_equation(epsilon, forward_rates, mismatch_positions, guide_length)


    # --- unbinding experiment starts after subjecting DNA to Cas9 for 12 hours ---
    everything_unbound = np.array([1.0] + [0.0] * (guide_length + 1))
    after_12hrs = dead_Cas.get_Probability(rate_matrix=rate_matrix, initial_condition=everything_unbound, T=12 * 3600)


    # --- Flush out all freely floating dCas9 --> Set occupation in solution to zero ---
    after_12hrs[0] = 0.0

    # --- Renormalize remainder ---
    after_12hrs = after_12hrs / np.sum(after_12hrs)

    # --- Competitor on-targets DNA catch all unbound dCas9 'immediately' --> Set on-rate to zero and rebuild matrix ---
    new_rate_matrix = rate_matrix.copy()
    new_rate_matrix[0][0] = 0.0
    new_rate_matrix[1][0] = 0.0

    #--- For time_k in timepoints solve the Master equation and track the fraction of dCas9 molecules in solution ---
    unbound_fraction = []
    for time in timepoints:
        Probabilities = dead_Cas.get_Probability(rate_matrix=new_rate_matrix, initial_condition=after_12hrs, T=time)
        unbound_fraction.append(Probabilities[0])

    # --- Use least-squares to fit ---
    _ , dissocation_rate = least_squares(x_points=timepoints, y_points=unbound_fraction)
    return dissocation_rate



def least_squares(x_points, y_points):
    size = len(x_points)
    X = np.ones((size, 2))
    X[:, 1] = x_points

    XT = X.transpose()
    Y = y_points

    a = np.dot(np.dot(np.linalg.inv(np.dot(XT, X)), XT), Y)

    intercept = a[0]
    slope = a[1]
    return intercept, slope

def least_squares_line_through_origin(x_points, y_points):
    return np.sum( x_points*y_points )/np.sum( x_points*x_points )