#########################################################
# code to extract model parameters from SA fit
#########################################################
import numpy as np
import pandas as pd
import read_model_ID


def kinetic_parameters(filename, ID_Cas="Clv_Saturated_general_energies_v2",
                       ID_dCas="general_energies_no_kPR",
                       concentration_nM=10.,
                       nmbr_fit_params=44, fatch_solution="best_solution", parameter_vector_SA=None):
    '''
    This is made based on the sequence averaged model.
    From the SA fit, we get the parameters, at the specified concentration

    :param filename: output file name from SA fit
    :param ID_Cas: model_id for active Cas
    :param ID_dCas: model_id for dead Cas
    :param concentration_nM: concentration in nM of originally stored parameters
    :param nmbr_fit_params: number of free-parameters in SAfit
    :return:
    '''

    if filename:
        # -- extract from output file SA fit  (use not the final solution perse, but whatever gave lowest chi2) ------
        SAfit = pd.read_csv(filename, delimiter='\t', index_col=False)  # might need to adjust "index_col=39" to make more general?
        SAfit.reset_index(inplace=True)
        best_solution = np.argmin(SAfit.Potential)
        if fatch_solution == "best_solution":
            parameters = load_simm_anneal(filename, nmbr_fit_params, fatch_solution=best_solution)
        else:
            parameters = load_simm_anneal(filename, nmbr_fit_params, fatch_solution=fatch_solution)


    # ---- this "IF" construction should make it possible to also be used during SA,
    # then you do not have a file, but the parameter vector -------
    elif parameter_vector_SA:
        parameters = parameter_vector_SA

    # --- split into parameters fitted using dCas9 and Cas9 (Nucleaseq is done under saturating conditions) ---
    # ---- might need to adjust this part to make more general ? ----
    Cas_params = get_fit_parameters(ID_Cas, parameters)
    dCas_params= get_fit_parameters(ID_dCas, parameters)


    # --- get epsilon and forward rates ----
    epsilon, forward_rates = read_model_ID.unpack_parameters(dCas_params, model_id=ID_dCas)

    # --- epsilon PAM at 1nM ---
    epsilon_1nM = epsilon.copy()
    epsilon_1nM[0] += np.log(concentration_nM)

    # --- binding rate at 1nM ---
    kon = forward_rates[0] * concentration_nM**(-1)

    # --- internal forward rate ---
    kf = forward_rates[1]

    # --- catalytic rate ----
    _, forward_rates = read_model_ID.unpack_parameters(Cas_params, model_id=ID_Cas)
    forward_rates[0] = kon
    kcat = forward_rates[-1]
    return Cas_params, dCas_params, epsilon_1nM, forward_rates, kon, kf, kcat



############################################
def get_fit_parameters(modelID, parameters):
    if modelID == "general_energies_no_kPR":
        params = np.array(parameters[0:43])

    elif modelID == "Clv_Saturated_general_energies_v2":
        params = np.append(parameters[1:41], parameters[42:44])

    elif modelID == "Clv_100nM_general_energies_v2":
        params = np.array(parameters[:])

    else:
        print("Unknown model ID: " + modelID)
    return params



def load_simm_anneal(filename, Nparams, fatch_solution='final'):
    '''
    Load the parameter set from simmulated annealing.
    Fix indexing based on parameterisation to correctly import the table
    :param filename: filename output from SA fit
    :param Nparams: number of free-parameters in fit
    :param fatch_solution: allows to get intermediate solution. By default set to fatch the final solution
    :return:
    '''

    fit = pd.read_csv(filename, delimiter='\t', index_col=False)
    fit = fit.reset_index()
    final_result = []
    for param in range(1, Nparams + 1):
        col = 'Parameter ' + str(param)

        if fatch_solution == 'final':
            final_result.append(fit[col].iloc[-1])
        else:
            final_result.append(fit[col].iloc[fatch_solution])

    sa_result = np.array(final_result)
    return sa_result


def change_concentration(epsilon_1nM, forward_rates_1nM, new_concentration, ref_concentration=1.0):
    '''
    translate parameters to new concentration
    Here it is assumed you started with paramters at 1nM
    (can change using the 'ref_concentration')
    :param epsilon_1nM:
    :param forward_rates_1nM:
    :param new_concentration:
    :param ref_concentration:
    :return:
    '''

    # -- adjust epsilon PAM ---
    epsilon_new = np.array(epsilon_1nM).copy()
    epsilon_new[0] -= np.log(new_concentration/ref_concentration)

    # --- adjust binding rate from solution ---
    forward_rates_new = np.array(forward_rates_1nM).copy()
    forward_rates_new[0] *= new_concentration/ref_concentration
    return epsilon_new, forward_rates_new




