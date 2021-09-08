
#########################################################
# A function just to interpret the array of free-parameters in terms of our model parameters
# (Energies and Rates)
#
# used in "kinetic_parameters.py"
#########################################################
import numpy as np


def unpack_parameters(parameters, model_id, guide_length=20):
    """
    Use model ID to construct vector of epsilon values and forward rates.

    For every parametrization add a new case/ model_id
    :param parameters:
    :param model_id:
    :param guide_length:
    :return:
    """

    epsilon = np.zeros(2 * guide_length + 1)
    forward_rates = np.ones(guide_length + 2)

    if model_id == 'fit_landscape_v1':
        '''
        in stead of fitting the epsilon_C, we fit the cummalative sum of them, that is the energies of the bound states 
        '''
        # General position dependency
        energies_match = parameters[:21]
        epsilonPAM = parameters[0]
        epsilonC = -1 * np.diff(energies_match)

        epsilonI = parameters[21:-2]

        epsilon = list([epsilonPAM]) + list(epsilonC) + list(epsilonI)
        epsilon = np.array(epsilon)

        # --- rates: sol->PAM (concentration dependent), 1 constant forward rate for all remaining transitions
        rate_sol_to_PAM = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-1]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'Fixed_barrier_1+valley_free_ei_free_kclv':
        if len(parameters) != 29:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:13] = [-6.333715891930001, -1.3649045526199999, 4.4446667591199995, -1.53708382417, 0.852684737896,
                         -0.055977932609400004, -2.48652563315, 1.4324592539399998, 3.7556719419099998, 1.15654483726,
                         1.0245442149600001, 0.619640712364]
        # first bump from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        epsilon[13:21] = parameters[0:8]  # rest of landscape
        epsilon[21:41] = parameters[8:28]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.80705381635
        rate_internal = 10 ** 2.80705381635  # from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv


    elif model_id == 'Fixed_barrier_1+valley_free_ei_aba':
        if len(parameters) != 29:
            print('Wrong number of parameters')
            return

        epsilon[0] = 2.73766502381  # from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        epsilon[1:13] = [-6.333715891930001, -1.3649045526199999, 4.4446667591199995, -1.53708382417, 0.852684737896,
                         -0.055977932609400004, -2.48652563315, 1.4324592539399998, 3.7556719419099998, 1.15654483726,
                         1.0245442149600001, 0.619640712364]
        # first bump from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        epsilon[13:21] = parameters[0:8]  # rest of landscape
        epsilon[21:41] = parameters[8:28]

        rate_sol_to_PAM = 8.58596308e-04  # from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_PAM_to_R1 = 10 ** 2.80705381635
        rate_internal = 10 ** 2.80705381635  # from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_clv = 0.

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv


    elif model_id == 'Fixed_barrier_1_free_ei_free_kclv':
        if len(parameters) != 33:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:9] = [-6.333715891930001, -1.3649045526199999, 4.4446667591199995, -1.53708382417, 0.852684737896,
                        -0.055977932609400004, -2.48652563315, 1.4324592539399998]
        # first bump from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:41] = parameters[12:32]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.80705381635
        rate_internal = 10 ** 2.80705381635  # from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Fixed_barrier_1_free_ei_aba':
        if len(parameters) != 33:
            print('Wrong number of parameters')
            return

        epsilon[0] = 2.73766502381  # from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        epsilon[1:9] = [-6.333715891930001, -1.3649045526199999, 4.4446667591199995, -1.53708382417, 0.852684737896,
                        -0.055977932609400004, -2.48652563315, 1.4324592539399998]
        # first bump from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:41] = parameters[12:32]

        rate_sol_to_PAM = 8.58596308e-04  # from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_PAM_to_R1 = 10 ** 2.80705381635
        rate_internal = 10 ** 2.80705381635  # from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_clv = 0.

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Fixed_barrier_1_fixed_ei_free_kclv':
        if len(parameters) != 13:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:9] = [-6.333715891930001, -1.3649045526199999, 4.4446667591199995, -1.53708382417, 0.852684737896,
                        -0.055977932609400004, -2.48652563315, 1.4324592539399998]
        # first bump from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:41] = [5.65333843414, 4.11041237185, 6.4824076359200005, 6.9767208316, 6.26648140987, 7.3885677834,
                          6.899487638360001,
                          6.22048036118, 8.99420177438, 7.2536151786800005, 7.40355907947, 7.02417975261, 7.73956089271,
                          7.88443131328,
                          7.6448096557500005, 6.35536811387, 5.13351785131, 4.24795057861, 5.83044257892, 2.41878154505]
        # mismatches from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.80705381635
        rate_internal = 10 ** 2.80705381635  # from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Fixed_barrier_1_fixed_ei_aba':
        if len(parameters) != 13:
            print('Wrong number of parameters')
            return

        epsilon[0] = 2.73766502381  # from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        epsilon[1:9] = [-6.333715891930001, -1.3649045526199999, 4.4446667591199995, -1.53708382417, 0.852684737896,
                        -0.055977932609400004, -2.48652563315, 1.4324592539399998]
        # first bump from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:41] = [5.65333843414, 4.11041237185, 6.4824076359200005, 6.9767208316, 6.26648140987, 7.3885677834,
                          6.899487638360001,
                          6.22048036118, 8.99420177438, 7.2536151786800005, 7.40355907947, 7.02417975261, 7.73956089271,
                          7.88443131328,
                          7.6448096557500005, 6.35536811387, 5.13351785131, 4.24795057861, 5.83044257892, 2.41878154505]
        # mismatches from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        rate_sol_to_PAM = 8.58596308e-04  # from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_PAM_to_R1 = 10 ** 2.80705381635
        rate_internal = 10 ** 2.80705381635  # from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_clv = 0.

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Fixed_barrier_1_constant_ei_free_kclv':
        if len(parameters) != 13:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:9] = [-6.333715891930001, -1.3649045526199999, 4.4446667591199995, -1.53708382417, 0.852684737896,
                        -0.055977932609400004, -2.48652563315, 1.4324592539399998]
        # first bump from ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:41] = 6.9
        # mismatches from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.80705381635
        rate_internal = 10 ** 2.80705381635  # from: ../fits_Stijn/18_7_2019/fit_18_7_2019_sim_17.txt
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv


    elif model_id == 'Sequence_dependent_clv_v3':
        if len(parameters) != 14:
            print('Wrong number of parameters')

            return

        epsilonConfig = np.zeros(21)
        epsilonConfig[0] = -100.0  # PAM
        epsilonConfig[1:21] = [-8.51732646, -0.92309739, 4.31859981, -0.65323077, 0.56563897, -5.95998582,
                               -4.36131563, 2.83445163, 0.44828209, 9.52067702, 2.93084561, -6.94015605,
                               2.58949616, -6.66097551, 0.07608957, 6.30479899, 0.26091506, -0.91985863,
                               -4.12120118, 4.4768232]  # Configuration Energies from cleavage fit 3_4 fit 1

        epsilonBind = np.zeros(16)
        epsilonBind[0] = parameters[0]  # AA
        epsilonBind[1] = 0.  # AT
        epsilonBind[2] = parameters[1]  # AC
        epsilonBind[3] = parameters[2]  # AG
        epsilonBind[4] = 0.  # UA
        epsilonBind[5] = parameters[3]  # UT
        epsilonBind[6] = parameters[4]  # UC
        epsilonBind[7] = parameters[5]  # UG
        epsilonBind[8] = parameters[6]  # CA
        epsilonBind[9] = parameters[7]  # CT
        epsilonBind[10] = parameters[8]  # CC
        epsilonBind[11] = 0.  # CG
        epsilonBind[12] = parameters[9]  # GA
        epsilonBind[13] = parameters[10]  # GT
        epsilonBind[14] = 0.  # GC
        epsilonBind[15] = parameters[11]  # GG

        #  parameters:
        #  0  1  2  3  4  5  6  7  8  9 10 11
        #  A  A  A  U  U  U  C  C  C  G  G  G
        #  A  C  G  T  C  G  A  T  C  A  T  G

        #  epsilonBind:
        #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #  A  A  A  A  U  U  U  U  C  C  C  C  G  G  G  G
        #  A  T  C  G  A  T  C  G  A  T  C  G  A  T  C  G

        rate_sol_to_PAM = 1000.0
        rate_internal = 10 ** parameters[-2]
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = rate_clv

        return epsilonConfig, epsilonBind, forward_rates

    elif model_id == 'Sequence_dependent_clv_v2':
        if len(parameters) != 38:
            print('Wrong number of parameters')
            return

        epsilonConfig = np.zeros(21)
        epsilonConfig[0] = -100.0  # PAM
        epsilonConfig[1:21] = parameters[0:20]  # Configuration Energies

        epsilonBind = np.zeros(16)  # Binding Energies
        epsilonBind[0] = parameters[0 + 20]  # AA
        epsilonBind[1] = parameters[1 + 20]  # AT
        epsilonBind[2] = parameters[2 + 20]  # AC
        epsilonBind[3] = parameters[3 + 20]  # AG
        epsilonBind[4] = parameters[4 + 20]  # UA
        epsilonBind[5] = parameters[5 + 20]  # UT
        epsilonBind[6] = parameters[6 + 20]  # UC
        epsilonBind[7] = parameters[7 + 20]  # UG
        epsilonBind[8] = parameters[8 + 20]  # CA
        epsilonBind[9] = parameters[9 + 20]  # CT
        epsilonBind[10] = parameters[10 + 20]  # CC
        epsilonBind[11] = parameters[11 + 20]  # CG
        epsilonBind[12] = parameters[12 + 20]  # GA
        epsilonBind[13] = parameters[13 + 20]  # GT
        epsilonBind[14] = parameters[14 + 20]  # GC
        epsilonBind[15] = parameters[15 + 20]  # GG

        #  parameters:
        #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #  A  A  A  A  U  U  U  U  C  C  C  C  G  G  G  G
        #  A  T  C  G  A  T  C  G  A  T  C  G  A  T  C  G

        #  epsilonBind:
        #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #  A  A  A  A  U  U  U  U  C  C  C  C  G  G  G  G
        #  A  T  C  G  A  T  C  G  A  T  C  G  A  T  C  G

        rate_sol_to_PAM = 1000.0
        rate_internal = 10 ** parameters[-2]
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = rate_clv

        return epsilonConfig, epsilonBind, forward_rates

    elif model_id == 'Sequence_dependent_clv_v1':
        if len(parameters) != 35:
            print('Wrong number of parameters')
            return

        epsilonConfig = np.zeros(21)
        epsilonConfig[0] = -100.0  # PAM
        epsilonConfig[1:21] = parameters[0:20]  # Configuration Energies

        epsilonBind = np.zeros(16)  # Binding Energies
        epsilonBind[0] = parameters[0 + 20]  # AA
        epsilonBind[1] = parameters[1 + 20]  # AT
        epsilonBind[2] = parameters[3 + 20]  # AC
        epsilonBind[3] = parameters[2 + 20]  # AG
        epsilonBind[4] = parameters[4 + 20]  # UA
        epsilonBind[5] = parameters[7 + 20]  # UT
        epsilonBind[6] = parameters[12 + 20]  # UC
        epsilonBind[7] = parameters[10 + 20]  # UG
        epsilonBind[8] = parameters[3 + 20]  # CA
        epsilonBind[9] = parameters[6 + 20]  # CT
        epsilonBind[10] = parameters[11 + 20]  # CC
        epsilonBind[11] = parameters[9 + 20]  # CG
        epsilonBind[12] = parameters[2 + 20]  # GA
        epsilonBind[13] = parameters[5 + 20]  # GT
        epsilonBind[14] = parameters[9 + 20]  # GC
        epsilonBind[15] = parameters[8 + 20]  # GG

        #  parameters:
        #  0  1  2  3  4  5  6  7  8  9 10 11 12
        #  A  A  A  A  A  T  T  T  G  G  G  C  C
        #  A  T  G  C  U  G  C  U  G  C  U  C  U

        #  epsilonBind:
        #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #  A  A  A  A  U  U  U  U  C  C  C  C  G  G  G  G
        #  A  T  C  G  A  T  C  G  A  T  C  G  A  T  C  G

        rate_sol_to_PAM = 1000.0
        rate_internal = 10 ** parameters[-2]
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = rate_clv

        return epsilonConfig, epsilonBind, forward_rates


    elif model_id == 'Clv_Saturated_fixed_kf_general_energies_v2':
        if len(parameters) != 41:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:] = parameters[:-1]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 200.0
        rate_internal = 200.0
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'First_bump_fixed_for_engineered_cas_fixed_ei':
        if len(parameters) != 13:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:9] = [-2.81040538514, 0.128379438824, -1.3398902138600002, 2.7652185198900003,
                        -2.14593791558, -3.32591676706, 0.660542944429,
                        1.44885468799]  # first bump from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt
        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:41] = [6.21210100464, 4.15685735918, 5.83016614093, 6.3469277879299995, 5.0974118607,
                          6.11732311667, 6.71550763398, 5.800000188080001, 8.798329158660001,
                          7.23490655459, 7.205774376210001, 6.204434796699999, 7.374378511639999,
                          7.108293219149999, 6.51786601314, 5.95938096651, 4.1476939067, 2.22435794942,
                          4.85593404545,
                          6.91967070006]  # mismatches from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.5336611063
        rate_internal = 10 ** 2.5336611063
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'First_bump_fixed_for_engineered_cas_on_fixed_ei':
        if len(parameters) != 12:
            print('Wrong number of parameters')
            return

        epsilon[0] = 1.4  # predefined epsilon PAM at saturation
        epsilon[1:9] = [-2.81040538514, 0.128379438824, -1.3398902138600002, 2.7652185198900003,
                        -2.14593791558, -3.32591676706, 0.660542944429,
                        1.44885468799]  # first bump from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt
        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:41] = [6.21210100464, 4.15685735918, 5.83016614093, 6.3469277879299995, 5.0974118607,
                          6.11732311667, 6.71550763398, 5.800000188080001, 8.798329158660001,
                          7.23490655459, 7.205774376210001, 6.204434796699999, 7.374378511639999,
                          7.108293219149999, 6.51786601314, 5.95938096651, 4.1476939067, 2.22435794942,
                          4.85593404545,
                          6.91967070006]  # mismatches from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt

        rate_sol_to_PAM = 10 ** -2.4  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.5336611063
        rate_internal = 10 ** 2.5336611063
        rate_clv = 0

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Engineered_cas_fixed_ei':
        if len(parameters) != 21:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:21] = parameters[0:20]
        epsilon[21:41] = [6.21210100464, 4.15685735918, 5.83016614093, 6.3469277879299995, 5.0974118607,
                          6.11732311667, 6.71550763398, 5.800000188080001, 8.798329158660001,
                          7.23490655459, 7.205774376210001, 6.204434796699999, 7.374378511639999,
                          7.108293219149999, 6.51786601314, 5.95938096651, 4.1476939067, 2.22435794942,
                          4.85593404545,
                          6.91967070006]  # mismatches from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.5336611063
        rate_internal = 10 ** 2.5336611063
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Engineered_cas_on_fixed_ei':
        if len(parameters) != 20:
            print('Wrong number of parameters')
            return

        epsilon[0] = 1.4  # predefined epsilon PAM at saturation
        epsilon[1:21] = parameters[0:20]  # rest of landscape
        epsilon[21:41] = [6.21210100464, 4.15685735918, 5.83016614093, 6.3469277879299995, 5.0974118607,
                          6.11732311667, 6.71550763398, 5.800000188080001, 8.798329158660001,
                          7.23490655459, 7.205774376210001, 6.204434796699999, 7.374378511639999,
                          7.108293219149999, 6.51786601314, 5.95938096651, 4.1476939067, 2.22435794942,
                          4.85593404545,
                          6.91967070006]  # mismatches from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt

        rate_sol_to_PAM = 10 ** -2.4  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.5336611063
        rate_internal = 10 ** 2.5336611063
        rate_clv = 0

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Engineered_cas_constant_ei':
        if len(parameters) != 21:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:21] = parameters[0:20]
        epsilon[21:41] = 6.

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.5336611063
        rate_internal = 10 ** 2.5336611063
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Engineered_cas_on_constant_ei':
        if len(parameters) != 20:
            print('Wrong number of parameters')
            return

        epsilon[0] = 1.4  # predefined epsilon PAM at saturation
        epsilon[1:21] = parameters[0:20]  # rest of landscape
        epsilon[21:41] = 6.

        rate_sol_to_PAM = 10 ** -2.4  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.5336611063
        rate_internal = 10 ** 2.5336611063
        rate_clv = 0

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv


    elif model_id == 'First_bump_fixed_for_engineered_cas':
        if len(parameters) != 25:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:9] = [-2.81040538514, 0.128379438824, -1.3398902138600002, 2.7652185198900003,
                        -2.14593791558, -3.32591676706, 0.660542944429,
                        1.44885468799]  # first bump from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt
        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:29] = [6.21210100464, 4.15685735918, 5.83016614093, 6.3469277879299995, 5.0974118607,
                          6.11732311667, 6.71550763398,
                          5.800000188080001]  # mismatches from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt
        epsilon[29:41] = parameters[12:24]  # rest of mismatches

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.5336611063
        rate_internal = 10 ** 2.5336611063
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'First_bump_fixed_for_engineered_cas_on':
        if len(parameters) != 24:
            print('Wrong number of parameters')
            return

        epsilon[0] = 1.4
        epsilon[1:9] = [-2.81040538514, 0.128379438824, -1.3398902138600002, 2.7652185198900003,
                        -2.14593791558, -3.32591676706, 0.660542944429,
                        1.44885468799]  # first bump from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt
        epsilon[9:21] = parameters[0:12]  # rest of landscape
        epsilon[21:29] = [6.21210100464, 4.15685735918, 5.83016614093, 6.3469277879299995, 5.0974118607,
                          6.11732311667, 6.71550763398,
                          5.800000188080001]  # mismatches from ../fits_Stijn/13_6_2019/fit_13_6_2019_sim_2.txt
        epsilon[29:41] = parameters[12:24]  # rest of mismatches

        rate_sol_to_PAM = 10 ** -2.4  # predefined at saturation
        rate_PAM_to_R1 = 10 ** 2.5336611063
        rate_internal = 10 ** 2.5336611063
        rate_clv = 0

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv


    elif model_id == 'Clv_100nM_general_energies_v2':
        if len(parameters) != 44:
            print('Wrong number of parameters')
            return
        # fix at 100nM
        epsilon[0] = parameters[0] - np.log(10.0)
        epsilon[1:] = parameters[1:-3]

        rate_sol_to_PAM = 10 * 10 ** parameters[-3]  # predefined at saturation
        rate_PAM_to_R1 = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-2]  # rate from PAM to R is equal to internal rate
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Clv_Saturated_general_energies_v2':
        if len(parameters) != 42:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:] = parameters[:-2]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-2]  # rate from PAM to R is equal to internal rate
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Clv_init_limit_Saturated_general_energies_fixed_koff':
        if len(parameters) != 43:
            print('Wrong number of parameters')
            return

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** parameters[-3]
        rate_internal = 10 ** parameters[-2]  # rate from PAM to R is equal to internal rate
        rate_clv = 10 ** parameters[-1]

        epsilon[0] = -100.  # -np.log(rate_sol_to_PAM) #since rate_PAM_to_sol == 1 Hz
        epsilon[1:] = parameters[:-3]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Bnd_init_limit_general_energies_fixed_koff':
        # ---- have the rate from PAM into R-loop the same as the forward rate within R-loop
        if len(parameters) != 43:
            print('Wrong number of parameters')
            return
        # General position dependency
        epsilon[:] = parameters[:-2]

        # --- rates: sol->PAM (concentration dependent), 1 constant forward rate for all remaining transitions
        rate_sol_to_PAM = np.exp(-epsilon[0])  # since rate_PAM_to_sol == 1 Hz
        rate_PAM_to_R1 = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-1]  # rate from PAM to R is equal to internal rate
        rate_clv = 0.  # dCas9

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Clv_Saturated_general_energies_landscape':
        if len(parameters) != 42:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1] = -parameters[0]
        for i in range(2, 21):
            epsilon[i] = -(parameters[i - 1] - parameters[i - 2])
        epsilon[21:41] = parameters[20:40]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-2]  # rate from PAM to R is equal to internal rate
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Clv_Saturated_edit_boyle_landscape':
        if len(parameters) != 35:
            print('Wrong number of parameters')
            return

        landscape = np.zeros(21)
        landscape[0] = 1.389248 - 1.389248  # PAM Boyle, minus PAM Boyle :)
        landscape[1:9] = parameters[0:8]  # first bump
        landscape[9:13] = [5.412620 - 1.389248, 1.547533 - 1.389248, -0.105180 - 1.389248,
                           -0.153215 - 1.389248]  # well defined dip, from Boyle
        landscape[13:18] = parameters[8:13]  # second bump
        landscape[18:21] = [-0.361180 - 1.389248, -4.009278 - 1.389248,
                            -8.223548 - 1.389248]  # well defined second dip, from Boyle

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        for i in range(1, 21):
            epsilon[i] = -(landscape[i] - landscape[i - 1])
        epsilon[21:41] = parameters[13:33]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-2]  # rate from PAM to R is equal to internal rate
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Clv_Saturated_edit_boyle_landscape_flat':
        if len(parameters) != 24:
            print('Wrong number of parameters')
            return

        landscape = np.zeros(21)
        landscape[0] = 1.389248 - 1.389248  # PAM Boyle, minus PAM Boyle :)
        landscape[1:10] = np.ones(9) * parameters[0]  # first bump of constant height
        landscape[10:13] = [1.547533 - 1.389248, -0.105180 - 1.389248,
                            -0.153215 - 1.389248]  # well defined dip, from Boyle
        landscape[13:18] = np.ones(5) * parameters[1]  # second bump of constant height
        landscape[18:21] = [-0.361180 - 1.389248, -4.009278 - 1.389248,
                            -8.223548 - 1.389248]  # well defined second dip, from Boyle

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        for i in range(1, 21):
            epsilon[i] = -(landscape[i] - landscape[i - 1])
        epsilon[21:41] = parameters[2:22]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-2]  # rate from PAM to R is equal to internal rate
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Clv_Saturated_edit_boyle_landscape_flat_constant_ei':
        if len(parameters) != 6:
            print('Wrong number of parameters')
            return

        landscape = np.zeros(21)
        landscape[0] = 1.389248 - 1.389248  # PAM Boyle, minus PAM Boyle :)
        landscape[1:10] = np.ones(9) * parameters[0]  # first bump of constant height
        landscape[10:13] = [1.547533 - 1.389248, -0.105180 - 1.389248,
                            -0.153215 - 1.389248]  # well defined dip, from Boyle
        landscape[13:18] = np.ones(5) * parameters[1]  # second bump of constant height
        landscape[18:21] = [-0.361180 - 1.389248, -4.009278 - 1.389248,
                            -8.223548 - 1.389248]  # well defined second dip, from Boyle

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        for i in range(1, 21):
            epsilon[i] = -(landscape[i] - landscape[i - 1])
        epsilon[21:37] = parameters[2]
        epsilon[37:41] = parameters[3]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-2]  # rate from PAM to R is equal to internal rate
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'On_edit_boyle_landscape':
        if len(parameters) != 35:
            print('Wrong number of parameters')
            return
        landscape = np.zeros(21)
        landscape[0] = 1.389248 - 1.389248  # PAM Boyle, minus PAM Boyle :)
        landscape[1:9] = parameters[0:8]  # first bump
        landscape[9:13] = [5.412620 - 1.389248, 1.547533 - 1.389248, -0.105180 - 1.389248,
                           -0.153215 - 1.389248]  # well defined dip, from Boyle
        landscape[13:18] = parameters[8:13]  # second bump
        landscape[18:21] = [-0.361180 - 1.389248, -4.009278 - 1.389248,
                            -8.223548 - 1.389248]  # well defined second dip, from Boyle

        epsilon[0] = 1.389248  # Boyle PAM
        for i in range(1, 21):
            epsilon[i] = -(landscape[i] - landscape[i - 1])
        epsilon[21:41] = parameters[13:33]

        rate_sol_to_PAM = 10 ** parameters[-2]
        rate_PAM_to_R1 = 10 ** parameters[-1]
        rate_internal = 10 ** parameters[-1]  # rate from PAM to R is equal to internal rate
        rate_clv = 0

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'On_edit_boyle_landscape_flat':
        if len(parameters) != 24:
            print('Wrong number of parameters')
            return

        landscape = np.zeros(21)
        landscape[0] = 1.389248 - 1.389248  # PAM Boyle, minus PAM Boyle :)
        landscape[1:10] = np.ones(9) * parameters[0]  # first bump of constant height
        landscape[10:13] = [1.547533 - 1.389248, -0.105180 - 1.389248,
                            -0.153215 - 1.389248]  # well defined dip, from Boyle
        landscape[13:18] = np.ones(5) * parameters[1]  # second bump of constant height
        landscape[18:21] = [-0.361180 - 1.389248, -4.009278 - 1.389248,
                            -8.223548 - 1.389248]  # well defined second dip, from Boyle

        epsilon[0] = 1.389248  # Boyle PAM
        for i in range(1, 21):
            epsilon[i] = -(landscape[i] - landscape[i - 1])
        epsilon[21:41] = parameters[2:22]

        rate_sol_to_PAM = 10 ** parameters[-2]
        rate_PAM_to_R1 = 10 ** parameters[-1]
        rate_internal = 10 ** parameters[-1]  # rate from PAM to R is equal to internal rate
        rate_clv = 0

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'On_edit_boyle_landscape_flat_constant_ei':
        if len(parameters) != 6:
            print('Wrong number of parameters')
            return

        landscape = np.zeros(21)
        landscape[0] = 1.389248 - 1.389248  # PAM Boyle, minus PAM Boyle :)
        landscape[1:10] = np.ones(9) * parameters[0]  # first bump of constant height
        landscape[10:13] = [1.547533 - 1.389248, -0.105180 - 1.389248,
                            -0.153215 - 1.389248]  # well defined dip, from Boyle
        landscape[13:18] = np.ones(5) * parameters[1]  # second bump of constant height
        landscape[18:21] = [-0.361180 - 1.389248, -4.009278 - 1.389248,
                            -8.223548 - 1.389248]  # well defined second dip, from Boyle

        epsilon[0] = 1.389248  # Boyle PAM
        for i in range(1, 21):
            epsilon[i] = -(landscape[i] - landscape[i - 1])
        epsilon[21:37] = parameters[2]
        epsilon[37:41] = parameters[3]

        rate_sol_to_PAM = 10 ** parameters[-2]
        rate_PAM_to_R1 = 10 ** parameters[-1]
        rate_internal = 10 ** parameters[-1]  # rate from PAM to R is equal to internal rate
        rate_clv = 0

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv

    elif model_id == 'Clv_init_limit_Saturated_general_energies_v2':
        if len(parameters) != 43:
            print('Wrong number of parameters')
            return

        epsilon[0] = -100.0  # predefined epsilon PAM at saturation
        epsilon[1:] = parameters[:-3]

        rate_sol_to_PAM = 1000.0  # predefined at saturation
        rate_PAM_to_R1 = 10 ** parameters[-3]
        rate_internal = 10 ** parameters[-2]
        rate_clv = 10 ** parameters[-1]

        forward_rates = forward_rates * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv


    elif model_id == 'Clv_init_limit_general_energies_v2':
        # General position dependency
        epsilon = parameters[:-4]

        rate_sol_to_PAM = 10 ** parameters[-4]
        rate_PAM_to_R1 = 10 ** parameters[-3]
        rate_internal = 10 ** parameters[-2]
        rate_clv = 10 ** parameters[-1]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = rate_clv


    elif model_id == 'general_energies_no_kPR_fixed_PAM':
        # ---- have the rate from PAM into R-loop the same as the forward rate within R-loop
        if len(parameters) != 42:
            print('Wrong number of parameters')
            return
        # General position dependency
        epsilon[0] = 1.4
        epsilon[1:] = parameters[:-2]

        # --- rates: sol->PAM (concentration dependent), 1 constant forward rate for all remaining transitions
        rate_sol_to_PAM = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-1]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'general_energies_no_kPR_fixed_PAM_landscape':
        # ---- have the rate from PAM into R-loop the same as the forward rate within R-loop
        if len(parameters) != 42:
            print('Wrong number of parameters')
            return
        # General position dependency
        epsilon[0] = 1.4
        epsilon[1] = -parameters[0]
        for i in range(2, 21):
            epsilon[i] = -(parameters[i - 1] - parameters[i - 2])
        epsilon[21:41] = parameters[20:40]

        # --- rates: sol->PAM (concentration dependent), 1 constant forward rate for all remaining transitions
        rate_sol_to_PAM = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-1]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'general_energies_no_kPR':
        # ---- have the rate from PAM into R-loop the same as the forward rate within R-loop
        if len(parameters) != 43:
            print('Wrong number of parameters')
            return
        # General position dependency
        epsilon = parameters[:-2]

        # --- rates: sol->PAM (concentration dependent), 1 constant forward rate for all remaining transitions
        rate_sol_to_PAM = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-1]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'general_energies_no_kPR_landscape':
        # ---- have the rate from PAM into R-loop the same as the forward rate within R-loop
        if len(parameters) != 43:
            print('Wrong number of parameters')
            return
        landscape = np.array(parameters[0:21])
        landscape[0] = 0.

        epsilon[0] = parameters[0]
        for i in range(1, 21):
            epsilon[i] = -(landscape[i] - landscape[i - 1])
        epsilon[21:41] = parameters[21:41]

        # --- rates: sol->PAM (concentration dependent), 1 constant forward rate for all remaining transitions
        rate_sol_to_PAM = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-1]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave


    elif model_id == 'landscape_lowest_chi_squared_fit_rates':
        # ---- fix the energies---
        # (copied from parameter file: '../data/25_10_2018/fit_25_10_2018_sim_22.txt') ----
        epsilon = np.array([1.43364597e+00, -2.51895658e+00, -8.38107740e-01,
                            -1.00837871e+00, -3.89888343e+00, 4.98565931e+00,
                            -2.24062010e+00, 1.75709991e+00, 1.48346110e+00,
                            -2.56251518e+00, 4.76022290e+00, 1.66832631e+00,
                            -4.41487326e-04, -3.01917678e+00, 1.70186470e+00,
                            -2.69692160e+00, 4.63508021e+00, 3.43845249e+00,
                            -3.53360655e+00, 3.90785543e+00, 3.95624011e+00,
                            8.41041112e+00, 3.52511767e+00, 6.47092824e+00,
                            6.29617812e+00, 5.87466899e+00, 4.02069468e+00,
                            6.97289538e+00, 5.39037459e+00, 6.53991724e+00,
                            6.04624779e+00, 6.11140010e+00, 4.95893203e+00,
                            5.40442705e+00, 5.69985755e+00, 5.12293027e+00,
                            5.62074797e+00, 4.81777124e+00, 7.94515945e+00,
                            9.77311952e+00, 6.84175107e+00])

        #  epsilon = np.array([ 1.31882561, -6.5880103 ,  1.59239502,  0.46021068, -2.49593644,
        #  0.09580053,  4.54430596, -3.37045113,  0.37192334,  1.02581499,
        #  4.12556609,  1.64960851, -0.03692466, -4.49653651,  4.39600456,
        # -3.57616013,  3.90152848,  3.48127153, -4.66585257,  1.77729046,
        #  8.90727104,  7.95522837,  4.24585854,  8.89394253,  8.89430408,
        #  5.15323997,  4.0149383 ,  6.61232836,  5.12389258,  7.22642299,
        #  6.06820965,  5.94807726,  4.90830081,  4.99741095,  6.38253949,
        #  5.87159526,  6.62698767,  5.87749165,  5.58373498,  9.01010833,
        #  4.79058499])

        # --- fit the timescales ----
        rate_sol_to_PAM = 10 ** parameters[0]
        rate_PAM_to_R1 = 10 ** parameters[1]
        rate_internal = 10 ** parameters[2]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'Boyle_median_landscape_fit_rates':

        # ---- fix the energies
        # (copied from parameter file: '../data/25_10_2018/median_landscape_Boyle_2Dgrid.txt' on 04/12/2018) ----
        epsilon = np.array([1.37168412, -4.15848621, -1.9680678, -0.88508854, -0.70265355,
                            2.00690677, 0.44846725, -0.98458337, 0.69054452, 1.39284825,
                            3.99499433, 1.64210983, 0.01835355, -3.80696161, 2.1347165,
                            -0.80030528, 1.97963966, 3.08286715, -0.99001786, 2.70016623,
                            3.62856361, 7.15113618, 3.43168686, 8.05355657, 7.47981321,
                            5.62454054, 3.95761041, 6.69713956, 5.23678395, 7.46232812,
                            6.10345114, 6.1114001, 4.97393321, 5.29781208, 6.10472344,
                            5.43762448, 5.02455717, 4.3662047, 3.43647645, 7.07048864,
                            5.22717571])

        # --- fit the timescales ----
        rate_sol_to_PAM = 10 ** parameters[0]
        rate_PAM_to_R1 = 10 ** parameters[1]
        rate_internal = 10 ** parameters[2]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = 0.0  # dCas9 does not cleave


    elif model_id == 'general_energies_rates':
        epsilon = parameters[:(2 * guide_length + 1)]
        forward_rates = np.ones(guide_length + 2)
        forward_rates[:-1] = 10 ** np.array(parameters[(2 * guide_length + 1):])
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'general_energies':
        # General position dependency + minimal amount of rates
        epsilon = parameters[:-2]
        forward_rates = np.ones(guide_length + 2) * parameters[-2]  # internal rates
        forward_rates[0] = parameters[-1]  # from solution to PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'general_energies_v2':
        # General position dependency + minimal amount of rates
        epsilon = parameters[:-2]
        forward_rates = np.ones(guide_length + 2) * 10 ** parameters[-2]  # internal rates
        forward_rates[0] = 10 ** parameters[-1]  # from solution to PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'init_limit_general_energies_v0':
        # General position dependency
        epsilon = parameters[:-3]
        forward_rates = np.ones(guide_length + 2) * parameters[-2]  # internal rates
        forward_rates[0] = parameters[-1]  # from solution to PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave
        # first rate from PAM into R-loop is less or equal to other internal rates
        forward_rates[1] = np.exp(-parameters[-3]) * parameters[-2]


    elif model_id == 'init_limit_general_energies':
        # General position dependency
        epsilon = parameters[:-3]
        forward_rates = np.ones(guide_length + 2) * 10 ** parameters[-2]  # internal rates
        forward_rates[0] = 10 ** parameters[-1]  # from solution to PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave
        # first rate from PAM into R-loop is less or equal to other internal rates
        forward_rates[1] = np.exp(-parameters[-3]) * 10 ** parameters[-2]

    elif model_id == 'init_limit_general_energies_v2':
        # General position dependency
        epsilon = parameters[:-3]

        rate_sol_to_PAM = 10 ** parameters[-3]
        rate_PAM_to_R1 = 10 ** parameters[-2]
        rate_internal = 10 ** parameters[-1]

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = 0.0  # dCas9 does not cleave


    elif model_id == 'init_limit_fast_internal_general_energies':
        # General position dependency for energies
        epsilon = parameters[:-3]

        internal_rates = 10 ** parameters[-1]  # directly set the rate (the order of magnitude)
        rate_sol_to_PAM = 10 ** parameters[-3]  # directly set a rate (the order of magnitude)
        rate_PAM_to_R1 = np.exp(-parameters[-2] + epsilon[
            1]) * internal_rates  # through placement of the transition state above R1's energy.

        forward_rates = np.ones(guide_length + 2) * internal_rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[1] = rate_PAM_to_R1
        forward_rates[-1] = 0.0





    elif model_id == 'constant_eps_I':
        # General position dependency for matches, constant mismatch penalty
        epsPAM = parameters[0]
        epsilonC = parameters[1:(guide_length + 1)]
        epsilonI = parameters[guide_length + 1]

        epsilon[0] = epsPAM
        epsilon[1:(guide_length + 1)] = epsilonC
        epsilon[(guide_length + 1):] = epsilonI

        forward_rates = np.ones(guide_length + 2) * parameters[-2]  # internal rates
        forward_rates[0] = parameters[-1]  # from solution to PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    elif model_id == 'init_limit_lock_const_EpsI':
        e_PAM = parameters[0]
        ec_1 = parameters[1]
        ec_first = parameters[2]
        ec_second = parameters[3]
        e_I = parameters[4]
        x = parameters[5]
        k_PAM = parameters[6]
        E_barr = parameters[7]
        k = parameters[8]
        k_1 = k * np.exp(-E_barr)

        epsilon[0] = e_PAM
        epsilon[1] = ec_1
        epsilon[2:x + 1] = ec_first
        epsilon[x + 1:guide_length + 1] = ec_second
        epsilon[guide_length + 1:] = e_I

        forward_rates = np.ones(guide_length + 2) * k  # internal rates
        forward_rates[0] = k_PAM
        forward_rates[1] = k_1
        forward_rates[-1] = 0.0

    elif model_id == 'init_limit_two_drops_fixed_BP':
        pos1 = 10
        pos2 = 18

        e_PAM = parameters[0]
        ec_1 = parameters[1]
        ec_first = parameters[2]
        drop1 = parameters[3]
        ec_second = parameters[4]
        drop2 = parameters[5]
        ec_third = parameters[6]
        e_I = parameters[7]
        k_PAM = parameters[8]
        E_barr = parameters[9]
        k = parameters[10]
        k_1 = k * np.exp(-E_barr)

        epsilon[0] = e_PAM
        epsilon[1] = ec_1
        epsilon[2:pos1 + 1] = ec_first
        epsilon[pos1 + 1] = drop1
        epsilon[pos1 + 2:pos2 + 1] = ec_second
        epsilon[pos2 + 1] = drop2
        epsilon[pos2 + 2:guide_length + 1] = ec_third
        epsilon[guide_length + 1:] = e_I

        forward_rates = np.ones(guide_length + 2) * k  # internal rates
        forward_rates[0] = k_PAM
        forward_rates[1] = k_1
        forward_rates[-1] = 0.0

    elif model_id == 'init_limit_two_drops':

        e_PAM = parameters[0]
        ec_1 = parameters[1]
        ec_first = parameters[2]
        drop1 = parameters[3]
        ec_second = parameters[4]
        drop2 = parameters[5]
        ec_third = parameters[6]
        pos1 = parameters[7]
        pos2 = parameters[8]
        e_I = parameters[9]
        k_PAM = parameters[10]
        E_barr = parameters[11]
        k = parameters[12]
        k_1 = k * np.exp(-E_barr)

        epsilon[0] = e_PAM
        epsilon[1] = ec_1
        epsilon[2:pos1 + 1] = ec_first
        epsilon[pos1 + 1] = drop1
        epsilon[pos1 + 2:pos2 + 1] = ec_second
        epsilon[pos2 + 1] = drop2
        epsilon[pos2 + 2:guide_length + 1] = ec_third
        epsilon[guide_length + 1:] = e_I

        forward_rates = np.ones(guide_length + 2) * k  # internal rates
        forward_rates[0] = k_PAM
        forward_rates[1] = k_1
        forward_rates[-1] = 0.0

    elif model_id == 'init_limit_5EpsC_2EpsI':
        e_PAM = parameters[0]
        ec_1 = parameters[1]
        ec_2 = parameters[2]
        ec_3 = parameters[3]
        ec_4 = parameters[4]
        ec_5 = parameters[5]
        eI_1 = parameters[6]
        eI_2 = parameters[7]
        bp2 = parameters[8]
        bp3 = parameters[9]
        bp4 = parameters[10]
        bpI = parameters[11]
        k_PAM = parameters[12]
        E_barr = parameters[13]
        k = parameters[14]
        k_1 = k * np.exp(-E_barr)

        epsilon[0] = e_PAM
        epsilon[1] = ec_1
        epsilon[2:bp2 + 1] = ec_2
        epsilon[bp2 + 1:bp3 + 1] = ec_3
        epsilon[bp3 + 1:bp4 + 1] = ec_4
        epsilon[bp4 + 1:guide_length + 1] = ec_5
        epsilon[guide_length + 1:guide_length + bpI + 1] = eI_1
        epsilon[guide_length + bpI + 1:] = eI_2

        forward_rates = np.ones(guide_length + 2) * k  # internal rates
        forward_rates[0] = k_PAM
        forward_rates[1] = k_1
        forward_rates[-1] = 0.0

    elif model_id == 'fixed_rates':
        # ---- have the rate from PAM into R-loop the same as the forward rate within R-loop
        if len(parameters) != 41:
            print('Wrong number of parameters')
            return
        # General position dependency
        epsilon = parameters

        # --- rates: sol->PAM (concentration dependent), 1 constant forward rate for all remaining transitions
        rate_sol_to_PAM = 0.00038941973449552436
        rate_internal = 471.7450318294534

        forward_rates = np.ones(guide_length + 2) * rate_internal  # internal rates
        forward_rates[0] = rate_sol_to_PAM
        forward_rates[-1] = 0.0  # dCas9 does not cleave

    else:
        print('Watch out! Non-existing model-ID..')
        return

    return epsilon, forward_rates


def combined_model(parameters, model_ID):
    model_ID_clv, model_ID_on = model_ID.split('+')

    if model_ID == 'Clv_Saturated_general_energies_landscape+general_energies_no_kPR_fixed_PAM_landscape':
        if len(parameters) != 43:
            print('Wrong number of parameters')
            return
        parameters_clv = np.append(parameters[0:40], parameters[41:43])
        parameters_on = np.array(parameters[0:42])

    elif model_ID == 'Clv_Saturated_edit_boyle_landscape+On_edit_boyle_landscape':
        if len(parameters) != 36:
            print('Wrong number of parameters')
            return
        parameters_clv = np.append(parameters[0:33], parameters[34:36])
        parameters_on = np.array(parameters[0:35])

    elif model_ID == 'Clv_Saturated_edit_boyle_landscape_flat+On_edit_boyle_landscape_flat':
        if len(parameters) != 25:
            print('Wrong number of parameters')
            return
        parameters_clv = np.append(parameters[0:22], parameters[23:25])
        parameters_on = np.array(parameters[0:24])


    elif model_ID == 'Clv_100nM_general_energies_v2+general_energies_no_kPR':
        if len(parameters) != 44:
            print('Wrong number of parameters')
            return
        parameters_clv = parameters[:]
        parameters_on = np.array(parameters[0:43])


    elif model_ID == 'Clv_Saturated_general_energies_v2+general_energies_no_kPR':
        if len(parameters) != 44:
            print('Wrong number of parameters')
            return
        parameters_clv = np.append(parameters[1:41], parameters[42:44])
        parameters_on = np.array(parameters[0:43])

    elif model_ID == 'Clv_Saturated_general_energies_landscape+general_energies_no_kPR_landscape':
        if len(parameters) != 44:
            print('Wrong number of parameters')
            return
        parameters_clv = np.append(parameters[1:41], parameters[42:44])
        parameters_on = np.array(parameters[0:43])

    elif model_ID == 'Clv_Saturated_edit_boyle_landscape_flat_constant_ei+On_edit_boyle_landscape_flat_constant_ei':
        if len(parameters) != 7:
            print('Wrong number of parameters')
            return
        parameters_clv = np.append(parameters[0:4], parameters[5:7])
        parameters_on = np.array(parameters[0:6])

    elif model_ID == 'First_bump_fixed_for_engineered_cas+First_bump_fixed_for_engineered_cas_on':
        if len(parameters) != 25:
            print('Wrong number of parameters')
            return
        parameters_clv = parameters
        parameters_on = parameters[:-1]

    elif model_ID == 'First_bump_fixed_for_engineered_cas_fixed_ei+First_bump_fixed_for_engineered_cas_on_fixed_ei':
        if len(parameters) != 13:
            print('Wrong number of parameters')
            return
        parameters_clv = parameters
        parameters_on = parameters[:-1]

    elif model_ID == 'Engineered_cas_fixed_ei+Engineered_cas_on_fixed_ei' or model_ID == 'Engineered_cas_constant_ei+Engineered_cas_on_constant_ei':
        if len(parameters) != 21:
            print('Wrong number of parameters')
            return
        parameters_clv = parameters
        parameters_on = parameters[:-1]

    elif model_ID == 'Clv_init_limit_Saturated_general_energies_fixed_koff+Bnd_init_limit_general_energies_fixed_koff':
        if len(parameters) != 44:
            print('Wrong number of parameters')
            return
        parameters_clv = parameters[1:]
        parameters_on = parameters[:-1]

    return model_ID_clv, model_ID_on, parameters_clv, parameters_on