import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
sns.set_style('ticks')
from dead_Cas import get_energies
from active_Cas import get_mismatch_positions, get_transition_states, get_transition_landscape


def plot_landscape(guide, target, epsilon, Cas, show_plot=True, axis=None, rel_concentration=1.):
    '''
    Plot the (free-)energy landscape of the on-target

    Added option to plot at different concentrations.
    Default is now set to 1 nM, the parameters are ASSUMED to be at 1 nM as well, hence concentration=0.1

    :param parameters:
    :param model_id:
    :return:
    '''

    # ---- retrieve model parameters from fit result -----
    # epsilon[0] -= np.log(rel_concentration)
    # fwrd_rates[0] += np.log10(rel_concentration)

    # ---- Get (possibly) mismatched energy landscape ----


    mismatch_positions = get_mismatch_positions(guide, target,Cas)
    energies = get_energies(epsilon, mismatch_positions, guide_length=20)
    energies[0] -= np.log(rel_concentration)

    # ---- Determine free-energy landscape ----
    landscape = [0.0] + list(np.cumsum(energies))
    landscape = np.array(landscape)


    if show_plot:
        if axis:
            axis.plot(range(-1, 21), landscape,
                 marker="o",
                 markersize=8,
                 markeredgewidth=2,
                 markerfacecolor="white");
            axis.set_xlabel('targeting progression', fontsize=12)
            axis.set_ylabel(r'free-energy ($k_BT$)', fontsize=12)
            axis.set_xticks(range(-1, 21))
            axis.set_xticklabels(['S', 'P', 1, '', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20],
                       rotation=0
                       , fontsize=10);
            axis.set_yticklabels(axis.get_yticks(),fontsize=12)
            plt.grid('on')
            sns.despine(ax=axis)
        else:
            plt.figure()
            axis = plt.plot(range(-1,21),landscape,
                            marker="o",
                            markersize=8,
                            markeredgewidth=2,
                            markerfacecolor="white");
            plt.xlabel('targeting progression', fontsize=12)
            plt.ylabel(r'free-energy ($k_BT$)',fontsize=12)
            plt.xticks(range(-1,21),
                       [ 'S','P',1,'', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20], rotation=0
                       ,fontsize=12);
            plt.yticks(fontsize=12)
            plt.grid('on')
            sns.despine()
    return landscape



def plot_transition_landscape(guide, target, epsilon,forward_rates, Cas, show_plot=True, axis=None, rel_concentration=1.):
    mismatch_positions = get_mismatch_positions(guide, target, Cas)
    energies = get_energies(epsilon, mismatch_positions, guide_length=20)
    energies[0] -= np.log(rel_concentration)

    Delta = get_transition_states(epsilon, forward_rates, mismatch_positions, Cas.guide_length)
    landscape = -1*get_transition_landscape(Delta)


    if show_plot:
        if axis:
            axis.plot(range(-1, 21), landscape,
                 marker="o",
                 markersize=8,
                 markeredgewidth=2,
                 markerfacecolor="white");
            axis.set_xlabel('targeting progression', fontsize=12)
            axis.set_ylabel(r'free-energy ($k_BT$)', fontsize=12)
            axis.set_xticks(range(-1, 21))
            axis.set_xticklabels(['S', 'P', 1, '', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20],
                       rotation=0
                       , fontsize=10);
            axis.set_yticklabels(axis.get_yticks(),fontsize=12)
            plt.grid('on')
            sns.despine(ax=axis)
        else:
            plt.figure()
            axis = plt.plot(range(0,21),landscape,
                            marker="o",
                            markersize=8,
                            markeredgewidth=2,
                            markerfacecolor="white");
            plt.xlabel('targeting progression', fontsize=12)
            plt.ylabel(r'free-energy ($k_BT$)',fontsize=12)
            plt.xticks(range(-1,21),
                       [ 'P',1,'', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20], rotation=0
                       ,fontsize=12);
            plt.yticks(fontsize=12)
            plt.grid('on')
            sns.despine()
    return landscape




def plot_mismatch_penalties(epsilon,axis=None, color=None,show_plot=True):
    '''
    plot mismatch penalties VS position as a bar plot
    :param parameters:
    :param model_id:
    :return:
    '''
    epsilon_I = epsilon[21:]
    if show_plot:
        if axis is None:
            ax = plt.gca()
        else:
            ax = axis
        if color:
            ax.bar([i+0.5 for i in range(1,21)], epsilon_I, color=color)
        else:
            ax.bar([i + 0.5 for i in range(1, 21)], epsilon_I)
        # window dressing:
        ax.set_xlabel('targeting progression', fontsize=10)
        ax.set_ylabel(r'mismatch penalties ($k_BT$)',fontsize=10)
        ax.set_xticks(np.arange(1,21)+0.5)
        ax.set_xticklabels([1, '', '', '', 5, '', '', '', '', 10, '', '', '', '', 15, '', '', '', '', 20],
                           rotation=0,
                           fontsize=10);
        ax.set_yticklabels(ax.get_yticks(),fontsize=15)
    return epsilon_I