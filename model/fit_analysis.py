import os

import pandas as pd
import numpy as np

from typing import Union

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib import animation, colors
import seaborn as sns
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model.hybridization_kinetics import Searcher, SearcherPlotter
from model.training_set import TrainingSet


class LogAnalyzer:
    """Class to read and interpret the log file of an optimization
    run.

    Attributes
    ----------
    log_file: str
        Path of the log file
    log_dataframe: pd.DataFrame
        Dataframe containing all the run information
    log_searchers: pd.Series
        Series containing the Searcher instances corresponding to all
        parameter vectors in the log dataframe
    total_steps: int
        Total number of steps in optimization run
    final_cost: float
        Fit cost of the final parameter vector
    landscape_lims: tuple
        Lower and upper limit of the on-target landscape (for plotting)
    mismatch_lims: tuple
        Lower and upper limit of the mismatch landscape (for plotting)
    rates_lims: tuple
        Lower and upper limit of the forward rates (for plotting)

    Methods
    -------
    plot_full_log_dashboard()
        Creates full log dashboard, with the final parameter state
    make_dashboard_video()
        Creates matplotlib.Animation object of the dashboard during the
        optimization run
    """

    dashboard_specs = {
        'size': (8, 6),  # in inch
        'dpi': 150,  # dots per inch
        'fps': 15  # frames per second
    }

    def __init__(self, log_file):
        self.log_file = log_file

        self.log_dataframe = self.read_log_file()
        self.total_steps = self.get_total_steps()
        self.log_searchers = self.make_searcher_series()

        self.exit_message = self.get_exit_message()
        self.final_cost = self.get_final_cost()
        self.final_result = self.get_final_result()
        self.runtime = self.get_run_time()

        self.lowest_cost = self.get_lowest_cost()
        # self.best_searchers = self.find_best_searchers()

        self.landscape_lims, self.mismatch_lims, self.rates_lims = \
            self.get_plot_limits()

    def read_log_file(self):
        """Returns a pd.DataFrame with the log content"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()
            log_start = 0
            for line in log_lines:
                if line.split('\t')[0].strip() == 'Cycle no':
                    break
                else:
                    log_start += 1

        dataframe = pd.read_table(self.log_file,
                                  skiprows=log_start,
                                  delimiter='\t',
                                  skipinitialspace=True)
        return dataframe

    def get_exit_message(self):
        """Get exit message from log file"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()

        return log_lines[3].strip()

    def get_final_cost(self):
        """Get final cost from log file"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()

            for line in log_lines:
                if line[:10] == 'Final cost':
                    final_cost = float(line[-21:])
                    return final_cost

    def get_total_steps(self):
        """Get total steps from log file"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()

            for line in log_lines:
                if line[:17] == 'Total step number':
                    total_steps = int(line[-21:])
                    return total_steps

    def get_run_time(self):
        """Get run time from log file"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()

            for line in log_lines:
                if line[:12] == 'Time elapsed':
                    runtime = line[-21:].strip()
                    return runtime

    def get_final_result(self):
        """Get final result from log file"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()

            for line in log_lines:
                if line[:6] == 'result':
                    final_result = np.array(
                        [float(val) for val in line[:-1].split('\t')[1:]])
                    return final_result

    def get_plot_limits(self):
        """Get upper and lower bounds from log file"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()

            for line in log_lines:
                if line[:9] == 'lower bnd':
                    lower_bnd = np.array(
                        [float(val) for val in line[:-1].split('\t')[1:]])
                if line[:9] == 'upper bnd':
                    upper_bnd = np.array(
                        [float(val) for val in line[:-1].split('\t')[1:]])

        sep = (len(lower_bnd) - 3) // 2
        landscape_lims = (lower_bnd[:sep].min(), upper_bnd[:sep].max())
        mismatch_lims = (lower_bnd[sep:-3].min(), upper_bnd[sep:-3].max())
        rates_lims = (lower_bnd[-3:].min(), upper_bnd[-3:].max())
        rates_lims = tuple([10 ** val for val in rates_lims])  # log axs
        return landscape_lims, mismatch_lims, rates_lims

    def make_searcher_series(self):
        """Creates a Searcher instance for each parameter vector,
        returns them in a Series"""
        landscape_df = self.log_dataframe
        searcher_series = pd.Series(dtype=object)
        for i in landscape_df.index:
            param_vector = landscape_df.iloc[:, 4:].loc[i].to_numpy()
            searcher_series.loc[i] = Searcher.from_param_vector(param_vector)
        return searcher_series

    def get_lowest_cost(self) -> np.ndarray:
        lowest_cost = np.zeros(shape=self.total_steps)
        for row in self.log_dataframe.iterrows():
            lowest_cost[int(row[1]['Cycle no']):] = row[1]['Cost']
        return lowest_cost

    def prepare_opt_path_canvas(self, x_lims=None, y_lims=None,
                                title='Optimization path', axs=None):
        """Creates or adapts Figure and Axes objects such
        that an optimization path line can later be added."""

        if axs is None:
            _, axs = plt.subplots(1, 1, figsize=(3, 3))

        # x axis
        if x_lims is None:
            x_lims = (0, self.total_steps)
        axs.set_xlim(-.1 * x_lims[1], 1.1 * x_lims[1])
        axs.set_xlabel(r'Step number', **SearcherPlotter.label_style)

        # y axis
        axs.set_yscale('log')
        if y_lims is None:
            y_lims = (min(self.lowest_cost), max(self.lowest_cost))
        # calculate 10% overhang at both sides
        if y_lims[0] < y_lims[1]:
            y_overhang = (y_lims[1] / y_lims[0]) ** .1
            axs.set_ylim(y_lims[0] / y_overhang,
                         y_lims[1] * y_overhang)
        axs.set_ylabel(r'Potential $V$ (A.U.)',
                       **SearcherPlotter.label_style)

        # title
        axs.set_title(title, **SearcherPlotter.title_style)

        # styling
        axs.grid('on')
        sns.set_style('ticks')
        sns.despine(ax=axs)
        return axs

    @staticmethod
    def prepare_opt_path_lines(axs, color='cornflowerblue', **plot_kwargs):
        """Adds styled lines to the optimization path canvas"""
        # colored, left-side line
        past_line, = axs.plot([], [], color=color,
                              linewidth=1, zorder=2, **plot_kwargs)
        # colored, dot at current opt point
        current_line, = axs.plot([], [], color=color,
                                 linestyle='', zorder=2,
                                 marker='.', markersize=12,
                                 **plot_kwargs)
        # gray, right-side line
        future_line, = axs.plot([], [], color='darkgray', linewidth=1,
                                zorder=1.9,
                                **plot_kwargs)

        lines = [past_line, current_line, future_line]
        return lines

    def update_partial_opt_path(self, lines: list, i):
        """Updates optimization path up to data point i"""
        # past line
        lines[0].set_data(np.arange(0, i + 1), self.lowest_cost[:i + 1])
        # current line (point)
        lines[1].set_data(i, self.lowest_cost[i])
        # future line
        lines[2].set_data(np.arange(i, self.total_steps),
                          self.lowest_cost[i:])

    def plot_full_opt_path(self, x_lims=None, y_lims=None,
                           color='cornflowerblue', axs=None,
                           **plot_kwargs):
        """Creates complete optimization path plot"""
        axs = self.prepare_opt_path_canvas(
            x_lims, y_lims,
            title='Optimization path',
            axs=axs
        )
        lines = self.prepare_opt_path_lines(
            axs,
            color=color,
            **plot_kwargs
        )
        lines[0].set_data(np.arange(0, self.total_steps), self.lowest_cost)
        return axs

    def prepare_log_dashboard_canvas(self):
        """Creates Figure and Axes objects to which all optimization
        plots (landscape/mismatch/rates/opt path plot) can later be
        added."""
        fig = plt.figure(
            figsize=self.dashboard_specs['size'],
            constrained_layout=True,
        )
        grid = fig.add_gridspec(ncols=2, nrows=2,
                                width_ratios=[2, 1],
                                height_ratios=[1, 1])

        axs = [
            fig.add_subplot(grid[0, 0]),  # landscape plot
            fig.add_subplot(grid[1, 0]),  # mismatch plot
            fig.add_subplot(grid[0, 1]),  # rates plot
            fig.add_subplot(grid[1, 1])  # opt path plot
        ]

        initial_searcher = self.log_searchers[0]
        axs[0] = (SearcherPlotter(initial_searcher)
                  .prepare_landscape_canvas(self.landscape_lims,
                                            title='On-target landscape',
                                            axs=axs[0]))
        axs[1] = (SearcherPlotter(initial_searcher)
                  .prepare_landscape_canvas(self.mismatch_lims,
                                            title='Mismatch penalties',
                                            axs=axs[1]))
        axs[2] = (SearcherPlotter(initial_searcher)
                  .prepare_rates_canvas(self.rates_lims,
                                        title='Forward rates',
                                        axs=axs[2]))
        axs[3] = self.prepare_opt_path_canvas(x_lims=None, y_lims=None,
                                              title='Optimization path',
                                              axs=axs[3])
        return fig, axs

    def prepare_log_dashboard_line(self, axs, color='cornflowerblue',
                                   **plot_kwargs):
        """Adds styled lines to the log dashboard canvas"""
        initial_plot = SearcherPlotter(self.log_searchers[0])
        lines = [
                    initial_plot.prepare_landscape_line(axs[0], color,
                                                        **plot_kwargs),
                    initial_plot.prepare_landscape_line(axs[1], color,
                                                        **plot_kwargs),
                    initial_plot.prepare_rates_line(axs[2], color,
                                                    **plot_kwargs)
                ] + self.prepare_opt_path_lines(axs[3], color, **plot_kwargs)

        return lines

    def update_log_dashboard_line(self, lines, i, blit=False):
        """Updates log dashboard up to data point i"""

        i = min(i, self.total_steps - 1)
        self.update_partial_opt_path(lines[3:6], i)

        # j points to the best searcher at point i (j <= i)
        j = self.log_dataframe.index[self.log_dataframe['Cycle no'] <= i][-1]

        if blit and self.log_dataframe.loc[j, 'Cycle no'] < i:
            pass  # prevents redundant line updating
        else:
            plotter = SearcherPlotter(self.log_searchers[j])
            plotter.update_on_target_landscape(lines[0])
            plotter.update_mismatches(lines[1])
            plotter.update_rates(lines[2])

    def plot_final_log_dashboard(self, color):
        """Creates full log dashboard, with the result and the
         full optimization path"""
        fig, axs = self.prepare_log_dashboard_canvas()
        lines = self.prepare_log_dashboard_line(axs, color=color)

        plotter = SearcherPlotter(
            Searcher.from_param_vector(self.final_result)
        )
        plotter.update_on_target_landscape(lines[0])
        plotter.update_mismatches(lines[1])
        plotter.update_rates(lines[2])
        self.update_partial_opt_path(lines[3:6], self.total_steps - 1)

        lines[4].set_data([], [])
        if len(lines) == 7:
            lines[6].txt.set_text('')
        return fig, axs, lines

    def make_dashboard_video(self):
        video = DashboardVideo([self.log_file]).make_video()
        return video

    def compare_to_data(self, training_set: TrainingSet):
        data = training_set.data
        training_set.run_all_simulations(self.final_result,
                                         multiprocessing=False)
        data['simulation'] = training_set.simulated_values
        return data

    @staticmethod
    def plot_single_mm_fit(data_df, experiment_name,
                           color=None, axs=None):
        ylabel = ''
        if color is None:
            if experiment_name.lower() == 'nucleaseq':
                color = 'orange'
                ylabel = r'$k_{clv} ($s$^{-1})$'
            elif experiment_name.lower() == 'champ':
                color = 'tab:blue'
                ylabel = r'$K_A ($nM$^{-1})$'
            else:
                color = 'tab:blue'

        df_subset = data_df.loc[
            (data_df.mismatch_number == 1) &
            (data_df.experiment_name == experiment_name),
            ['value', 'error', 'weight', 'simulation']
        ]

        if axs is None:
            _, axs = plt.subplots(1, 1)

        axs.errorbar(x=np.arange(1, 21),
                     y=df_subset['value'], yerr=df_subset['error'],
                     fmt='.', color=color, markersize=12,
                     label='data')
        axs.plot(np.arange(1, 21), df_subset['simulation'],
                 '-', color=color, linewidth=2, label='model')

        # window dressing
        axs.set_xticks(np.arange(1, 21))
        axs.set_xticks(np.arange(1, 21))
        axs.set_xticklabels(['1'] + 3 * [''] + ['5'] + 4 * [''] +
                            ['10'] + 4 * [''] + ['15'] + 4 * [''] + ['20'])
        axs.set_xlabel(r'mismatch position $b$', **SearcherPlotter.label_style)
        axs.set_yscale('log')
        axs.set_ylabel(ylabel, **SearcherPlotter.label_style)
        axs.legend(loc='upper left')

        partial_cost = (np.sum(df_subset['weight'] *
                               np.log10(df_subset['simulation'] /
                                        df_subset['value']) ** 2))
        axs.set_title(f'single mm cost: {partial_cost:.2e}', pad=18)

        return axs

    @staticmethod
    def plot_double_mm_fit(data_df, experiment_name,
                           cmap=None, axs=None):

        if experiment_name.lower() == 'nucleaseq':
            cmap = 'Oranges' if cmap is None else cmap
            zlabel = r'$k_{clv} ($s$^{-1})$'
        elif experiment_name.lower() == 'champ':
            cmap = 'Blues' if cmap is None else cmap
            zlabel = r'$K_A ($nM$^{-1})$'
        else:
            cmap = 'Blues' if cmap is None else cmap
            zlabel = ''

        df_subset = data_df.loc[
            (data_df.mismatch_number == 2) &
            (data_df.experiment_name == experiment_name),
            ['mismatch_array', 'value', 'error', 'weight', 'simulation']
        ]

        def mismatch_array_to_coordinates(mm_array):
            x = mm_array.index('1')
            y = x + 1 + mm_array[x + 1:].index('1')
            return x, y

        show_matrix = np.zeros(shape=(20, 20))
        for row in df_subset.iterrows():
            x, y = mismatch_array_to_coordinates(row[1]['mismatch_array'])
            show_matrix[y, x] = row[1]['value']
            show_matrix[x, y] = row[1]['simulation']

        if axs is None:
            _, axs = plt.subplots(1, 1)

        im = axs.imshow(show_matrix, cmap=cmap, norm=colors.LogNorm(),
                        origin='lower')

        label1 = AnchoredText('  data', frameon=False, pad=-0.5,
                              loc='lower left', bbox_to_anchor=(0., 1.),
                              bbox_transform=axs.transAxes,
                              prop={'size': 10,
                                    'horizontalalignment': 'left'})
        label2 = AnchoredText('model  ', frameon=False, pad=-0.5,
                              loc='lower left', bbox_to_anchor=(1., 0.),
                              bbox_transform=axs.transAxes,
                              prop={'size': 10,
                                    'horizontalalignment': 'right',
                                    'rotation': 270.})
        axs.add_artist(label1)
        axs.add_artist(label2)

        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        bar = plt.colorbar(im, cax=cax)
        cax.set_title(zlabel, **SearcherPlotter.label_style)

        # window dressing
        axs.set_xlabel(r'mismatch 1 position $b_1$',
                       **SearcherPlotter.label_style)
        axs.set_xticks(np.arange(0, 20))
        axs.set_xticklabels(['1'] + 3 * [''] + ['5'] + 4 * [''] +
                            ['10'] + 4 * [''] + ['15'] + 4 * [''] + ['20'])

        axs.set_ylabel(r'mismatch 2 position $b_2$',
                       **SearcherPlotter.label_style)
        axs.set_yticks(np.arange(0, 20))
        axs.set_yticklabels(['1'] + 3 * [''] + ['5'] + 4 * [''] +
                            ['10'] + 4 * [''] + ['15'] + 4 * [''] + ['20'])

        partial_cost = (np.sum(df_subset['weight'] *
                               np.log10(df_subset['simulation'] /
                                        df_subset['value']) ** 2))
        axs.set_title(f'double mm cost: {partial_cost:.2e}', pad=18)

        return axs, cax

    @staticmethod
    def plot_fit_correlation(data_df, experiment_name,
                             color=None, axs=None):
        ylabel = ''
        if color is None:
            if experiment_name.lower() == 'nucleaseq':
                color = 'orange'
                ylabel = r'$k_{clv} ($s$^{-1})$'
            elif experiment_name.lower() == 'champ':
                color = 'tab:blue'
                ylabel = r'$K_A ($nM$^{-1})$'
            else:
                color = 'tab:blue'

        df_subset = data_df.loc[
            (data_df.experiment_name == experiment_name),
            ['mismatch_number', 'value', 'error', 'simulation']
        ]

        if axs is None:
            _, axs = plt.subplots(1, 1)

        axs.scatter(
            df_subset.loc[df_subset.mismatch_number == 0, 'value'],
            df_subset.loc[df_subset.mismatch_number == 0, 'simulation'],
            marker='o', facecolor='none', edgecolor=color, alpha=0.7,
            s=24, label='on-target'
        )
        axs.scatter(
            df_subset.loc[df_subset.mismatch_number == 1, 'value'],
            df_subset.loc[df_subset.mismatch_number == 1, 'simulation'],
            marker='s', facecolor='none', edgecolor=color, alpha=0.7,
            s=24, label='off-target, 1 mm'
        )
        axs.scatter(
            df_subset.loc[df_subset.mismatch_number == 2, 'value'],
            df_subset.loc[df_subset.mismatch_number == 2, 'simulation'],
            marker='^', facecolor='none', edgecolor=color, alpha=0.7,
            s=24, label='off-target, 2 mm'
        )

        # window dressing
        axs.set_xscale('log')
        axs.set_xlabel('data - ' + ylabel, **SearcherPlotter.label_style)
        axs.set_yscale('log')
        axs.set_ylabel('model - ' + ylabel, **SearcherPlotter.label_style)

        # limits
        axs.set_aspect('equal', adjustable='box')
        minlim = min(axs.get_xlim()[0], axs.get_ylim()[0])
        extramaxlim = .11 if experiment_name == 'Champ' else 0.
        maxlim = max(axs.get_xlim()[1], axs.get_ylim()[1], extramaxlim)
        axs.set_xlim(minlim, maxlim)
        axs.set_ylim(minlim, maxlim)
        axs.plot([0, 1], [0, 1], '--k', transform=axs.transAxes,
                 linewidth=1, zorder=0)
        axs.minorticks_on()

        # ticks
        ticks = 10. ** np.arange(np.ceil(np.log10(minlim)),
                                 np.ceil(np.log10(maxlim)),
                                 1)
        axs.set_xticks(ticks)
        axs.set_yticks(ticks)

        axs.legend(loc='upper left', handlelength=1.)

        #TODO: display correlation coefficient in title

        return axs

    def make_fit_dashboard(self, training_set,
                           experiment_names: Union[str, list]):
        data_df = self.compare_to_data(training_set)

        if type(experiment_names) == str:
            experiment_names = [experiment_names]

        fig = plt.figure(
            figsize=(12, 8),
            constrained_layout=True,
        )
        grid = fig.add_gridspec(ncols=3, nrows=1+len(experiment_names),
                                width_ratios=[1, 1.12, 1],
                                height_ratios=([0.2] +
                                               len(experiment_names) * [1]),
                                hspace=.05, wspace=.08)

        axs = []
        for i in range(len(experiment_names)):
            axs += [
                fig.add_subplot(grid[i+1, 0]),
                fig.add_subplot(grid[i+1, 1]),
                fig.add_subplot(grid[i+1, 2])
            ]

            axs[3 * i] = self.plot_single_mm_fit(data_df,
                                                 experiment_names[i],
                                                 axs=axs[3 * i])
            axs[3 * i + 1] = self.plot_double_mm_fit(data_df,
                                                     experiment_names[i],
                                                     axs=axs[3 * i + 1])
            axs[3 * i + 2] = self.plot_fit_correlation(data_df,
                                                       experiment_names[i],
                                                       axs=axs[3 * i + 2])

        return fig, axs


class DashboardVideo:
    """
    Class to create videos from multiple optimization runs.

    Attributes
    ----------
    log_files: list
        List of paths to optimization run log files

    Methods
    -------
    make_video()
        Creates matplotlib.Animation instance with the log dashboard
        showing the dynamics of the optimization run
    save_video(video_path)
        Saves animation as .mp4-file (this takes quite long).

    """
    default_color_list = ['#5dd39e', '#348aa7']  # blues/greens
    # what about ['#6495ED', '#9CD08F'] ?

    default_alpha = .55

    dashboard_specs = LogAnalyzer.dashboard_specs

    # might want to write out the dpi and fps here

    def __init__(self, log_files: list):
        self.log_files = log_files

        self.analyzers = [LogAnalyzer(filename) for filename in log_files]
        self.analyzers.sort(key=lambda analyzer: analyzer.final_cost)

        self.total_step_no = max(a.total_steps for a in self.analyzers)
        self.landscape_lims, self.mismatch_lims, self.rates_lims = \
            self.get_plot_limits()

        self.fig, self.axs, self.lines, self.video = None, None, None, None

    def get_plot_limits(self):
        """Finds the upper and lower limits of all runs."""
        landscape_lims = (float('inf'), -float('inf'))
        mismatch_lims = (float('inf'), -float('inf'))
        rates_lims = (float('inf'), -float('inf'))

        for analyzer in self.analyzers:
            new_landscape_lims, new_mismatch_lims, new_rates_lims = \
                analyzer.get_plot_limits()

            # This code is horrible
            if new_landscape_lims[0] < landscape_lims[0]:
                landscape_lims = (new_landscape_lims[0], landscape_lims[1])
            if new_landscape_lims[1] > landscape_lims[1]:
                landscape_lims = (landscape_lims[0], new_landscape_lims[1])
            if new_mismatch_lims[0] < mismatch_lims[0]:
                mismatch_lims = (new_mismatch_lims[0], mismatch_lims[1])
            if new_mismatch_lims[1] > mismatch_lims[1]:
                mismatch_lims = (mismatch_lims[0], new_mismatch_lims[1])
            if new_rates_lims[0] < rates_lims[0]:
                rates_lims = (new_rates_lims[0], rates_lims[1])
            if new_rates_lims[1] > rates_lims[1]:
                rates_lims = (rates_lims[0], new_rates_lims[1])

        return landscape_lims, mismatch_lims, rates_lims

    def get_max_step_no(self):
        return max([analyzer.total_steps for analyzer in self.analyzers])

    @staticmethod
    def make_color_map(colors: list, nodes=None):
        if nodes is None:
            nodes = np.linspace(0, 1, len(colors))
        cmap = LinearSegmentedColormap.from_list(
            'dashboard_cmap',
            list(zip(nodes, colors))
        )
        return cmap

    @classmethod
    def get_color_list(cls, length: int, colors: list = None):
        if colors is None:
            colors = cls.default_color_list
        cmap = cls.make_color_map(colors)
        color_list = [to_hex(cmap(x))
                      for x in np.linspace(0, 1, length)]
        return color_list

    def init_video(self):
        fig, axs = self.analyzers[0].prepare_log_dashboard_canvas()
        axs[3] = self.analyzers[0].prepare_opt_path_canvas(
            x_lims=(0, self.get_max_step_no()),
            axs=axs[3]
        )

        lines = []

        color_list = self.get_color_list(length=len(self.analyzers))

        for k in range(len(self.analyzers)):
            lines += self.analyzers[k].prepare_log_dashboard_line(
                axs,
                color=color_list[k],
                **{'alpha': self.default_alpha}
            )

        at = AnchoredText('', frameon=False, loc='upper right',
                          prop={'size': 10,
                                'horizontalalignment': 'right'})
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axs[3].add_artist(at)
        lines += [at]

        return fig, axs, lines

    def make_frame(self, i, skipframes=1):
        j = int(i * skipframes)
        blit = True if skipframes == 1 else False

        for k in reversed(range(len(self.analyzers))):
            self.analyzers[k].update_log_dashboard_line(
                self.lines[6 * k:6 * (k + 1)],
                j,
                blit=blit
            )
        # update label
        self.lines[-1].txt.set_text(f'Step no. {j:d}')
        return self.lines

    def make_video(self, fps=None, skipframes=1):

        if fps is None:
            fps = self.dashboard_specs['fps']

        # initializing the video
        self.fig, self.axs, self.lines = self.init_video()

        # making the video
        self.video = animation.FuncAnimation(
            fig=self.fig,
            func=lambda i: self.make_frame(i, skipframes),
            frames=(self.total_step_no // skipframes + 1),
            interval=1000 / fps,
            blit=True
        )
        return self.video

    def save_video(self, video_path,
                   video_writer: callable = animation.FFMpegWriter,
                   fps=None, skipframes=1):

        if fps is None:
            fps = self.dashboard_specs['fps']

        if video_path[:-4] != '.mp4':
            video_path += '.mp4'

        # alternative writer: animation.FFMpegFileWriter
        # more on https://matplotlib.org/stable/api/animation_api.html
        writer = video_writer(fps=fps)

        self.video.save(video_path, writer=writer,
                        dpi=self.dashboard_specs['dpi'])


class RunAnalyzer:
    """
    Analyzes all the results of an optimization run. A RunAnalyzer
    object is essentially a collection of LogAnalyzer objects that has
    a number of methods to do quick analyses on them.

    Attributes
    ----------
    analyzers: list of LogAnalyzer
        The LogAnalyzer objects based on the log.txt-files in the
        job directory.

    Methods
    -------
    summarize()
        Makes a dataframe that summarizes all run information
    display_top_dashboard()
        Makes a dashboard figure of the best runs in the job.
    display_full_dashboard()
        Makes a dashboard figure of all the runs in the job.
    make_cost_histogram()
        Makes a histogram figure of the final costs of all runs.
    animate_dashboard()
        Makes an animated dashboard of all the runs or the best runs
        in the job. Can also save a mp4-video.

    """

    dashboard_specs = LogAnalyzer.dashboard_specs
    default_alpha = DashboardVideo.default_alpha
    default_color_list = DashboardVideo.default_color_list

    def __init__(self, job_dirs: Union[list, str]):
        """
        Constructor

        Parameters
        ----------
        job_dirs: list or str
            Multiple or one job directories that contain log.txt-files.
            Job directories are what is being produced by the cluster
            script, i.e. '20220215_471745'.
        """

        # get analyzers
        self.job_ids = []
        self.run_ids = []
        self.analyzers = []
        self.log_list = []

        if type(job_dirs) == str:
            job_dirs = [job_dirs]

        for path in job_dirs:
            for root, _, files in os.walk(path):
                if 'log.txt' in files:
                    self.job_ids += [path[-15:]]
                    self.run_ids += [int(root[-3:])]
                    self.analyzers += [
                        LogAnalyzer(os.path.join(root, 'log.txt'))]
                    self.log_list += [os.path.join(root, 'log.txt')]

        self.run_no = len(self.analyzers)

    def summarize(self) -> pd.DataFrame:
        """Makes a dataframe that summarizes all run information"""
        summary = pd.DataFrame(
            data={
                'Job id': self.job_ids,
                'Run id': self.run_ids,
                'Final cost': [a.final_cost for a in self.analyzers],
                'Total evals': [a.total_steps for a in self.analyzers],
                'Runtime': [a.runtime for a in self.analyzers],
                'Exit message': [a.exit_message for a in self.analyzers]
            })
        return summary

    def display_top_dashboard(self, top=1):
        """Makes a dashboard figure of the best runs in the job. By
        default, it displays a dashboard for the single best run."""

        sorted_analyzers = sorted(self.analyzers,
                                  key=lambda analyzer: analyzer.final_cost)

        fig, axs = sorted_analyzers[0].prepare_log_dashboard_canvas()

        total_steps = max([analyzer.total_steps
                           for analyzer in sorted_analyzers[:top]])
        axs[3] = sorted_analyzers[0].prepare_opt_path_canvas(
            x_lims=(0, total_steps),
            axs=axs[3]
        )

        color_list = DashboardVideo.get_color_list(length=top)

        lines = []
        for k in range(top):
            lines += sorted_analyzers[k].prepare_log_dashboard_line(
                axs,
                color=color_list[k],
                **{'alpha': self.default_alpha}
            )

            sorted_analyzers[k].update_log_dashboard_line(
                lines[6 * k:6 * (k + 1)],
                total_steps,
                blit=False
            )

        # remove grey line and dot from opt path plot
        for dump_index in sorted((list(range(4, 6 * top, 6)) +
                                  list(range(5, 6 * top, 6))),
                                 reverse=True):
            dump_line = lines.pop(dump_index)
            dump_line.remove()

        return fig, axs, lines

    def display_full_dashboard(self):
        """Makes a dashboard figure of all the runs in the job."""
        fig, axs, lines = self.display_top_dashboard(top=self.run_no)
        return fig, axs, lines

    def make_cost_histogram(self):
        """Makes a histogram figure of the final costs of all runs."""

        costs = [a.final_cost for a in self.analyzers]
        bins = 10. ** np.arange(
            min(-4, np.floor(np.log10(min(costs)))),
            max(0, np.ceil(np.log10(max(costs)))) + .5,
            .5
        )

        fig, axs = plt.subplots(1, 1, figsize=(3, 4))
        axs.set_xscale('log')
        axs.hist(costs, bins=bins)
        axs.set_xlabel('Final cost (A.U.)', **SearcherPlotter.label_style)
        axs.set_ylabel('Run count', **SearcherPlotter.label_style)
        return fig, axs

    def animate_dashboard(self, top: int = None,
                          save_path: str = '',
                          skipframes: int = 1, fps: int = None):
        """
        Makes an animated dashboard of all the runs or the best runs
        in the job. Can also save a mp4-video.

        Parameters
        ----------
        top: int
            The number of (best) runs to show results of
        save_path: str
            Location where to store a .mp4-video. If not provided,
            no video will be saved (this is the default).
        skipframes: int
            Number of frames to skip. Set these to 100 / 1000 to speed
            up the animation process for long optimization runs.
        fps: int
            Framerate (1/s). Default follows from the DashboardVideo
            class variable dashboard_specs.
        """

        # find the log paths associated with top solutions
        if top is not None:
            d = dict(
                zip(self.log_list, [a.final_cost for a in self.analyzers]))
            log_list = list(dict(sorted(d.items(), key=lambda item: item[1]))
                            .keys())[:top]
        else:
            log_list = self.log_list

        videomaker = DashboardVideo(log_list)

        if fps is None:
            fps = DashboardVideo.dashboard_specs['fps']

        videomaker.make_video(fps=fps, skipframes=skipframes)

        if save_path != '':
            videomaker.save_video(video_path=save_path, fps=fps)

        return videomaker.video
