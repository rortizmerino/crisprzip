import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib import animation
import seaborn as sns
from matplotlib.offsetbox import AnchoredText

from model.hybridization_kinetics import Searcher, SearcherPlotter


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

        self.final_cost = self.get_final_cost()
        self.final_result = self.get_final_result()

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

    def get_final_cost(self):
        """Get final cost from log file"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()

            for line in log_lines:
                if line[:10] == 'Final cost':
                    final_cost = float(line[-21:])
                    return final_cost

    def get_total_steps(self):
        """Get final cost from log file"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()

            for line in log_lines:
                if line[:17] == 'Total step number':
                    total_steps = int(line[-21:])
                    return total_steps

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

    # def get_min_evals(self) -> np.ndarray:
    #     """Outdated. Returns an array of the lowest costs"""
    #     return np.minimum.accumulate(self.log_dataframe['Cost'].to_numpy())

    def get_lowest_cost(self) -> np.ndarray:
        lowest_cost = np.zeros(shape=self.total_steps)
        for row in self.log_dataframe.iterrows():
            lowest_cost[int(row[1]['Cycle no']):] = row[1]['Cost']
        return lowest_cost

    # def find_best_searchers(self) -> list:
    #     """Returns a list of the best searchers (corresponding to
    #     the lowest cost array)"""
    #
    #     best_searchers = []
    #     best_searcher_so_far = self.log_searchers[0]
    #
    #     for i in self.log_dataframe.index:
    #         if self.log_dataframe.loc[i, 'Cost gain'] < 0:
    #             best_searcher_so_far = self.log_searchers[i]
    #         best_searchers += [best_searcher_so_far]
    #
    #     return best_searchers

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
                                 # copies marker size from SearcherPlotter
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
        lines[0].set_data(np.arange(0, i+1), self.lowest_cost[:i+1])
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

    default_alpha = .75

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

    @staticmethod
    def make_color_map(colors: list, nodes=None):
        if nodes is None:
            nodes = np.linspace(0, 1, len(colors))
        cmap = LinearSegmentedColormap.from_list(
            'dashboard_cmap',
            list(zip(nodes, colors))
        )
        return cmap

    def get_color_list(self, colors: list = None):
        if colors is None:
            colors = self.default_color_list
        cmap = self.make_color_map(colors)
        color_list = [to_hex(cmap(x))
                      for x in np.linspace(0, 1, len(self.analyzers))]
        return color_list

    def init_video(self):
        fig, axs = self.analyzers[0].prepare_log_dashboard_canvas()
        lines = []

        color_list = self.get_color_list()

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

    def make_frame(self, i):
        for k in reversed(range(len(self.analyzers))):
            self.analyzers[k].update_log_dashboard_line(
                self.lines[6 * k:6 * (k + 1)],
                i,
                blit=True
            )
        # update label
        self.lines[-1].txt.set_text(f'Step no. {i:d}')
        return self.lines

    def make_video(self, fps=None):

        if fps is None:
            fps = self.dashboard_specs['fps']

        # initializing the video
        self.fig, self.axs, self.lines = self.init_video()

        # making the video
        self.video = animation.FuncAnimation(
            fig=self.fig,
            func=self.make_frame,
            # fargs=(lines,),
            frames=self.total_step_no,
            interval=1 / fps,
            blit=True
        )
        return self.video

    def save_video(self, video_path,
                   video_writer: callable = animation.FFMpegWriter,
                   fps=None):

        if fps is None:
            fps = self.dashboard_specs['fps']

        if video_path[:-4] != '.mp4':
            video_path += '.mp4'

        # alternative writer: animation.FFMpegFileWriter
        # more on https://matplotlib.org/stable/api/animation_api.html
        writer = video_writer(fps=fps)

        self.video.save(video_path, writer=writer,
                        dpi=self.dashboard_specs['dpi'])
