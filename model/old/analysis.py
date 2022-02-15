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
    total_cycles: int
        Total number of cycles in optimization run
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
        self.total_cycles = self.log_dataframe.index[-1]
        self.log_searchers = self.make_searcher_series()
        self.final_cost = self.get_final_cost()
        self.landscape_lims, self.mismatch_lims, self.rates_lims = \
            self.get_plot_limits()

    def read_log_file(self):
        """Returns a pd.DataFrame with the log content"""
        with open(self.log_file, 'r') as log_reader:
            log_lines = log_reader.readlines()
            file_length = 0
            for line in log_lines:
                file_length += 1
                last_line = line
        cycle_no = int(last_line.split('\t')[0])

        dataframe = pd.read_table(self.log_file,
                                  skiprows=(file_length - cycle_no - 2),
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
        if 'U_PAM' in landscape_df.columns:
            pam_sensing = True
        else:
            pam_sensing = False

        searcher_series = pd.Series(dtype=object)
        for i in landscape_df.index:
            param_vector = landscape_df.iloc[:, 5:].loc[i].to_numpy()
            guide_length = (param_vector.size - pam_sensing - 3) // 2

            searcher_series.loc[i] = Searcher(
                on_target_landscape=param_vector[:guide_length + pam_sensing],
                mismatch_penalties=param_vector[guide_length + pam_sensing:-3],
                forward_rates={
                    'k_on': 10 ** param_vector[-3],
                    'k_f': 10 ** param_vector[-2],
                    'k_clv': 10 ** param_vector[-1]
                },
                pam_detection=pam_sensing
            )
        return searcher_series

    @staticmethod
    def map_temp(temp):
        """Function to map %-temperature Series to x-axis values"""
        return 100 / temp

    def prepare_opt_path_canvas(self, x_lims=None, y_lims=None,
                                title='Optimization path', axs=None):
        """Creates or adapts Figure and Axes objects such
        that an optimization path line can later be added."""

        opt_table = self.log_dataframe[['Cycle no', 'Cost', 'Temp (%)']]
        if axs is None:
            _, axs = plt.subplots(1, 1, figsize=(3, 3))

        # x axis
        axs.set_xscale('log')
        if x_lims is None:
            x_lims = (min(self.map_temp(opt_table['Temp (%)'])),
                      max(self.map_temp(opt_table['Temp (%)'])))
        # calculate 10% overhang at both sides
        x_overhang = (x_lims[1] / x_lims[0]) ** .1
        axs.set_xlim(x_lims[0] / x_overhang,
                     x_lims[1] * x_overhang)

        axs.set_xlabel(r'Coldness $\beta$ (A.U.)',
                       **SearcherPlotter.label_style)

        # y axis
        axs.set_yscale('log')
        if y_lims is None:
            y_lims = (min(opt_table['Cost']),
                      max(opt_table['Cost']))
        # calculate 10% overhang at both sides
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
        lines[0].set_data(
            self.map_temp(self.log_dataframe.loc[:i, 'Temp (%)']),
            self.log_dataframe.loc[:i, 'Cost']
        )
        # current line (point)
        lines[1].set_data(
            self.map_temp(self.log_dataframe.loc[i, 'Temp (%)']),
            self.log_dataframe.loc[i, 'Cost']
        )
        # future line
        lines[2].set_data(
            self.map_temp(self.log_dataframe.loc[i:, 'Temp (%)']),
            self.log_dataframe.loc[i:, 'Cost']
        )

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
        lines[0].set_data(
            self.map_temp(self.log_dataframe['Temp (%)']),
            self.log_dataframe['Cost']
        )
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
                                        title='Forward_rates',
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

    def update_log_dashboard_line(self, lines, i):
        """Updates log dashboard up to data point i"""
        i = min(i, self.total_cycles)
        plotter = SearcherPlotter(self.log_searchers[i])
        plotter.update_on_target_landscape(lines[0])
        plotter.update_mismatches(lines[1])
        plotter.update_rates(lines[2])
        self.update_partial_opt_path(lines[3:6], i)

    def plot_full_log_dashboard(self, color):
        """Creates full log dashboard, with the final parameter state"""
        fig, axs = self.prepare_log_dashboard_canvas()
        lines = self.prepare_log_dashboard_line(axs, color=color)
        self.update_log_dashboard_line(lines, self.total_cycles)
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

        self.total_cycle_no = max(a.total_cycles for a in self.analyzers)
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
                i
            )
        # update label
        self.lines[-1].txt.set_text(f'Cycle no. {i:d}')
        return self.lines

    def make_video(self):

        # initializing the video
        self.fig, self.axs, self.lines = self.init_video()

        # making the video
        self.video = animation.FuncAnimation(
            fig=self.fig,
            func=self.make_frame,
            # fargs=(lines,),
            frames=self.total_cycle_no + 1,
            interval=1 / self.dashboard_specs['fps'],
            blit=True
        )
        return self.video

    def save_video(self, video_path,
                   video_writer: callable = animation.FFMpegWriter):

        if video_path[:-4] != '.mp4':
            video_path += '.mp4'

        # alternative writer: animation.FFMpegFileWriter
        # more on https://matplotlib.org/stable/api/animation_api.html
        writer = video_writer(fps=self.dashboard_specs['fps'])

        self.video.save(video_path, writer=writer,
                        dpi=self.dashboard_specs['dpi'])


'''
PREVIOUS VERSION OF THE VIDEO MAKER

    def make_log_dashboard_video(self, video_name, video_path='',
                                 skip_cycles=1, color_list=None, alpha=.75):

        # non-interactive backend for png-production
        matplotlib.use('Agg')

        if video_name[-4:] != '.mp4':
            video_name += '.mp4'

        if video_path != '' and not os.path.exists(video_path):
            os.makedirs(video_path)

        frames = list(range(0, self.total_cycle_no + 1, skip_cycles))
        if self.total_cycle_no not in frames:
            frames += [self.total_cycle_no]

        # png for high quality, jpg for high speed
        img_format = '.png'

        img_array = []
        for i in frames:
            frame_path = os.path.join(video_path, '/temp',
                                      'frame{i:09d}{img_format}')

            # save frame
            _ = self.make_log_dashboard_frame(i, color_list=color_list,
                                              alpha=alpha)
            plt.savefig(frame_path, dpi=self.dashboard_specs['dpi'])
            plt.close()

            # store frame in img_array
            img = cv2.imread(frame_path)
            img_array.append(img)

        height, width, layers = img_array[-1].shape
        size = (width, height)

        # initialize mp4 video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(video_path, video_name),
                              fourcc, self.dashboard_specs['fps'], size)

        # add images to video and remove files
        for i in range(len(img_array)):
            out.write(img_array[i])
            os.remove(os.path.join(video_path, '/temp',
                                   f'frame{frames[i]:09d}.png'))
        out.release()
'''
