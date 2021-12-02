import os

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

from hybridization_kinetics import Searcher, SearcherPlotter


# TODO: make this class for multiple logs
class MultipleLogAnalyzer:
    pass


class LogAnalyzer:

    dashboard_specs = {
        'size': (8, 6),  # in inch
        'dpi': 150,  # dots per inch
        'fps': 15  # frames per second
    }

    def __init__(self, log_file):
        self.log_file = log_file
        self.log_dataframe = self.read_log_file()
        self.log_searchers = self.make_searcher_series()
        self.landscape_lims, self.mismatch_lims, self.rates_lims =\
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
                    'k_on': param_vector[-3],
                    'k_f': param_vector[-2],
                    'k_clv': param_vector[-1]
                },
                pam_detection=pam_sensing
            )
        return searcher_series

    @staticmethod
    def map_temp(temp):
        """Function to map %-temperature Series to x-axis values"""
        return 100 / temp

    def plot_opt_path_upto(self, i, x_lims=None, y_lims=None,
                           color='cornflowerblue', axs=None):

        opt_table = self.log_dataframe[['Cycle no', 'Avg cost', 'Temp (%)']]

        if axs is None:
            _, axs = plt.subplots(1, 1, figsize=(3, 3))

        axs.plot(
            self.map_temp(opt_table.loc[:i, 'Temp (%)']),
            opt_table.loc[:i, 'Avg cost'],
            color=color, linewidth=2
        )

        # window dressing
        axs.set_xlabel(r'Coldness $\beta$ (A.U.)',
                       **SearcherPlotter.label_style)
        axs.set_ylabel(r'Potential $V$ (A.U.)', **SearcherPlotter.label_style)
        axs.set_title('Optimization path', **SearcherPlotter.title_style)

        axs.grid('on')
        sns.set_style('ticks')
        sns.despine(ax=axs)
        return axs

    def plot_opt_path_full(self, x_lims=None, y_lims=None,
                           color='cornflowerblue', axs=None):
        axs = self.plot_opt_path_upto(self.log_dataframe.index[-1],
                                      x_lims, y_lims, color, axs)
        return axs

    def plot_opt_path_partial(self, i, x_lims=None, y_lims=None,
                              color='cornflowerblue', axs=None,
                              cycle_label=True):
        # first, plot full opt path
        axs = self.plot_opt_path_full(x_lims, y_lims,
                                      color='darkgray', axs=axs)
        # then, add (colored) partial opt path
        axs = self.plot_opt_path_upto(i, x_lims, y_lims,
                                      color=color, axs=axs)
        axs.scatter(
            self.map_temp(self.log_dataframe.loc[i, 'Temp (%)']),
            self.log_dataframe.loc[i, 'Avg cost'],
            zorder=2.5, color=color, **SearcherPlotter.scatter_style
        )

        # add label with cycle number
        if cycle_label:
            at = AnchoredText('Cycle no.\n%d' %
                              self.log_dataframe.loc[i, 'Cycle no'],
                              frameon=False, loc='upper right',
                              prop={'size': 10,
                                    'horizontalalignment': 'right'})
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            axs.add_artist(at)

        return axs

    def make_log_dashboard_frame(self, i, color='cornflowerblue'):
        fig = plt.figure(
            figsize=self.dashboard_specs['size'],
            constrained_layout=True,
        )
        grid = fig.add_gridspec(ncols=2, nrows=2,
                                width_ratios=[2, 1],
                                height_ratios=[1, 1])

        landscape_axs = fig.add_subplot(grid[0, 0])
        mismatch_axs = fig.add_subplot(grid[1, 0])
        rate_axs = fig.add_subplot(grid[0, 1])
        opt_path_axs = fig.add_subplot(grid[1, 1])

        searcher = self.log_searchers[i]
        searcher.plot_on_target_landscape(y_lims=self.landscape_lims,
                                          color=color, axs=landscape_axs)
        searcher.plot_penalties(y_lims=self.mismatch_lims, color=color,
                                axs=mismatch_axs)
        searcher.plot_forward_rates(y_lims=self.rates_lims, color=color,
                                    axs=rate_axs)
        self.plot_opt_path_partial(i, color=color, axs=opt_path_axs)
        return fig

    def make_log_dashboard_video(self, video_name, video_path='',
                                 skip_cycles=1, color='cornflowerblue'):

        if video_name[-4:] != '.mp4':
            video_name += '.mp4'

        if video_path != '' and not os.path.exists(video_path):
            os.makedirs(video_path)

        frames = self.log_dataframe.index[::skip_cycles].to_list()
        final_frame = self.log_dataframe.index[-1]
        if final_frame not in frames:
            frames += [final_frame]

        img_array = []
        for i in frames:
            frame_path = os.path.join(video_path, f'frame{i:09d}.png')

            # save frame
            _ = self.make_log_dashboard_frame(i, color=color)
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
            os.remove(video_path + f'/frame{frames[i]:09d}.png')
        out.release()


def main():
    log_file = 'SimAnnealTestReport.txt'
    log_analyzer = LogAnalyzer(log_file)
    log_analyzer.make_log_dashboard_video('testvideo1.mp4', 'videos')


if __name__ == '__main__':
    main()
