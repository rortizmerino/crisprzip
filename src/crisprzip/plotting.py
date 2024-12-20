import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from czmodel.kinetics import *


class SearcherPlotter:
    """Base class that underlies plotting functionality in the Searcher
    and SearcherTargetComplex classes of this module, and the animation
    functionality of the LogAnalyzer and DashboardVideo classes of the
    analysis module. A user should not directly interact with this
    class.

    This class is built up to support 'blitting': redrawing only the
    parts of a figure that are updated. This significantly speeds up
    the video making process in the analysis module. It contains 4
    types of methods:
    1) prepare ... canvas  creates or adapts Figure and Axes objects
                           such that data can later be added
    2) prepare ... lines   adds properly styled lines to the canvas
                           that do not yet contain data (the term line
                           here includes dots or other shapes)
    3) update ...          adds data to the pre-configured line
    4) plot ...            combines the above 3 methods to make a
                           complete plot
    """

    # Some general style definitions
    title_style = {'fontweight': 'bold'}
    label_style = {'fontsize': 10}
    tick_style = {'labelsize': 10}
    line_style = {'linewidth': 2,
                  'marker': '.',
                  'markersize': 12}

    def __init__(self, searcher: Searcher):
        self.searcher = searcher

    def prepare_landscape_canvas(self, y_lims: tuple = None, title: str = '',
                                 axs: Axes = None) -> Axes:
        """Creates or adapts Figure and Axes objects such
        that an on-/off-target/penalties landscape line can later be
        added."""
        searcher = self.searcher
        if axs is None:
            _, axs = plt.subplots(1, 1, figsize=(4, 3))

        # x axis
        axs.set_xlim(-(searcher.pam_detection + 1.2),
                     searcher.guide_length + 2.2)
        axs.set_xticks(np.arange(-searcher.pam_detection,
                                 searcher.guide_length + 2))
        x_tick_labels = (
                ['S'] + searcher.pam_detection * ['P'] + ['1'] +
                ['' + (x % 5 == 0) * str(x) for x in
                 range(2, searcher.guide_length)] +
                [str(searcher.guide_length)] + ['C']
        )
        axs.set_xticklabels(x_tick_labels, rotation=0)
        axs.set_xlabel(r'Targeting progression $b$', **self.label_style)

        # y axis
        if y_lims is None:
            y_lims = (min(searcher.on_target_landscape.min(), 0),
                      searcher.on_target_landscape.max())
        axs.set_ylim(y_lims[0] - .5, y_lims[1] + .5)
        axs.set_ylabel(r'Free energy ($k_BT$)', **self.label_style)

        # title
        axs.set_title(title, **self.title_style)

        # style
        axs.tick_params(axis='both', **self.tick_style)
        axs.grid(True)
        return axs

    def prepare_rates_canvas(self, y_lims: tuple = None,
                             title: str = 'Transition rates',
                             axs: Axes = None,
                             extra_rates: dict = None) -> Axes:
        """Creates or adapts Figure and Axes objects such
        that rates points can later be added."""
        searcher = self.searcher
        if axs is None:
            _, axs = plt.subplots(1, 1, figsize=(3, 3))

        if extra_rates is not None:
            extra_labels = list(extra_rates.keys())
            extra_values = list(extra_rates.values())
        else:
            extra_labels = []
            extra_values = []

        # x axis
        axs.set_xlim(-.5, 2.5 + len(extra_labels))
        axs.set_xlabel(' ')
        axs.set_xticks(np.arange(3 + len(extra_labels)))
        x_tick_labels = (extra_labels +
                         [r'${k_{off}}$', r'${k_{f}}$', r'${k_{clv}}$'])
        axs.set_xticklabels(x_tick_labels, rotation=0)

        # y axis
        axs.set_yscale('log')
        if y_lims is None:
            all_rates = list(searcher.internal_rates.values()) + extra_values
            y_lims = (min(all_rates), max(all_rates))
        axs.set_ylim(y_lims[0] * 10 ** -.5, y_lims[1] * 10 ** .5)
        axs.set_ylabel(r'Rate (${s^{-1}})$', **self.label_style)

        # title
        axs.set_title(title, **self.title_style)

        # background
        axs.tick_params(axis='both', **self.tick_style)
        axs.tick_params(axis='x', labelsize=9)
        return axs

    def prepare_landscape_line(self, axs: Axes, color='cornflowerblue',
                               **plot_kwargs) -> Line2D:
        """Adds styled lines to the landscape canvas"""
        line, = axs.plot([], [], color=color,
                         **self.line_style, **plot_kwargs)
        return line

    def prepare_rates_line(self, axs: Axes, color='cornflowerblue',
                           **plot_kwargs) -> Line2D:
        """"Adds styled lines (points) to the rates canvas"""
        plot_kwargs['linestyle'] = ''
        line = self.prepare_landscape_line(axs, color=color, **plot_kwargs)
        return line

    def update_on_target_landscape(self, line: Line2D) -> None:
        """Updates landscape line to represent on-target landscape"""
        searcher = self.searcher
        line.set_data(
            np.arange(1 - searcher.pam_detection, searcher.guide_length + 2),
            np.concatenate(
                (np.zeros(1),  # pam / 1st state
                 searcher.on_target_landscape,
                 np.ones(1) * line.axes.get_ylim()[0])
            )
        )

    def update_solution_energies(self, lines: list, on_rates: list) -> None:
        """Updates the free energy level of the solution state(s)"""
        searcher = self.searcher
        for line, on_rate in zip(lines, on_rates):
            line.set_data(
                np.arange(-searcher.pam_detection, 1),
                np.array([searcher.calculate_solution_energy(on_rate), 0])
            )

    def update_mismatches(self, line: Line2D) -> None:
        """Updates landscape line to represent mismatches"""
        searcher = self.searcher
        line.set_data(
            np.arange(1, searcher.guide_length + 1),
            searcher.mismatch_penalties
        )

    def update_rates(self, line: Line2D, extra_rates: dict = None) -> None:
        """Updates rate points to represent forward rates"""
        searcher = self.searcher

        if extra_rates is not None:
            extra_values = list(extra_rates.values())
        else:
            extra_values = []

        forward_rates = extra_values + list(searcher.internal_rates.values())
        line.set_data(
            list(range(3+len(extra_values))),
            forward_rates
        )

    def plot_on_target_landscape(self, y_lims: tuple = None,
                                 color='cornflowerblue', axs: Axes = None,
                                 on_rates: list = None, **plot_kwargs) -> Axes:
        """Creates complete on-target landscape plot"""

        if y_lims is None and on_rates is not None:
            searcher = self.searcher
            solution_energies = [searcher.calculate_solution_energy(k_on)
                                 for k_on in on_rates]
            y_lims = (min(list(searcher.on_target_landscape) +
                          solution_energies + [0]),
                      max(list(searcher.on_target_landscape) +
                          solution_energies))

        axs = self.prepare_landscape_canvas(
            y_lims,
            title='On-target landscape',
            axs=axs
        )
        line = self.prepare_landscape_line(
            axs,
            color=color,
            **plot_kwargs
        )
        self.update_on_target_landscape(line)

        if on_rates is not None:
            solution_lines = []
            for _ in on_rates:
                solution_lines += [self.prepare_landscape_line(
                    axs, color=color, linestyle='dashed', **plot_kwargs
                )]
            self.update_solution_energies(solution_lines, on_rates)

        return axs

    def plot_off_target_landscape(self, mismatch_positions: MismatchPattern,
                                  y_lims: tuple = None,
                                  color='firebrick', axs: Axes = None,
                                  on_rates: list = None,
                                  **plot_kwargs) -> Axes:
        """Creates complete off-target landscape plot, based on the
        mismatch positions array"""

        searcher = self.searcher.probe_target(mismatch_positions)
        if y_lims is None and on_rates is not None:
            if on_rates is None:
                solution_energies = []
            else:
                solution_energies = [searcher.calculate_solution_energy(k_on)
                                     for k_on in on_rates]
            y_lims = (min(list(searcher.on_target_landscape) +
                          solution_energies + [0]),
                      max(list(searcher.off_target_landscape) +
                          solution_energies))

        axs = self.prepare_landscape_canvas(
            y_lims,
            title='Off-target landscape',
            axs=axs
        )
        lines = [
            self.prepare_landscape_line(axs, color='darkgray', **plot_kwargs),
            self.prepare_landscape_line(axs, color=color, **plot_kwargs)
        ]
        self.update_on_target_landscape(lines[0])
        lines[1].set_data(
            np.arange(1 - searcher.pam_detection, searcher.guide_length + 2),
            np.concatenate(
                (np.zeros(1),  # solution state
                 searcher.off_target_landscape,
                 np.ones(1) * lines[1].axes.get_ylim()[0])  # cleaved state
            )
        )

        if on_rates is not None:
            solution_lines = []
            for _ in on_rates:
                solution_lines += [self.prepare_landscape_line(
                    axs, color=color, linestyle='dashed', **plot_kwargs
                )]
            self.update_solution_energies(solution_lines, on_rates)

        return axs

    def plot_mismatch_penalties(self, y_lims: tuple = None,
                                color='firebrick', axs: Axes = None,
                                **plot_kwargs) -> Axes:
        """Creates complete mismatch landscape plot"""
        axs = self.prepare_landscape_canvas(
            y_lims,
            title='Mismatch penalties',
            axs=axs
        )
        line = self.prepare_landscape_line(axs, color=color, **plot_kwargs)
        self.update_mismatches(line)
        return axs

    def plot_internal_rates(self, y_lims: tuple = None,
                            color='cornflowerblue', axs: Axes = None,
                            extra_rates: dict = None, **plot_kwargs) -> Axes:
        """Creates complete forward rates plot"""
        axs = self.prepare_rates_canvas(
            y_lims,
            title='Transition rates',
            axs=axs,
            extra_rates=extra_rates
        )
        line = self.prepare_rates_line(axs, color=color, **plot_kwargs)
        self.update_rates(line, extra_rates=extra_rates)
        return axs
