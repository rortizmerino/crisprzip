"""
A collection of all plotting functionality in the project. Most of the
code is copied from a previous version of the model, so in general,
it is really hacky. Once we've been working with the new code architecture
for some time, it might make sense to spend time into making some kind of
plotly application with the several plots that are relevant in the
optimization procedure.

Functions:
    multiple(job_analyzer)
    plot_on_target_landscape(searcher)
    plot_off_target_landscape(searcher_target_complex)
    plot_mismatch_penalties(searcher)
    plot_internal_rates(searcher)
    plot_coarsegrained_landscape(searcher_target_complex)
    plot_coarsegrained_rates(searcher_target_complex)
    plot_optimization_path(evals_analyzer)
    plot_opt_temperature_path(opt_run_analyzer)
    make_optimization_dashboard(optrun_analyzer)
    make_optimization_video(optrun_analyzer)
    make_correlation_dashboard(optrun_analyzer)

Classes:
    SearcherPlotter
    OptPathPlotter
    OptDashboard
    DashboardVideo
    CorrelationPlot
"""

import seaborn as sns
from matplotlib import pyplot as plt, animation
from matplotlib.axes import Axes
from matplotlib.colors import to_hex, LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr

from crisprzipper.model.data import AggregateData
from crisprzipper.model.fitting import SearcherScorer
from crisprzipper.model.analysis import *


def multiple(plot_func: callable, job: JobAnalyzer,
             y_lims=None, cmap='viridis', axs=None,
             **kwargs) -> plt.Axes:
    """Loops over the analyzers in a cluster job and makes a plot for
    each of them."""

    if not isinstance(job, JobAnalyzer):
        raise ValueError(f"Searchers should be a JobAnalyzer object,"
                         f"not {type(job)}.")

    m = len(job.analyzers)
    colors = plt.get_cmap(cmap+'_r')(np.arange(m) / (m - 1))

    for i in range(m):
        axs = plot_func(
            job.analyzers[i],
            y_lims=y_lims,
            color=colors[i - 1],
            axs=axs,
            alpha=.7,
            **kwargs
        )
    return axs


def plot_on_target_landscape(searcher: Union[Searcher, ParameterVector,
                                             OptRunAnalyzer],
                             y_lims=None,
                             color='cornflowerblue',
                             axs=None, on_rates: list = None,
                             **plot_kwargs) -> plt.Axes:
    """Creates on-target landscape plot"""
    if isinstance(searcher, OptRunAnalyzer):
        searcher = searcher.get_best_pvec().to_searcher()
    elif isinstance(searcher, ParameterVector):
        searcher = searcher.to_searcher()
    elif not isinstance(searcher, Searcher):
        raise ValueError(f"Parameter 'searcher' cannot be of"
                         f"type {type(searcher)}.")

    axs = SearcherPlotter(searcher).plot_on_target_landscape(
        y_lims=y_lims,
        color=color,
        axs=axs,
        on_rates=on_rates,
        **plot_kwargs
    )
    return axs


def plot_penalties(searcher: Union[Searcher, OptRunLog],
                   y_lims=None, color='firebrick', axs=None,
                   **plot_kwargs) -> plt.Axes:
    """Creates mismatch penalties landscape plot"""
    if isinstance(searcher, OptRunAnalyzer):
        searcher = searcher.get_best_pvec().to_searcher()
    elif isinstance(searcher, ParameterVector):
        searcher = searcher.to_searcher()
    elif not isinstance(searcher, Searcher):
        raise ValueError(f"Parameter 'searcher' cannot be of"
                         f"type {type(searcher)}.")

    axs = SearcherPlotter(searcher).plot_mismatch_penalties(y_lims=y_lims,
                                                            color=color,
                                                            axs=axs,
                                                            **plot_kwargs)
    return axs


def plot_internal_rates(searcher: Searcher, y_lims=None,
                        color='cornflowerblue', axs=None,
                        extra_rates: dict = None, **plot_kwargs) -> plt.Axes:
    """Creates forward rates plot"""
    if isinstance(searcher, OptRunAnalyzer):
        searcher = searcher.get_best_pvec().to_searcher()
    elif isinstance(searcher, ParameterVector):
        searcher = searcher.to_searcher()
    elif not isinstance(searcher, Searcher):
        raise ValueError(f"Parameter 'searcher' cannot be of"
                         f"type {type(searcher)}.")

    axs = SearcherPlotter(searcher).plot_internal_rates(
        y_lims=y_lims,
        color=color,
        axs=axs,
        extra_rates=extra_rates,
        **plot_kwargs
    )
    return axs


def plot_off_target_landscape(searcher_target_complex: SearcherTargetComplex,
                              y_lims=None, color='firebrick',
                              axs=None, **plot_kwargs) -> plt.Axes:
    """Creates off-target landscape plot"""
    axs = SearcherPlotter(searcher_target_complex).plot_off_target_landscape(
        searcher_target_complex.target_mismatches,
        y_lims=y_lims, color=color, axs=axs, **plot_kwargs
    )
    return axs


def plot_optimization_path(analyzer: Union[OptRunAnalyzer, EvalsAnalyzer],
                           color='cornflowerblue', axs=None,
                           **plot_kwargs) -> plt.Axes:
    """Creates a plot of the optimization path with cost vs. time"""

    if isinstance(analyzer, OptRunAnalyzer):
        analyzer = analyzer.evals
    elif not isinstance(analyzer, EvalsAnalyzer):
        raise ValueError(f"Parameter 'analyzer' cannot be of"
                         f"type {type(analyzer)}.")

    log_summary = analyzer.summarize_log()
    obj = OptPathPlotter(analyzer.log.shape[0],
                         log_summary['cost'])
    return obj.plot_full_opt_path(color=color, axs=axs,
                                  **plot_kwargs)


def plot_opt_temperature_path(analyzer: OptRunAnalyzer, y_lims=None,
                              color='cornflowerblue', axs=None,
                              **plot_kwargs) -> plt.Axes:
    """Creates a plot of the optimization path with cost vs. temperature"""

    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(3, 3))

    log_summary = analyzer.get_run_details()
    axs.plot(
        1 / log_summary["temp"],
        log_summary["cost"], color=color, **plot_kwargs
    )
    axs.set_xscale('log')
    axs.set_xlabel(r"Coldness $\beta$ ($1/k_BT$)")
    axs.set_yscale('log')
    axs.set_ylim(y_lims)
    axs.set_ylabel(r"Potential $V$ (A.U.)")
    axs.grid("on")
    return axs


def make_optimization_dashboard(analyzer: OptRunAnalyzer,
                                color='cornflowerblue'):
    """Make full dashboard, with final landscape, penalties, rates, and
    optimization path."""
    return OptDashboard(analyzer).plot_final_log_dashboard(color)


@path_handling
def make_optimization_video(analyzer: Union[OptRunAnalyzer, JobAnalyzer],
                            fps=25, skipframes=1000,
                            savepath: Union[Path, str] = None):
    """Animate the full dashboard to show changes during the
    optimization process. If savepath is given, the animation will be
    saved (which usually takes very long)."""
    if isinstance(analyzer, JobAnalyzer):
        vidmaker = DashboardVideo(analyzer.analyzers)
    elif isinstance(analyzer, OptRunAnalyzer):
        vidmaker = DashboardVideo([analyzer])
    else:
        raise ValueError(f"Parameter 'analyzer' cannot be of"
                         f"type {type(analyzer)}.")

    anim = vidmaker.make_video(fps, skipframes)
    if savepath is not None:
        vidmaker.save_video(video_path=savepath.as_posix(), fps=fps)
    return anim


def make_correlation_dashboard(pvec: Union[ParameterVector, OptRunAnalyzer],
                               dataset: AggregateData, color='cornflowerblue',
                               axs=None) -> None:
    """Shows how the experiment simulations of a pvec relate to
    the data in a dataset"""
    if isinstance(pvec, OptRunAnalyzer):
        pvec = pvec.get_best_pvec()
    elif not isinstance(pvec, ParameterVector):
        raise ValueError(f"Parameter 'pvec' cannot be of"
                         f"type {type(pvec)}.")

    if dataset.exp_type.name == "NUCLEASEQ":
        color = 'tab:orange'
    elif dataset.exp_type.name == "CHAMP":
        color = 'tab:blue'

    CorrelationPlot(pvec, dataset).make_fit_dashboard(axs=axs,
                                                      color=color)


def make_all_correlation_dashboards() -> None:
    """Same as above, but uses the analyzer metadata to find the
    datasets and produces multiple correlation dashboards."""
    pass

# WARNING: the code below is copied and made to work with patches.
# Documentation is outdated too. It is recommended not to edit.


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
        sns.set_style('ticks')
        sns.despine(ax=axs)
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
        sns.set_style('ticks')
        sns.despine(ax=axs)
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


class OptPathPlotter:

    def __init__(self, total_steps: int, cost_series: pd.Series):
        self.total_steps = total_steps
        self.lowest_cost = self._get_lowest_cost(cost_series)

    def _get_lowest_cost(self, cost_series: pd.Series):
        lowest_cost = np.zeros(shape=self.total_steps)
        for i in cost_series.index:
            lowest_cost[i:] = cost_series[i]
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

        zorder_ref = plot_kwargs.pop("zorder", 2.)

        # colored, left-side line
        past_line, = axs.plot([], [], color=color,
                              linewidth=1, zorder=zorder_ref, **plot_kwargs)
        # colored, dot at current opt point
        current_line, = axs.plot([], [], color=color,
                                 linestyle='', zorder=zorder_ref,
                                 marker='.', markersize=12,
                                 **plot_kwargs)
        # gray, right-side line
        future_line, = axs.plot([], [], color='darkgray', linewidth=1,
                                zorder=zorder_ref - 1,
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


class OptDashboard:

    dashboard_specs = {
        'size': (8, 6),  # in inch
        'dpi': 150,  # dots per inch
        'fps': 15  # frames per second
    }

    # noinspection PyUnresolvedReferences
    def __init__(self, run: OptRunAnalyzer):

        self.run = run
        self.log_pvecs = run.get_summarized_pvecs()
        self.log_searchers = pd.Series(
            index=self.log_pvecs.index,
            data=[pvec.to_searcher() for pvec in self.log_pvecs.to_list()]
        )
        self.log_dataframe = run.evals.summarize_log()
        self.total_steps = run.result['nfev']
        self.opt_path_plotter = OptPathPlotter(self.total_steps,
                                               self.log_dataframe["cost"])
        self.landscape_lims, self.mismatch_lims, self.rates_lims =\
            self.get_plot_limits()

    def get_plot_limits(self):
        """Get upper and lower bounds from log file"""

        lower_bnd = np.array(self.run.pvec_info["lb"])
        lb_searcher = (getattr(crisprzipper.model.parameter_vector,
                               self.run.pvec_type)(lower_bnd)
                       .to_searcher())

        upper_bnd = np.array(self.run.pvec_info["ub"])
        ub_searcher = (getattr(crisprzipper.model.parameter_vector,
                               self.run.pvec_type)(upper_bnd)
                       .to_searcher())

        landscape_lims = (np.min(lb_searcher.on_target_landscape),
                          np.max(ub_searcher.on_target_landscape))
        mismatch_lims = (np.min(lb_searcher.mismatch_penalties),
                         np.max(ub_searcher.mismatch_penalties))
        rates_lims = (np.min(list(lb_searcher.internal_rates.values())),
                      np.max(list(ub_searcher.internal_rates.values())))

        return landscape_lims, mismatch_lims, rates_lims

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
                  .prepare_rates_canvas(
            self.rates_lims,
            title='Transition rates',
            axs=axs[2],
            extra_rates={
                r'$k_{on}^{NuSeq}$': 1.,
                r'$k_{on}^{Champ}$': 1.
            }))
        axs[3] = self.opt_path_plotter.prepare_opt_path_canvas(
            x_lims=None, y_lims=None,
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
                ] + (self.opt_path_plotter
                     .prepare_opt_path_lines(axs[3], color, **plot_kwargs))

        return lines

    def update_log_dashboard_line(self, lines, i, blit=False):
        """Updates log dashboard up to data point i"""

        i = min(i, self.total_steps - 1)
        self.opt_path_plotter.update_partial_opt_path(lines[3:6], i)

        # j points to the best searcher at point i (j <= i)
        j = self.log_dataframe.index[self.log_dataframe.index <= i][-1]

        if blit and j < i:
            pass  # prevents redundant line updating
        else:
            pvec = self.log_pvecs[j]
            plotter = SearcherPlotter(self.log_searchers[j])
            plotter.update_on_target_landscape(lines[0])
            plotter.update_mismatches(lines[1])
            plotter.update_rates(lines[2], extra_rates={
                r'$k_{on}^{NuSeq}$': pvec.to_binding_rate(
                    ExperimentType.NUCLEASEQ
                ),
                r'$k_{on}^{Champ}$': pvec.to_binding_rate(
                    ExperimentType.CHAMP
                )
            })

    def plot_final_log_dashboard(self, color):
        """Creates full log dashboard, with the result and the
         full optimization path"""
        fig, axs = self.prepare_log_dashboard_canvas()
        lines = self.prepare_log_dashboard_line(axs, color=color)

        best_pvec = self.run.get_best_pvec()
        plotter = SearcherPlotter(
            best_pvec.to_searcher()
        )
        plotter.update_on_target_landscape(lines[0])
        plotter.update_mismatches(lines[1])
        plotter.update_rates(lines[2], extra_rates={
            r'$k_{on}^{NuSeq}$': best_pvec.to_binding_rate(
                ExperimentType.NUCLEASEQ
            ),
            r'$k_{on}^{Champ}$': best_pvec.to_binding_rate(
                ExperimentType.CHAMP
            )
        })
        self.opt_path_plotter.update_partial_opt_path(lines[3:6],
                                                      self.total_steps - 1)

        lines[4].set_data([], [])
        if len(lines) == 7:
            lines[6].txt.set_text('')
        return fig, axs, lines


class DashboardVideo:
    """
    Class to create videos from multiple optimization runs.
    """
    default_cmap = plt.get_cmap("viridis")
    # what about ['#6495ED', '#9CD08F'] ?

    default_alpha = .55

    dashboard_specs = OptDashboard.dashboard_specs

    # might want to write out the dpi and fps here

    def __init__(self, analyzers: List[OptRunAnalyzer]):

        self.analyzers = analyzers
        self.analyzers.sort(key=lambda analyzer: analyzer.result["fun"])
        self.dashboards = [OptDashboard(analyzer)
                           for analyzer in self.analyzers]

        self.total_step_no = max(a.result["nfev"] for a in self.analyzers)
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
                OptDashboard(analyzer).get_plot_limits()

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
        return max([analyzer.result["nfev"] for analyzer in self.analyzers])

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
            cmap = cls.default_cmap
        else:
            cmap = cls.make_color_map(colors)
        color_list = [to_hex(cmap(x))
                      for x in np.linspace(0, 1, length)]
        return color_list

    def init_video(self):
        fig, axs = self.dashboards[0].prepare_log_dashboard_canvas()
        axs[3] = self.dashboards[0].opt_path_plotter.prepare_opt_path_canvas(
            x_lims=(0, self.get_max_step_no()),
            axs=axs[3]
        )

        lines = []

        color_list = self.get_color_list(length=len(self.analyzers))

        for k in range(len(self.analyzers)):
            lines += self.dashboards[k].prepare_log_dashboard_line(
                axs,
                color=color_list[k],
                **{'alpha': self.default_alpha,
                   'zorder': 3 - k/len(self.analyzers)}
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

        for k in range(len(self.analyzers)):
            self.dashboards[k].update_log_dashboard_line(
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


class CorrelationPlot:

    def __init__(self, pvec: ParameterVector, dataset: AggregateData):
        self.pvec = pvec
        self.scorer = SearcherScorer(pvec.to_searcher())
        self.dataset = dataset
        self.cost = self.scorer.compare(
            self.dataset,
            self.pvec.to_binding_rate(self.dataset.exp_type)
        )

    def compare_subset_to_data(self, mm_num: int):

        simvalues = self.scorer.run_experiments(
            self.dataset.to_mm_num_subset(mm_num),
            self.pvec.to_binding_rate(self.dataset.exp_type)
        )
        cost = self.scorer.calculate_sqrd_error(
            self.dataset.to_mm_num_subset(mm_num), simvalues,
            log=True, weigh_errors=True, weigh_multiplicity=False
        )

        # Do a correction for taking only one of the three subsets
        mult_weight = 1/3
        original_error_weights = self.dataset.weigh_errors(
            self.dataset.data, relative=True, normalize=False
        )
        correction = (original_error_weights[self.dataset.get_mm_nums() ==
                                             mm_num].mean() /
                      original_error_weights.sum())
        cost = mult_weight * correction * cost
        return simvalues, cost

    def plot_on_target_fit(self, color=None, axs=None):
        ylabel = ''
        if self.dataset.exp_type.name.upper() == 'NUCLEASEQ':
            ylabel = r'$k_{clv} ($s$^{-1})$'
        elif self.dataset.exp_type.name.upper() == 'CHAMP':
            ylabel = r'$K_A ($nM$^{-1})$'

        if axs is None:
            _, axs = plt.subplots(1, 1)

        simvalues, cost = self.compare_subset_to_data(0)

        color = 'tab:blue' if color is None else color
        axs.plot([1], self.dataset.to_mm_num_subset(0).data['value'],
                 '.', color=color, alpha=.7, markersize=10,
                 label='data')
        axs.plot([1], simvalues,
                 'x', color=color, alpha=.7, markersize=8, label='model')

        # window dressing
        axs.set_xticks([1])
        axs.set_xticklabels(['on-target'])
        axs.set_yscale('log')
        axs.set_ylabel(ylabel)
        axs.legend(loc='lower right')
        axs.set_title(f'on-target cost:\n'
                      f'{cost:.2e} ({100*cost/self.cost:.1f}%)',
                      pad=18)

        return axs

    def plot_single_mm_fit(self, color=None, axs=None):
        ylabel = ''
        if self.dataset.exp_type.name.upper() == 'NUCLEASEQ':
            ylabel = r'$k_{clv} ($s$^{-1})$'
        elif self.dataset.exp_type.name.upper() == 'CHAMP':
            ylabel = r'$K_A ($nM$^{-1})$'

        if axs is None:
            _, axs = plt.subplots(1, 1)

        simvalues, cost = self.compare_subset_to_data(1)

        color = 'tab:blue' if color is None else color

        axs.plot(np.arange(1, 21),
                 self.dataset.to_mm_num_subset(1).data['value'],
                 '.', color=color, alpha=.7, markersize=10,
                 label='data')
        axs.plot(np.arange(1, 21), simvalues,
                 '-', color=color, alpha=.7, linewidth=1.5, label='model')

        # window dressing
        axs.set_xticks(np.arange(1, 21))
        axs.set_xticks(np.arange(1, 21))
        axs.set_xticklabels(['1'] + 3 * [''] + ['5'] + 4 * [''] +
                            ['10'] + 4 * [''] + ['15'] + 4 * [''] + ['20'])
        axs.set_xlabel(r'mismatch position $b$')
        axs.set_yscale('log')
        axs.set_ylabel(ylabel)
        axs.legend(loc='lower right')
        axs.set_title(f'single mm cost:\n'
                      f'{cost:.2e} ({100*cost/self.cost:.1f}%)',
                      pad=18)

        return axs

    def plot_double_mm_fit(self, cmap=None, axs=None):
        ylabel = ''
        if self.dataset.exp_type.name.upper() == 'NUCLEASEQ':
            ylabel = r'$k_{clv} ($s$^{-1})$'
        elif self.dataset.exp_type.name.upper() == 'CHAMP':
            ylabel = r'$K_A ($nM$^{-1})$'

        if axs is None:
            _, axs = plt.subplots(1, 1)

        simvalues, cost = self.compare_subset_to_data(2)

        cmap = 'Blues' if cmap is None else cmap

        def mismatch_array_to_coordinates(mm_array):
            b1 = mm_array.index('1')
            b2 = b1 + 1 + mm_array[b1 + 1:].index('1')
            return b1, b2

        show_matrix = np.zeros(shape=(20, 20))
        i = 0
        for _, row in self.dataset.to_mm_num_subset(2).data.iterrows():
            x, y = mismatch_array_to_coordinates(row['mismatch_array'])
            show_matrix[y, x] = row['value']
            show_matrix[x, y] = simvalues[i]
            i += 1

        im = axs.imshow(show_matrix, cmap=cmap, norm=LogNorm(),
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
        _ = plt.colorbar(im, cax=cax)
        cax.set_title(ylabel)

        # window dressing
        axs.set_xlabel(r'mismatch 1 position $b_1$')
        axs.set_xticks(np.arange(0, 20))
        axs.set_xticklabels(['1'] + 3 * [''] + ['5'] + 4 * [''] +
                            ['10'] + 4 * [''] + ['15'] + 4 * [''] + ['20'])

        axs.set_ylabel(r'mismatch 2 position $b_2$')
        axs.set_yticks(np.arange(0, 20))
        axs.set_yticklabels(['1'] + 3 * [''] + ['5'] + 4 * [''] +
                            ['10'] + 4 * [''] + ['15'] + 4 * [''] + ['20'])

        axs.set_title(f'double mm cost:\n'
                      f'{cost:.2e} ({100*cost/self.cost:.1f}%)',
                      pad=18)

        return axs, cax

    def plot_fit_correlation(self, color=None, axs=None):
        ylabel = ''
        if self.dataset.exp_type.name.upper() == 'NUCLEASEQ':
            ylabel = r'$k_{clv} ($s$^{-1})$'
        elif self.dataset.exp_type.name.upper() == 'CHAMP':
            ylabel = r'$K_A ($nM$^{-1})$'

        if axs is None:
            _, axs = plt.subplots(1, 1)

        axs.scatter(
            self.dataset.to_mm_num_subset(0).data['value'],
            self.compare_subset_to_data(0)[0],
            marker='o', facecolor='none', edgecolor=color, alpha=0.7,
            s=24, label='on-target'
        )
        axs.scatter(
            self.dataset.to_mm_num_subset(1).data['value'],
            self.compare_subset_to_data(1)[0],
            marker='s', facecolor='none', edgecolor=color, alpha=0.7,
            s=24, label='off-target, 1 mm'
        )
        axs.scatter(
            self.dataset.to_mm_num_subset(2).data['value'],
            self.compare_subset_to_data(2)[0],
            marker='^', facecolor='none', edgecolor=color, alpha=0.7,
            s=24, label='off-target, 2 mm'
        )

        # window dressing
        axs.set_xscale('log')
        axs.set_xlabel('data - ' + ylabel)
        axs.set_yscale('log')
        axs.set_ylabel('model - ' + ylabel)

        # limits
        axs.set_aspect('equal', adjustable='box')
        minlim = min(axs.get_xlim()[0], axs.get_ylim()[0])
        extramaxlim = .11 if self.dataset.exp_type.value == 'CHAMP' else 0.
        maxlim = max(axs.get_xlim()[1], axs.get_ylim()[1], extramaxlim)
        axs.set_xlim(minlim, maxlim)
        axs.set_ylim(minlim, maxlim)
        axs.plot([minlim, maxlim], [minlim, maxlim], '--k',
                 linewidth=1, zorder=0)
        axs.minorticks_on()

        # ticks
        ticks = 10. ** np.arange(np.ceil(np.log10(minlim)),
                                 np.ceil(np.log10(maxlim)),
                                 1)
        axs.set_xticks(ticks)
        axs.set_yticks(ticks)

        axs.legend(loc='upper left', handlelength=1.)

        try:
            correlation, _ = pearsonr(
                np.log10(self.dataset.data['value']),
                np.log10(self.scorer.run_experiments(
                    self.dataset,
                    self.pvec.to_binding_rate(self.dataset.exp_type))
                )
            )
            axs.set_title('correlation: %.2f' % correlation, pad=18)
        except ValueError:
            axs.set_title('correlation unknown', pad=18)

        return axs

    def make_fit_dashboard(self, axs=None, color='tab:blue'):

        if axs is None:
            fig = plt.figure(
                figsize=(14, 4),
                constrained_layout=True,
            )
            grid = fig.add_gridspec(ncols=4, nrows=2,
                                    width_ratios=[.45, 1, 1.12, 1],
                                    height_ratios=([.1, 1]),
                                    hspace=.02, wspace=.08)

            title_axs = []
            axs = []
            title_axs += [fig.add_subplot(grid[0, :])]
            title_axs[-1].set_axis_off()
            title_axs[-1].text(.5, .5, self.dataset.exp_type.name.lower(),
                               horizontalalignment='center',
                               fontsize=14)
            axs += [
                fig.add_subplot(grid[1, 0]),
                fig.add_subplot(grid[1, 1]),
                fig.add_subplot(grid[1, 2]),
                fig.add_subplot(grid[1, 3])
            ]

        axs[0] = self.plot_on_target_fit(axs=axs[0], color=color)
        axs[1] = self.plot_single_mm_fit(axs=axs[1], color=color)
        axs[2] = self.plot_double_mm_fit(axs=axs[2],
                                         cmap=("Oranges"
                                               if color == "tab:orange"
                                               else None))
        axs[3] = self.plot_fit_correlation(axs=axs[3], color=color)

        # adjust on-target scale to 1 mm scale
        axs[0].set_ylim(axs[1].get_ylim())
