:py:mod:`czmodel.plotting`
==========================

.. py:module:: czmodel.plotting

.. autodoc2-docstring:: czmodel.plotting
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SearcherPlotter <czmodel.plotting.SearcherPlotter>`
     - .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter
          :summary:

API
~~~

.. py:class:: SearcherPlotter(searcher: czmodel.kinetics.Searcher)
   :canonical: czmodel.plotting.SearcherPlotter

   .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.__init__

   .. py:attribute:: title_style
      :canonical: czmodel.plotting.SearcherPlotter.title_style
      :value: None

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.title_style

   .. py:attribute:: label_style
      :canonical: czmodel.plotting.SearcherPlotter.label_style
      :value: None

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.label_style

   .. py:attribute:: tick_style
      :canonical: czmodel.plotting.SearcherPlotter.tick_style
      :value: None

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.tick_style

   .. py:attribute:: line_style
      :canonical: czmodel.plotting.SearcherPlotter.line_style
      :value: None

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.line_style

   .. py:method:: prepare_landscape_canvas(y_lims: tuple = None, title: str = '', axs: matplotlib.axes.Axes = None) -> matplotlib.axes.Axes
      :canonical: czmodel.plotting.SearcherPlotter.prepare_landscape_canvas

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.prepare_landscape_canvas

   .. py:method:: prepare_rates_canvas(y_lims: tuple = None, title: str = 'Transition rates', axs: matplotlib.axes.Axes = None, extra_rates: dict = None) -> matplotlib.axes.Axes
      :canonical: czmodel.plotting.SearcherPlotter.prepare_rates_canvas

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.prepare_rates_canvas

   .. py:method:: prepare_landscape_line(axs: matplotlib.axes.Axes, color='cornflowerblue', **plot_kwargs) -> matplotlib.lines.Line2D
      :canonical: czmodel.plotting.SearcherPlotter.prepare_landscape_line

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.prepare_landscape_line

   .. py:method:: prepare_rates_line(axs: matplotlib.axes.Axes, color='cornflowerblue', **plot_kwargs) -> matplotlib.lines.Line2D
      :canonical: czmodel.plotting.SearcherPlotter.prepare_rates_line

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.prepare_rates_line

   .. py:method:: update_on_target_landscape(line: matplotlib.lines.Line2D) -> None
      :canonical: czmodel.plotting.SearcherPlotter.update_on_target_landscape

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.update_on_target_landscape

   .. py:method:: update_solution_energies(lines: list, on_rates: list) -> None
      :canonical: czmodel.plotting.SearcherPlotter.update_solution_energies

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.update_solution_energies

   .. py:method:: update_mismatches(line: matplotlib.lines.Line2D) -> None
      :canonical: czmodel.plotting.SearcherPlotter.update_mismatches

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.update_mismatches

   .. py:method:: update_rates(line: matplotlib.lines.Line2D, extra_rates: dict = None) -> None
      :canonical: czmodel.plotting.SearcherPlotter.update_rates

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.update_rates

   .. py:method:: plot_on_target_landscape(y_lims: tuple = None, color='cornflowerblue', axs: matplotlib.axes.Axes = None, on_rates: list = None, **plot_kwargs) -> matplotlib.axes.Axes
      :canonical: czmodel.plotting.SearcherPlotter.plot_on_target_landscape

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.plot_on_target_landscape

   .. py:method:: plot_off_target_landscape(mismatch_positions: czmodel.kinetics.MismatchPattern, y_lims: tuple = None, color='firebrick', axs: matplotlib.axes.Axes = None, on_rates: list = None, **plot_kwargs) -> matplotlib.axes.Axes
      :canonical: czmodel.plotting.SearcherPlotter.plot_off_target_landscape

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.plot_off_target_landscape

   .. py:method:: plot_mismatch_penalties(y_lims: tuple = None, color='firebrick', axs: matplotlib.axes.Axes = None, **plot_kwargs) -> matplotlib.axes.Axes
      :canonical: czmodel.plotting.SearcherPlotter.plot_mismatch_penalties

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.plot_mismatch_penalties

   .. py:method:: plot_internal_rates(y_lims: tuple = None, color='cornflowerblue', axs: matplotlib.axes.Axes = None, extra_rates: dict = None, **plot_kwargs) -> matplotlib.axes.Axes
      :canonical: czmodel.plotting.SearcherPlotter.plot_internal_rates

      .. autodoc2-docstring:: czmodel.plotting.SearcherPlotter.plot_internal_rates
