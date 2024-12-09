:py:mod:`czmodel.kinetics`
==========================

.. py:module:: czmodel.kinetics

.. autodoc2-docstring:: czmodel.kinetics
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Searcher <czmodel.kinetics.Searcher>`
     - .. autodoc2-docstring:: czmodel.kinetics.Searcher
          :summary:
   * - :py:obj:`BareSearcher <czmodel.kinetics.BareSearcher>`
     - .. autodoc2-docstring:: czmodel.kinetics.BareSearcher
          :summary:
   * - :py:obj:`GuidedSearcher <czmodel.kinetics.GuidedSearcher>`
     - .. autodoc2-docstring:: czmodel.kinetics.GuidedSearcher
          :summary:
   * - :py:obj:`SearcherTargetComplex <czmodel.kinetics.SearcherTargetComplex>`
     - .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex
          :summary:
   * - :py:obj:`SearcherSequenceComplex <czmodel.kinetics.SearcherSequenceComplex>`
     - .. autodoc2-docstring:: czmodel.kinetics.SearcherSequenceComplex
          :summary:

API
~~~

.. py:class:: Searcher(on_target_landscape: czmodel.nucleic_acid.ArrayLike, mismatch_penalties: czmodel.nucleic_acid.ArrayLike, internal_rates: dict, pam_detection=True, *args, **kwargs)
   :canonical: czmodel.kinetics.Searcher

   .. autodoc2-docstring:: czmodel.kinetics.Searcher

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.kinetics.Searcher.__init__

   .. py:method:: get_forward_rate_array(k_on, dead=False)
      :canonical: czmodel.kinetics.Searcher.get_forward_rate_array

      .. autodoc2-docstring:: czmodel.kinetics.Searcher.get_forward_rate_array

   .. py:method:: probe_target(target_mismatches: czmodel.nucleic_acid.MismatchPattern) -> czmodel.kinetics.SearcherTargetComplex
      :canonical: czmodel.kinetics.Searcher.probe_target

      .. autodoc2-docstring:: czmodel.kinetics.Searcher.probe_target

   .. py:method:: calculate_solution_energy(k_on)
      :canonical: czmodel.kinetics.Searcher.calculate_solution_energy

      .. autodoc2-docstring:: czmodel.kinetics.Searcher.calculate_solution_energy

.. py:class:: BareSearcher(on_target_landscape: czmodel.nucleic_acid.ArrayLike, mismatch_penalties: czmodel.nucleic_acid.ArrayLike, internal_rates: dict, pam_detection=True, *args, **kwargs)
   :canonical: czmodel.kinetics.BareSearcher

   Bases: :py:obj:`czmodel.kinetics.Searcher`

   .. autodoc2-docstring:: czmodel.kinetics.BareSearcher

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.kinetics.BareSearcher.__init__

   .. py:method:: from_searcher(searcher, protospacer: str, weight: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.Tuple[float, float]] = None, *args, **kwargs)
      :canonical: czmodel.kinetics.BareSearcher.from_searcher
      :classmethod:

      .. autodoc2-docstring:: czmodel.kinetics.BareSearcher.from_searcher

   .. py:method:: to_searcher(protospacer: str, weight: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.Tuple[float, float]] = None) -> czmodel.kinetics.Searcher
      :canonical: czmodel.kinetics.BareSearcher.to_searcher

      .. autodoc2-docstring:: czmodel.kinetics.BareSearcher.to_searcher

   .. py:method:: bind_guide_rna(protospacer: str, weight: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.Tuple[float, float]] = None) -> czmodel.kinetics.GuidedSearcher
      :canonical: czmodel.kinetics.BareSearcher.bind_guide_rna

      .. autodoc2-docstring:: czmodel.kinetics.BareSearcher.bind_guide_rna

   .. py:method:: probe_target(target_mismatches: czmodel.nucleic_acid.MismatchPattern) -> czmodel.kinetics.SearcherTargetComplex
      :canonical: czmodel.kinetics.BareSearcher.probe_target

      .. autodoc2-docstring:: czmodel.kinetics.BareSearcher.probe_target

   .. py:method:: probe_sequence(protospacer: str, target_seq: str, weight: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.Tuple[float, float]] = None) -> czmodel.kinetics.SearcherSequenceComplex
      :canonical: czmodel.kinetics.BareSearcher.probe_sequence

      .. autodoc2-docstring:: czmodel.kinetics.BareSearcher.probe_sequence

.. py:class:: GuidedSearcher(on_target_landscape: czmodel.nucleic_acid.np.ndarray, mismatch_penalties: czmodel.nucleic_acid.np.ndarray, internal_rates: dict, protospacer: str, weight: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.Tuple[float, float]] = None, *args, **kwargs)
   :canonical: czmodel.kinetics.GuidedSearcher

   Bases: :py:obj:`czmodel.kinetics.BareSearcher`

   .. autodoc2-docstring:: czmodel.kinetics.GuidedSearcher

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.kinetics.GuidedSearcher.__init__

   .. py:method:: to_searcher(*args, **kwargs) -> czmodel.kinetics.Searcher
      :canonical: czmodel.kinetics.GuidedSearcher.to_searcher

   .. py:method:: to_bare_searcher() -> czmodel.kinetics.BareSearcher
      :canonical: czmodel.kinetics.GuidedSearcher.to_bare_searcher

      .. autodoc2-docstring:: czmodel.kinetics.GuidedSearcher.to_bare_searcher

   .. py:method:: set_on_target(protospacer: str) -> None
      :canonical: czmodel.kinetics.GuidedSearcher.set_on_target

      .. autodoc2-docstring:: czmodel.kinetics.GuidedSearcher.set_on_target

   .. py:method:: probe_sequence(target_seq: str, weight: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.Tuple[float, float]] = None, *args, **kwargs) -> czmodel.kinetics.SearcherSequenceComplex
      :canonical: czmodel.kinetics.GuidedSearcher.probe_sequence

      .. autodoc2-docstring:: czmodel.kinetics.GuidedSearcher.probe_sequence

.. py:class:: SearcherTargetComplex(on_target_landscape: czmodel.nucleic_acid.np.ndarray, mismatch_penalties: czmodel.nucleic_acid.np.ndarray, internal_rates: dict, target_mismatches: czmodel.nucleic_acid.MismatchPattern, *args, **kwargs)
   :canonical: czmodel.kinetics.SearcherTargetComplex

   Bases: :py:obj:`czmodel.kinetics.Searcher`

   .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex.__init__

   .. py:method:: _get_off_target_landscape()
      :canonical: czmodel.kinetics.SearcherTargetComplex._get_off_target_landscape

      .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex._get_off_target_landscape

   .. py:method:: _get_landscape_diff()
      :canonical: czmodel.kinetics.SearcherTargetComplex._get_landscape_diff

      .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex._get_landscape_diff

   .. py:method:: _get_backward_rate_array()
      :canonical: czmodel.kinetics.SearcherTargetComplex._get_backward_rate_array

      .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex._get_backward_rate_array

   .. py:method:: get_rate_matrix(on_rate: float, dead=False) -> czmodel.nucleic_acid.np.ndarray
      :canonical: czmodel.kinetics.SearcherTargetComplex.get_rate_matrix

      .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex.get_rate_matrix

   .. py:method:: solve_master_equation(initial_condition: czmodel.nucleic_acid.np.ndarray, time: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.np.ndarray], on_rate: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.np.ndarray], dead=False, rebinding=True, mode='fast') -> czmodel.nucleic_acid.np.ndarray
      :canonical: czmodel.kinetics.SearcherTargetComplex.solve_master_equation

      .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex.solve_master_equation

   .. py:method:: get_cleaved_fraction(time: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.np.ndarray], on_rate: float) -> czmodel.nucleic_acid.np.ndarray
      :canonical: czmodel.kinetics.SearcherTargetComplex.get_cleaved_fraction

      .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex.get_cleaved_fraction

   .. py:method:: get_bound_fraction(time: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.np.ndarray], on_rate: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.np.ndarray], pam_inclusion: float = 1.0) -> czmodel.nucleic_acid.np.ndarray
      :canonical: czmodel.kinetics.SearcherTargetComplex.get_bound_fraction

      .. autodoc2-docstring:: czmodel.kinetics.SearcherTargetComplex.get_bound_fraction

.. py:class:: SearcherSequenceComplex(on_target_landscape: czmodel.nucleic_acid.np.ndarray, mismatch_penalties: czmodel.nucleic_acid.np.ndarray, internal_rates: dict, protospacer: str, target_seq: str, weight: czmodel.nucleic_acid.Union[float, czmodel.nucleic_acid.Tuple[float, float]] = None)
   :canonical: czmodel.kinetics.SearcherSequenceComplex

   Bases: :py:obj:`czmodel.kinetics.GuidedSearcher`, :py:obj:`czmodel.kinetics.SearcherTargetComplex`

   .. autodoc2-docstring:: czmodel.kinetics.SearcherSequenceComplex

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.kinetics.SearcherSequenceComplex.__init__

   .. py:method:: _get_off_target_landscape(weight=None)
      :canonical: czmodel.kinetics.SearcherSequenceComplex._get_off_target_landscape
