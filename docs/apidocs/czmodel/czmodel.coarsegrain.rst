:py:mod:`czmodel.coarsegrain`
=============================

.. py:module:: czmodel.coarsegrain

.. autodoc2-docstring:: czmodel.coarsegrain
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CoarseGrainedComplex <czmodel.coarsegrain.CoarseGrainedComplex>`
     - .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedComplex
          :summary:
   * - :py:obj:`CoarseGrainedSequenceComplex <czmodel.coarsegrain.CoarseGrainedSequenceComplex>`
     - .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedSequenceComplex
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`coarse_grain_landscape <czmodel.coarsegrain.coarse_grain_landscape>`
     - .. autodoc2-docstring:: czmodel.coarsegrain.coarse_grain_landscape
          :summary:

API
~~~

.. py:function:: coarse_grain_landscape(searcher_target_complex: typing.Union[czmodel.kinetics.SearcherTargetComplex, czmodel.kinetics.SearcherSequenceComplex], intermediate_range: typing.Tuple[int] = (7, 14))
   :canonical: czmodel.coarsegrain.coarse_grain_landscape

   .. autodoc2-docstring:: czmodel.coarsegrain.coarse_grain_landscape

.. py:class:: CoarseGrainedComplex(on_target_landscape: czmodel.nucleic_acid.np.ndarray, mismatch_penalties: czmodel.nucleic_acid.np.ndarray, internal_rates: dict, target_mismatches: czmodel.nucleic_acid.MismatchPattern, *args, **kwargs)
   :canonical: czmodel.coarsegrain.CoarseGrainedComplex

   Bases: :py:obj:`czmodel.kinetics.SearcherTargetComplex`

   .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedComplex

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedComplex.__init__

   .. py:method:: get_coarse_grained_rates(intermediate_range=(7, 14))
      :canonical: czmodel.coarsegrain.CoarseGrainedComplex.get_coarse_grained_rates

      .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedComplex.get_coarse_grained_rates

   .. py:method:: __calculate_effective_rate(forward_rate_array, backward_rate_array, start=0, stop=None)
      :canonical: czmodel.coarsegrain.CoarseGrainedComplex.__calculate_effective_rate
      :classmethod:

      .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedComplex.__calculate_effective_rate

   .. py:method:: __setup_partial_rate_matrix(forward_rate_array: numpy.ndarray, backward_rate_array: numpy.ndarray, start: int = 0, stop: int = None, final_state_absorbs=False) -> numpy.ndarray
      :canonical: czmodel.coarsegrain.CoarseGrainedComplex.__setup_partial_rate_matrix
      :staticmethod:

      .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedComplex.__setup_partial_rate_matrix

.. py:class:: CoarseGrainedSequenceComplex(*args, **kwargs)
   :canonical: czmodel.coarsegrain.CoarseGrainedSequenceComplex

   Bases: :py:obj:`czmodel.kinetics.SearcherSequenceComplex`, :py:obj:`czmodel.coarsegrain.CoarseGrainedComplex`

   .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedSequenceComplex

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.coarsegrain.CoarseGrainedSequenceComplex.__init__
