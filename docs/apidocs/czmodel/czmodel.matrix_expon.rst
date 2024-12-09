:py:mod:`czmodel.matrix_expon`
==============================

.. py:module:: czmodel.matrix_expon

.. autodoc2-docstring:: czmodel.matrix_expon
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`exponentiate_fast <czmodel.matrix_expon.exponentiate_fast>`
     - .. autodoc2-docstring:: czmodel.matrix_expon.exponentiate_fast
          :summary:
   * - :py:obj:`update_rate_matrix <czmodel.matrix_expon.update_rate_matrix>`
     - .. autodoc2-docstring:: czmodel.matrix_expon.update_rate_matrix
          :summary:
   * - :py:obj:`exponentiate_fast_var_onrate <czmodel.matrix_expon.exponentiate_fast_var_onrate>`
     - .. autodoc2-docstring:: czmodel.matrix_expon.exponentiate_fast_var_onrate
          :summary:
   * - :py:obj:`exponentiate_iterative <czmodel.matrix_expon.exponentiate_iterative>`
     - .. autodoc2-docstring:: czmodel.matrix_expon.exponentiate_iterative
          :summary:
   * - :py:obj:`exponentiate_iterative_var_onrate <czmodel.matrix_expon.exponentiate_iterative_var_onrate>`
     - .. autodoc2-docstring:: czmodel.matrix_expon.exponentiate_iterative_var_onrate
          :summary:

API
~~~

.. py:function:: exponentiate_fast(matrix: numpy.ndarray, time: numpy.ndarray)
   :canonical: czmodel.matrix_expon.exponentiate_fast

   .. autodoc2-docstring:: czmodel.matrix_expon.exponentiate_fast

.. py:function:: update_rate_matrix(ref_rate_matrix: numpy.ndarray, on_rate: float) -> numpy.ndarray
   :canonical: czmodel.matrix_expon.update_rate_matrix

   .. autodoc2-docstring:: czmodel.matrix_expon.update_rate_matrix

.. py:function:: exponentiate_fast_var_onrate(ref_matrix: numpy.ndarray, time: float, on_rates: numpy.ndarray)
   :canonical: czmodel.matrix_expon.exponentiate_fast_var_onrate

   .. autodoc2-docstring:: czmodel.matrix_expon.exponentiate_fast_var_onrate

.. py:function:: exponentiate_iterative(matrix: numpy.ndarray, time: numpy.ndarray)
   :canonical: czmodel.matrix_expon.exponentiate_iterative

   .. autodoc2-docstring:: czmodel.matrix_expon.exponentiate_iterative

.. py:function:: exponentiate_iterative_var_onrate(ref_matrix: numpy.ndarray, time: float, on_rates: numpy.ndarray)
   :canonical: czmodel.matrix_expon.exponentiate_iterative_var_onrate

   .. autodoc2-docstring:: czmodel.matrix_expon.exponentiate_iterative_var_onrate
