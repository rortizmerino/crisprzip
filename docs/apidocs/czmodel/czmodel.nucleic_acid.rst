:py:mod:`czmodel.nucleic_acid`
==============================

.. py:module:: czmodel.nucleic_acid

.. autodoc2-docstring:: czmodel.nucleic_acid
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`MismatchPattern <czmodel.nucleic_acid.MismatchPattern>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.MismatchPattern
          :summary:
   * - :py:obj:`TargetDna <czmodel.nucleic_acid.TargetDna>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna
          :summary:
   * - :py:obj:`GuideTargetHybrid <czmodel.nucleic_acid.GuideTargetHybrid>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid
          :summary:
   * - :py:obj:`NearestNeighborModel <czmodel.nucleic_acid.NearestNeighborModel>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`format_point_mutations <czmodel.nucleic_acid.format_point_mutations>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.format_point_mutations
          :summary:
   * - :py:obj:`get_tempdir <czmodel.nucleic_acid.get_tempdir>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.get_tempdir
          :summary:
   * - :py:obj:`clear_cache <czmodel.nucleic_acid.clear_cache>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.clear_cache
          :summary:
   * - :py:obj:`make_hybr_energy_func <czmodel.nucleic_acid.make_hybr_energy_func>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.make_hybr_energy_func
          :summary:
   * - :py:obj:`get_hybridization_energy <czmodel.nucleic_acid.get_hybridization_energy>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.get_hybridization_energy
          :summary:
   * - :py:obj:`get_na_energies_cached <czmodel.nucleic_acid.get_na_energies_cached>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.get_na_energies_cached
          :summary:
   * - :py:obj:`find_average_mm_penalties <czmodel.nucleic_acid.find_average_mm_penalties>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.find_average_mm_penalties
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`tempdir <czmodel.nucleic_acid.tempdir>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.tempdir
          :summary:
   * - :py:obj:`memory <czmodel.nucleic_acid.memory>`
     - .. autodoc2-docstring:: czmodel.nucleic_acid.memory
          :summary:

API
~~~

.. py:function:: format_point_mutations(protospacer: str, target_sequence: str) -> typing.List[str]
   :canonical: czmodel.nucleic_acid.format_point_mutations

   .. autodoc2-docstring:: czmodel.nucleic_acid.format_point_mutations

.. py:class:: MismatchPattern(array: numpy.typing.ArrayLike)
   :canonical: czmodel.nucleic_acid.MismatchPattern

   .. autodoc2-docstring:: czmodel.nucleic_acid.MismatchPattern

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.nucleic_acid.MismatchPattern.__init__

   .. py:method:: __repr__()
      :canonical: czmodel.nucleic_acid.MismatchPattern.__repr__

   .. py:method:: __str__()
      :canonical: czmodel.nucleic_acid.MismatchPattern.__str__

   .. py:method:: from_string(mm_array_string)
      :canonical: czmodel.nucleic_acid.MismatchPattern.from_string
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.MismatchPattern.from_string

   .. py:method:: from_mm_pos(guide_length: int, mm_pos_list: list = None, zero_based_index=False)
      :canonical: czmodel.nucleic_acid.MismatchPattern.from_mm_pos
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.MismatchPattern.from_mm_pos

   .. py:method:: from_target_sequence(protospacer: str, target_sequence: str) -> czmodel.nucleic_acid.MismatchPattern
      :canonical: czmodel.nucleic_acid.MismatchPattern.from_target_sequence
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.MismatchPattern.from_target_sequence

   .. py:method:: make_random(guide_length: int, mm_num: int, rng: typing.Union[int, numpy.random.Generator] = None)
      :canonical: czmodel.nucleic_acid.MismatchPattern.make_random
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.MismatchPattern.make_random

   .. py:method:: get_mm_pos()
      :canonical: czmodel.nucleic_acid.MismatchPattern.get_mm_pos

      .. autodoc2-docstring:: czmodel.nucleic_acid.MismatchPattern.get_mm_pos

.. py:function:: get_tempdir()
   :canonical: czmodel.nucleic_acid.get_tempdir

   .. autodoc2-docstring:: czmodel.nucleic_acid.get_tempdir

.. py:function:: clear_cache()
   :canonical: czmodel.nucleic_acid.clear_cache

   .. autodoc2-docstring:: czmodel.nucleic_acid.clear_cache

.. py:data:: tempdir
   :canonical: czmodel.nucleic_acid.tempdir
   :value: 'get_tempdir(...)'

   .. autodoc2-docstring:: czmodel.nucleic_acid.tempdir

.. py:data:: memory
   :canonical: czmodel.nucleic_acid.memory
   :value: 'Memory(...)'

   .. autodoc2-docstring:: czmodel.nucleic_acid.memory

.. py:function:: make_hybr_energy_func(protospacer: str, weight: typing.Union[float, typing.Tuple[float, float]] = None)
   :canonical: czmodel.nucleic_acid.make_hybr_energy_func

   .. autodoc2-docstring:: czmodel.nucleic_acid.make_hybr_energy_func

.. py:function:: get_hybridization_energy(protospacer: str, offtarget_seq: str = None, mutations: str = '', weight: typing.Union[float, typing.Tuple[float, float]] = None) -> numpy.ndarray
   :canonical: czmodel.nucleic_acid.get_hybridization_energy

   .. autodoc2-docstring:: czmodel.nucleic_acid.get_hybridization_energy

.. py:function:: get_na_energies_cached(protospacer: str, offtarget_seq: str = None) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
   :canonical: czmodel.nucleic_acid.get_na_energies_cached

   .. autodoc2-docstring:: czmodel.nucleic_acid.get_na_energies_cached

.. py:function:: find_average_mm_penalties(protospacer: str, weight: typing.Union[float, typing.Tuple[float, float]] = None)
   :canonical: czmodel.nucleic_acid.find_average_mm_penalties

   .. autodoc2-docstring:: czmodel.nucleic_acid.find_average_mm_penalties

.. py:class:: TargetDna(target_sequence, upstream_nt: str = None, downstream_nt: str = None)
   :canonical: czmodel.nucleic_acid.TargetDna

   .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna.__init__

   .. py:attribute:: bp_map
      :canonical: czmodel.nucleic_acid.TargetDna.bp_map
      :value: None

      .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna.bp_map

   .. py:method:: __str__()
      :canonical: czmodel.nucleic_acid.TargetDna.__str__

      .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna.__str__

   .. py:method:: __reverse_transcript(sequence: str) -> str
      :canonical: czmodel.nucleic_acid.TargetDna.__reverse_transcript
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna.__reverse_transcript

   .. py:method:: from_cas9_target(full_target: str) -> czmodel.nucleic_acid.TargetDna
      :canonical: czmodel.nucleic_acid.TargetDna.from_cas9_target
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna.from_cas9_target

   .. py:method:: apply_point_mut(mutation: str)
      :canonical: czmodel.nucleic_acid.TargetDna.apply_point_mut

      .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna.apply_point_mut

   .. py:method:: make_random(length: int, seed=None) -> czmodel.nucleic_acid.TargetDna
      :canonical: czmodel.nucleic_acid.TargetDna.make_random
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.TargetDna.make_random

.. py:class:: GuideTargetHybrid(guide: str, target: czmodel.nucleic_acid.TargetDna, state: int = 0)
   :canonical: czmodel.nucleic_acid.GuideTargetHybrid

   .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid

   .. rubric:: Initialization

   .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.__init__

   .. py:attribute:: bp_map
      :canonical: czmodel.nucleic_acid.GuideTargetHybrid.bp_map
      :value: None

      .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.bp_map

   .. py:method:: __str__()
      :canonical: czmodel.nucleic_acid.GuideTargetHybrid.__str__

      .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.__str__

   .. py:method:: from_cas9_protospacer(protospacer: str, mismatches: str = '', state: int = 0) -> czmodel.nucleic_acid.GuideTargetHybrid
      :canonical: czmodel.nucleic_acid.GuideTargetHybrid.from_cas9_protospacer
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.from_cas9_protospacer

   .. py:method:: from_cas9_offtarget(offtarget_seq: str, protospacer: str, state: int = 0) -> czmodel.nucleic_acid.GuideTargetHybrid
      :canonical: czmodel.nucleic_acid.GuideTargetHybrid.from_cas9_offtarget
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.from_cas9_offtarget

   .. py:method:: apply_point_mut(mutation: str) -> czmodel.nucleic_acid.GuideTargetHybrid
      :canonical: czmodel.nucleic_acid.GuideTargetHybrid.apply_point_mut

      .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.apply_point_mut

   .. py:method:: set_rloop_state(rloop_state)
      :canonical: czmodel.nucleic_acid.GuideTargetHybrid.set_rloop_state

      .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.set_rloop_state

   .. py:method:: find_mismatches()
      :canonical: czmodel.nucleic_acid.GuideTargetHybrid.find_mismatches

      .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.find_mismatches

   .. py:method:: get_mismatch_pattern() -> czmodel.nucleic_acid.MismatchPattern
      :canonical: czmodel.nucleic_acid.GuideTargetHybrid.get_mismatch_pattern

      .. autodoc2-docstring:: czmodel.nucleic_acid.GuideTargetHybrid.get_mismatch_pattern

.. py:class:: NearestNeighborModel
   :canonical: czmodel.nucleic_acid.NearestNeighborModel

   .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel

   .. py:attribute:: dna_dna_params_file
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.dna_dna_params_file
      :value: 'nucleicacid_params/santaluciahicks2004.json'

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.dna_dna_params_file

   .. py:attribute:: rna_dna_params_file
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.rna_dna_params_file
      :value: 'nucleicacid_params/alkan2018.json'

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.rna_dna_params_file

   .. py:attribute:: dna_dna_params
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.dna_dna_params
      :type: dict
      :value: None

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.dna_dna_params

   .. py:attribute:: rna_dna_params
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.rna_dna_params
      :type: dict
      :value: None

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.rna_dna_params

   .. py:attribute:: energy_unit
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.energy_unit
      :value: 'kbt'

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.energy_unit

   .. py:method:: load_data(force=False)
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.load_data
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.load_data

   .. py:method:: set_energy_unit(unit: str)
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.set_energy_unit
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.set_energy_unit

   .. py:method:: convert_units(energy_value: typing.Union[float, numpy.ndarray])
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.convert_units
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.convert_units

   .. py:method:: get_hybridization_energy(hybrid: czmodel.nucleic_acid.GuideTargetHybrid, weight: typing.Union[float, typing.Tuple[float, float]] = None) -> numpy.ndarray
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.get_hybridization_energy
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.get_hybridization_energy

   .. py:method:: dna_opening_energy(hybrid: czmodel.nucleic_acid.GuideTargetHybrid) -> numpy.ndarray
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.dna_opening_energy
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.dna_opening_energy

   .. py:method:: rna_duplex_energy(hybrid: czmodel.nucleic_acid.GuideTargetHybrid) -> numpy.ndarray
      :canonical: czmodel.nucleic_acid.NearestNeighborModel.rna_duplex_energy
      :classmethod:

      .. autodoc2-docstring:: czmodel.nucleic_acid.NearestNeighborModel.rna_duplex_energy
