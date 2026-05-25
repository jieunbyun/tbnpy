tbnpy.cpt
========

Overview
--------

The :class:`~tbnpy.cpt.Cpt` class stores a **conditional probability tensor** (CPT) in an
*event-table* form:

- ``C``: an event matrix whose rows enumerate compatible assignments of
  ``childs + parents`` (stored as **composite-state indices** for each variable).
- ``p``: probabilities for each row/event in ``C``.

It also supports sampling and log-probability evaluation, including cases where parent
assignments are provided as samples (optionally aligned with multiple evidence rows).

Quick start
-----------

.. code-block:: python

   import torch
   from tbnpy.variable import Variable
   from tbnpy.cpt import Cpt

   # Define variables (discrete example)
   X = Variable("X", ["0", "1"])
   Y = Variable("Y", ["0", "1"])

   # Event table for P(X | Y) with two events per value of Y (toy example)
   # Columns are [childs | parents] = [X | Y]
   C = [
       [0, 0],  # X=0, Y=0
       [1, 0],  # X=1, Y=0
       [0, 1],  # X=0, Y=1
       [1, 1],  # X=1, Y=1
   ]
   p = [0.8, 0.2, 0.1, 0.9]  # probabilities for each row in C

   cpt = Cpt(childs=[X], parents=[Y], C=C, p=p)

   # Sample X given parent samples Y
   Cs_par = torch.tensor([[0], [1], [1]])  # three parent samples: Y=0,1,1
   Cs_child, logp = cpt.sample(Cs_pars=Cs_par)

   # Evaluate log probability of full samples [X,Y]
   Cs_full = torch.tensor([[0, 0],
                           [1, 0],
                           [1, 1]])
   logp_full = cpt.log_prob(Cs_full)


Public API
----------

Cpt
~~~

.. py:class:: Cpt(childs: list, parents: list = [], C = [], p = [], Cs = [], ps = [], evidence = [], device: str = "cpu")

   Defines a conditional probability tensor (CPT) using tensor operations (PyTorch).

   Parameters
   ----------
   childs
      List of :class:`~tbnpy.variable.Variable` instances treated as the **child** variables.
   parents
      List of :class:`~tbnpy.variable.Variable` instances treated as the **parent** variables.
   C
      Event matrix (rows = events; columns = ``childs + parents``) storing **composite-state indices**.

      Accepted input types:

      - ``list`` (nested)
      - ``numpy.ndarray``
      - ``torch.Tensor``

      Stored as ``torch.int64``.
   p
      Probability vector aligned with rows of ``C`` (same number of rows).

      Accepted input types:

      - ``list``
      - ``numpy.ndarray``
      - ``torch.Tensor``

      Stored as ``torch.float32``. If provided as 1D, it is reshaped to a column ``(n_events, 1)``.
   Cs
      Stored samples of assignments (optional). See :meth:`~tbnpy.cpt.Cpt.sample` and
      :meth:`~tbnpy.cpt.Cpt.sample_evidence`.
   ps
      Stored sampling probabilities/log-probabilities (optional).
   evidence
      Observations of the child variables (optional). See :attr:`~tbnpy.cpt.Cpt.evidence`.
   device
      Torch device specifier (e.g. ``"cpu"``, ``"cuda"``, or ``torch.device(...)``).

   Attributes
   ----------
   childs : list[Variable]
      Child variables.
   parents : list[Variable]
      Parent variables.
   C : torch.Tensor
      Event matrix of shape ``(n_events, n_childs + n_parents)``.
   p : torch.Tensor
      Probability vector of shape ``(n_events, 1)`` (or sometimes treated as ``(n_events,)`` internally).
   Cs : torch.Tensor
      Sample matrix (shape depends on the sampling mode).
   ps : torch.Tensor
      Sampling probabilities (stored as log probabilities in some methods).
   evidence : torch.Tensor
      Evidence matrix for child variables, shape ``(n_evidence, n_childs)``.

   Notes
   -----
   - For consistency, ``C`` and ``Cs`` store **composite-state indices** (not raw basic-state indices).
   - Some internal methods convert composite-state indices to **binary** representations for fast
     compatibility checks. Those helpers are documented at a high level below.


Core data structures
^^^^^^^^^^^^^^^^^^^^

.. py:attribute:: Cpt.C
   :type: torch.Tensor

   Event matrix ``(n_events, n_childs + n_parents)`` of composite-state indices.

.. py:attribute:: Cpt.p
   :type: torch.Tensor

   Event probabilities aligned with ``C``. Shape is typically ``(n_events, 1)``.


Evidence
^^^^^^^^

.. py:attribute:: Cpt.evidence
   :type: torch.Tensor

   Evidence for the child variables.

   Input / storage rules
   ---------------------
   - If ``None`` or ``[]``: stored as an empty tensor of shape ``(0, n_childs)``.
   - If there is **one** child variable: accepts evidence of shape ``(N,)`` or ``(N, 1)``.
   - If there are **multiple** child variables: evidence must be shape ``(N, n_childs)``.


Sampling
^^^^^^^^

.. py:method:: Cpt.sample(n_sample: int | None = None, Cs_pars: torch.Tensor | None = None, batch_size: int = 100_000)

   Sample child assignments from this CPT.

   Parameters
   ----------
   n_sample
      Number of samples to generate **when there are no parents**.
   Cs_pars
      Parent samples (composite states) of shape ``(n_samples, n_parents)``
      **when parents exist**.
   batch_size
      Batch size used to avoid materialising very large intermediate tensors.

   Returns
   -------
   (Cs, ps)
      ``Cs`` is a tensor of sampled child composite states:

      - No parents: ``Cs`` has shape ``(n_sample, n_childs)``
      - With parents: ``Cs`` has shape ``(n_samples, n_childs)`` and aligns with rows of ``Cs_pars``

      ``ps`` stores the **log** probability of the selected event for each sample.

   Notes
   -----
   When parents exist, sampling is performed by:

   1. Converting event definitions and parent samples to a binary form;
   2. Zeroing out events incompatible with each parent sample; and
   3. Sampling an event index from the resulting conditional distribution.


.. py:method:: Cpt.sample_evidence(Cs_pars: torch.Tensor, batch_size: int = 100_000)

   Sample child assignments when parent samples are **aligned with multiple evidence rows**.

   Parameters
   ----------
   Cs_pars
      Parent samples arranged per evidence row, shape ``(n_evi, n_samples, n_parents)``.
   batch_size
      Batch size for internal computations.

   Returns
   -------
   (Cs_out, ps_out)
      ``Cs_out`` has shape ``(n_evi, n_samples, n_childs + n_parents)``,
      storing ``[childs | parents]`` for each evidence row and parent sample.

      ``ps_out`` has shape ``(n_evi, n_samples)`` and contains log probabilities.

   Notes
   -----
   The current module defines ``sample_evidence`` twice; the *second* definition (the vectorised version)
   is the one that is effective at runtime.


Log-probability evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:method:: Cpt.log_prob(Cs: torch.Tensor, batch_size: int = 100_000) -> torch.Tensor

   Compute ``log P(Cs)`` for each row of ``Cs`` where ``Cs`` stores assignments in the order
   ``[childs | parents]``.

   Parameters
   ----------
   Cs
      Composite-state assignments of shape ``(n_samples, n_childs + n_parents)``.
   batch_size
      Batch size for internal compatibility checks.

   Returns
   -------
   torch.Tensor
      Log probabilities of shape ``(n_samples,)``.

   Notes
   -----
   This method treats **all variables uniformly** when checking compatibility against the event table.


.. py:method:: Cpt.log_prob_evidence(Cs_par: torch.Tensor, batch_size: int = 100_000) -> torch.Tensor

   Compute ``log P(evidence | parents(samples))`` for each parent sample.

   Parameters
   ----------
   Cs_par
      Parent samples (composite states). Accepted shapes:

      - ``(N_samples, n_parents)``
      - ``(n_evi, N_samples, n_parents)`` (if evidence exists for parents too)

   batch_size
      Batch size for internal computations.

   Returns
   -------
   torch.Tensor
      Log probabilities of shape ``(N_samples,)``.


Advanced / internal helpers
---------------------------

The following methods are primarily implementation details used to speed up sampling/inference.
You may document them later if you decide they are part of the public API.

.. py:method:: Cpt._get_C_binary()

   Convert the event matrix ``C`` into a padded binary tensor of shape
   ``(n_events, n_vars, max_basic)`` for vectorised compatibility checks.

.. py:method:: Cpt.expand_and_check_compatibility(C_binary, samples)

   Parent-aware compatibility filtering used by :meth:`~tbnpy.cpt.Cpt.sample`.

.. py:method:: Cpt.expand_and_check_compatibility_all(C_binary, samples_binary)

   Compatibility filtering that treats all variables uniformly, used by :meth:`~tbnpy.cpt.Cpt.log_prob`.


Utility functions
-----------------

.. py:function:: get_names(var_list)

   Return the list of ``.name`` for variables in ``var_list``.

   Parameters
   ----------
   var_list
      List of :class:`~tbnpy.variable.Variable`.

   Returns
   -------
   list[str]
      Names of the variables.
