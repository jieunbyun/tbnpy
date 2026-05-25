tbnpy.variable
==============

Overview
--------

The :class:`~tbnpy.variable.Variable` class stores metadata about a Bayesian-network variable
(name, domain) and provides helper methods for working with **composite states** (sets of basic
states represented in a compact index form).

Quick start
-----------

.. code-block:: python

   from tbnpy.variable import Variable

   # Discrete variable: values define the ordered basic states
   v = Variable("damage_state", values=["none", "minor", "major"])

   # Example: the composite state {1, 2} (i.e., {"minor","major"})
   st = v.get_state({1, 2})
   S  = v.get_set(st)

Public API
----------

Variable
~~~~~~~~

.. py:class:: Variable(name: str, values=None)

   Manage information about a variable.

   Parameters
   ----------
   name
      Variable name used as an identifier (hashing, ordering, printing).
   values
      Domain specification.

      - **Discrete variable:** a ``list``/``numpy.ndarray`` of labels (states are indexed as
        ``0, 1, 2, ...`` in that order).
      - **Continuous variable:** a ``tuple`` ``(min, max)`` where bounds may be finite numbers
        or infinities (e.g. ``(-torch.inf, torch.inf)``).
      - **None:** allowed; indicates the domain is unspecified.

   Attributes
   ----------
   name : str
      Variable name.
   values : list | tuple | None
      Domain specification as described above.

   Notes
   -----
   **Ordering tip:** when applicable, prefer an ordering where *lower indices represent worse outcomes*,
   because some modules may assume this convention (e.g., ``["failure", "survival"]`` so that ``0 < 1``).


Properties
^^^^^^^^^^

.. py:attribute:: Variable.name
   :type: str

   Variable name.

.. py:attribute:: Variable.values
   :type: list | tuple | None

   Variable domain specification (discrete labels or continuous bounds).


Composite-state helpers
^^^^^^^^^^^^^^^^^^^^^^^

The following methods work with composite states. A **composite state** represents a *set* of basic
(discrete) states. Internally, the module uses a deterministic ordering over all non-empty subsets
(e.g., all 1-element subsets, then all 2-element subsets, ..., up to the full set).

.. py:method:: Variable.get_state(state_set: set) -> int

   Convert a set of basic-state indices to its composite-state index.

   Parameters
   ----------
   state_set
      A Python ``set`` of integer indices into ``values`` (e.g., ``{0}``, ``{1, 2}``).

   Returns
   -------
   int
      Composite-state index representing the same subset.

   Notes
   -----
   - ``state_set`` must be a non-empty set.
   - This method assumes a **discrete** variable (``values`` is list/ndarray).

   Examples
   --------
   .. code-block:: python

      v = Variable("x", ["low", "med", "high"])
      v.get_state({1, 2})  # -> composite index for {"med","high"}


.. py:method:: Variable.get_set(state: int) -> set

   Convert a composite-state index back to the corresponding set of basic-state indices.

   Parameters
   ----------
   state
      Composite-state index (non-negative integer).

   Returns
   -------
   set
      A set of basic-state indices (e.g., ``{0, 2}``).


.. py:method:: Variable.get_state_from_vector(vector) -> int

   Convert a binary indicator vector to a composite-state index.

   Parameters
   ----------
   vector
      A 1D list/array of length ``len(values)`` with entries in ``{0,1}``.
      A value of ``1`` indicates that the corresponding basic state is included in the subset.

   Returns
   -------
   int
      Composite-state index, or ``-1`` if the vector is all zeros (empty set).

Hidden helpers (intentionally omitted)
--------------------------------------

To keep this page focussed on the commonly used API, the following lower-level helper functions
are intentionally omitted from this page:

- ``get_Cbin_to_Cst``
- ``get_Cst_to_Cbin``
- ``build_state_to_binary_lookup``
- ``build_bitmask_to_state_lookup``

(They remain available in the module, but are treated as internal/advanced utilities.)

