tbnpy.inference
==============

Overview
--------

This module provides lightweight **forward-sampling utilities** for a Bayesian network defined by
a dictionary of *probability objects* (typically :class:`~tbnpy.cpt.Cpt` instances, but any object
with a compatible interface works).

A *probability object* ``P`` is assumed to expose:

- ``P.childs`` : list of child variables (each variable has ``.name``)
- ``P.parents`` : list of parent variables (each variable has ``.name``)
- ``P.sample(...)`` : sampling method (signature depends on whether parents exist)
- ``P.log_prob(...)`` : log-probability evaluation for rows of ``[childs | parents]``
- (optionally) ``P.sample_evidence(...)`` : evidence-aligned sampling

The key idea is to:

1. collect all ancestors of query nodes,
2. order them topologically (parents before children),
3. forward-sample along that order, storing samples back into each probability object.

Glossary
--------

- **probs**: ``dict[str, ProbObject]`` mapping node name → probability object.
- **node name**: a string key in ``probs``. In this module, node names are treated as *variable names*.
- **Cs**: sampled assignments stored as **composite-state indices**.
- **ps**: stored per-sample probability values. In most usages here, ``ps`` is **log probability**.

Quick start
-----------

.. code-block:: python

   # probs: {"X": P(X), "Y": P(Y|X), ...}
   ordered = get_ancestor_order(probs, query_nodes={"Y"})
   probs_s = sample(probs, query_nodes={"Y"}, n_sample=10_000)

   # probs_s["Y"].Cs contains samples for Y and its parents (if any)
   # probs_s["Y"].ps contains per-sample (log) probabilities

Public API
----------

Topological utilities
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: get_ancestor_order(probs: dict, query_nodes: list[str] | set[str]) -> list[str]

   Compute the set of all ancestors of the query nodes and return them in a valid
   topological order (parents appear before children).

   Parameters
   ----------
   probs
      Mapping from node name → probability object. Each probability object must provide:

      - ``childs``: list of child variables
      - ``parents``: list of parent variables, each having ``.name``
   query_nodes
      Iterable of node names whose marginals (or descendant computations) are of interest.

   Returns
   -------
   list[str]
      Topologically sorted list of all ancestors of ``query_nodes``, including the query nodes.

   Notes
   -----
   - The function performs validation and will raise ``AssertionError`` if inputs are inconsistent
     (e.g., missing nodes, missing attributes).
   - Cycles are detected indirectly via topological sorting consistency checks.


Forward sampling without evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: sample(probs: dict, query_nodes: list[str] | set[str], n_sample: int) -> dict

   Forward-sample all ancestors of ``query_nodes`` and return a deep-copied probability structure
   with stored samples and per-sample probabilities.

   Parameters
   ----------
   probs
      Mapping from node name → probability object.
   query_nodes
      Node names to condition the ancestral subgraph selection.
   n_sample
      Number of samples to generate.

   Returns
   -------
   dict
      A dictionary ``{node_name: prob_object}`` restricted to the ancestral subgraph, in ancestor order.
      For each returned probability object ``P``:

      - ``P.Cs`` is a tensor with shape ``(n_sample, n_childs)`` or ``(n_sample, n_childs + n_parents)``
        depending on the implementation of ``P.sample``.
      - ``P.ps`` is a tensor with shape ``(n_sample,)`` (often log-probabilities).

   How sampling is performed
   -------------------------
   1. Compute ancestor order using :func:`get_ancestor_order`.
   2. Deep-copy the needed probability objects.
   3. Build a lookup ``var_to_source`` to find where each variable’s samples are stored.
   4. For each node in topological order:

      - if the node has no parents: call ``P.sample(n_sample)``.
      - if parents exist: assemble parent sample matrix
        ``Cs_par`` of shape ``(n_sample, n_parents)`` and call ``P.sample(Cs_par)``.

   Important
   ---------
   The module assumes each variable appears as a **child** of exactly one probability object.
   If a variable is a child in multiple objects, an ``AssertionError`` is raised.


Forward sampling with evidence
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evidence is provided as a table (typically a pandas DataFrame) whose columns are variable names
and whose rows are evidence scenarios.

Two implementations are included:

- :func:`sample_evidence_v0`: uses ``prob.sample_evidence`` when parents exist (vectorised),
  and uses ``prob.log_prob`` for observed children.
- :func:`sample_evidence`: uses only ``prob.sample`` (no ``prob.sample_evidence``), which is
  sometimes easier to maintain/debug.

.. py:function:: sample_evidence_v0(probs: dict, query_nodes: list[str] | set[str], n_sample: int, evidence_df) -> dict

   Forward-sample all ancestors of ``query_nodes`` under multiple evidence rows.

   Parameters
   ----------
   probs
      Mapping from node name → probability object.
   query_nodes
      Node names of interest.
   n_sample
      Number of samples per evidence row.
   evidence_df
      A pandas-like DataFrame. Each column name must match a variable name.
      Shape ``(n_evi, n_evidence_vars)``.

   Returns
   -------
   dict
      ``{node_name: prob_object}`` for the ancestral subgraph. Each returned object contains:

      - ``prob_object.Cs`` of shape ``(n_evi, n_sample, n_childs + n_parents)``
        (or ``(n_evi, n_sample, n_childs)`` for root nodes / special cases)
      - ``prob_object.ps`` of shape ``(n_evi, n_sample)`` containing log-probabilities

   Observed child handling
   -----------------------
   If a node is observed (its name is a column in ``evidence_df``), this function:

   - sets child samples to the observed value repeated over samples, and
   - computes ``ps`` by evaluating ``prob.log_prob`` on the assembled ``[childs | parents]`` rows.

   Parent handling
   ---------------
   For each parent variable:

   - if the parent is observed in ``evidence_df``, the observed values are used,
   - otherwise, sampled values from earlier nodes are used.

   Notes
   -----
   - Evidence values are converted to torch tensors.
   - This implementation expects ``prob.sample_evidence(Cs_pars)`` to accept parent samples of
     shape ``(n_evi, n_sample, n_parents)``.


.. py:function:: sample_evidence(probs: dict, query_nodes: list[str] | set[str], n_sample: int, evidence_df) -> dict

   Forward-sample all ancestors of ``query_nodes`` under multiple evidence rows using **only**
   ``prob.sample`` (no ``prob.sample_evidence``).

   Parameters
   ----------
   probs
      Mapping from node name → probability object.
   query_nodes
      Node names of interest.
   n_sample
      Number of samples per evidence row.
   evidence_df
      A pandas-like DataFrame with evidence columns.

   Returns
   -------
   dict
      ``{node_name: prob_object}`` for the ancestral subgraph. Each returned object contains:

      - ``prob_object.Cs`` of shape ``(n_evi, n_sample, n_childs + n_parents)`` (non-root nodes)
        or ``(n_evi, n_sample, n_childs)`` (root / observed child cases)
      - ``prob_object.ps`` of shape ``(n_evi, n_sample)`` containing log-probabilities

   Implementation sketch
   ---------------------
   - Root nodes are sampled by generating ``n_evi * n_sample`` samples and reshaping to ``(n_evi, n_sample, ...)``.
   - Non-root nodes build a parent matrix ``Cs_par_flat`` of shape ``(n_evi*n_sample, n_parents)``,
     then call ``prob.sample(Cs_pars=Cs_par_flat)``, and reshape outputs back to evidence form.
   - Observed nodes compute log-probability via ``prob.log_prob`` without sampling.

