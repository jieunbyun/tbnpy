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
3. forward-sample along that order in batches, retaining samples only for the **query nodes**
   (intermediate ancestor samples are discarded after each batch to keep memory bounded).

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

.. py:function:: sample(probs: dict, query_nodes: list[str] | set[str], n_sample: int, batch_size: int = 50_000) -> dict

   Forward-sample all ancestors of ``query_nodes`` and return samples for the **query nodes only**.
   Intermediate ancestor samples are held only for the duration of the current batch and discarded
   afterwards, keeping peak memory bounded by ``batch_size`` rather than ``n_sample``.

   Parameters
   ----------
   probs
      Mapping from node name → probability object.
   query_nodes
      Node names whose samples will be returned. All ancestors are still sampled internally
      (they are required to generate query-node samples), but are not retained in the output.
   n_sample
      Total number of samples to generate.
   batch_size
      Maximum number of samples processed per batch. Controls peak memory: at any time, only
      ``min(batch_size, n_sample)`` samples per ancestor variable are held in memory.
      Defaults to ``50_000``.

   Returns
   -------
   dict
      A dictionary ``{node_name: prob_object}`` containing **only the query nodes** (not their
      ancestors). For each returned probability object ``P``:

      - ``P.Cs`` is a tensor with shape ``(n_sample, n_childs)`` or ``(n_sample, n_childs + n_parents)``
        depending on the implementation of ``P.sample``.
      - ``P.ps`` is a tensor with shape ``(n_sample,)`` (often log-probabilities).

   How sampling is performed
   -------------------------
   1. Compute ancestor order using :func:`get_ancestor_order`.
   2. Deep-copy the needed probability objects (used to call ``P.sample`` only; ``.Cs`` / ``.ps``
      are populated on query nodes at the end).
   3. Iterate over batches of size ``batch_size`` (outer loop). Within each batch, walk the nodes
      in topological order (inner loop), keeping a temporary ``batch_samples`` dictionary
      ``{var_name: tensor(n_batch,)}`` that holds only the current batch's ancestor values.
   4. For each node in the inner loop:

      - if the node has no parents: call ``P.sample(n_sample=n_batch)``.
      - if parents exist: assemble parent sample matrix ``Cs_par`` of shape
        ``(n_batch, n_parents)`` from ``batch_samples`` and call ``P.sample(Cs_pars=Cs_par)``.
      - if the node is a query node, append ``Cs_batch`` and ``ps_batch`` to the per-query
        accumulators.

   5. After all batches complete, concatenate the per-batch tensors for each query node along
      the sample dimension and attach them as ``P.Cs`` / ``P.ps``.

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
  and uses ``prob.log_prob`` for observed children. Returns the **full ancestral subgraph**
  (samples for every ancestor are retained).
- :func:`sample_evidence`: uses only ``prob.sample`` (no ``prob.sample_evidence``), processes
  samples in batches, and returns **only the query nodes** — intermediate ancestor samples
  are discarded after each batch to keep peak memory bounded.

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


.. py:function:: sample_evidence(probs: dict, query_nodes: list[str] | set[str], n_sample: int, evidence_df, batch_size: int = 50_000) -> dict

   Forward-sample all ancestors of ``query_nodes`` under multiple evidence rows using **only**
   ``prob.sample`` (no ``prob.sample_evidence``), and return samples for the **query nodes only**.
   Intermediate ancestor samples are held only for the duration of the current batch and discarded
   afterwards, keeping peak memory bounded by ``batch_size`` rather than ``n_sample``.

   Parameters
   ----------
   probs
      Mapping from node name → probability object.
   query_nodes
      Node names whose samples will be returned. All ancestors are still sampled internally
      (they are required to generate query-node samples), but are not retained in the output.
   n_sample
      Number of samples per evidence row.
   evidence_df
      A pandas-like DataFrame with evidence columns. Each column name must match a variable name.
      Shape ``(n_evi, n_evidence_vars)``.
   batch_size
      Maximum number of samples processed per batch (per evidence row). Controls peak memory:
      at any time, only ``(n_evi, min(batch_size, n_sample))`` samples per ancestor variable are
      held in memory. Defaults to ``50_000``.

   Returns
   -------
   dict
      ``{node_name: prob_object}`` containing **only the query nodes** (not their ancestors).
      Each returned object contains:

      - ``prob_object.Cs`` of shape ``(n_evi, n_sample, n_childs + n_parents)`` (non-root nodes)
        or ``(n_evi, n_sample, n_childs)`` (root / observed child cases)
      - ``prob_object.ps`` of shape ``(n_evi, n_sample)`` containing log-probabilities

   How sampling is performed
   -------------------------
   1. Compute ancestor order using :func:`get_ancestor_order`.
   2. Deep-copy the needed probability objects (used to call ``P.sample`` only; ``.Cs`` / ``.ps``
      are populated on query nodes at the end).
   3. Iterate over batches of size ``batch_size`` (outer loop). Within each batch, walk the nodes
      in topological order (inner loop), keeping a temporary ``batch_samples`` dictionary
      ``{var_name: tensor(n_evi, n_batch)}`` that holds only the current batch's ancestor values.
   4. For each node in the inner loop:

      - **Observed node** (``node in evidence_df.columns``): broadcast the observed values across
        samples and evaluate ``prob.log_prob`` on the assembled ``[childs | parents]`` rows.
      - **Root node** (no parents): generate ``n_evi * n_batch`` samples via ``P.sample`` and
        reshape to ``(n_evi, n_batch, n_childs)``.
      - **Non-root node**: assemble parent matrix of shape ``(n_evi * n_batch, n_parents)`` from
        ``batch_samples`` (or from ``evidence_df`` if a parent is observed), call
        ``P.sample(Cs_pars=...)``, and reshape back to evidence form.
      - If the node is a query node, append ``Cs_batch`` and ``ps_batch`` to the per-query
        accumulators.

   5. After all batches complete, concatenate the per-batch tensors for each query node along
      the sample dimension (``dim=1``) and attach them as ``P.Cs`` / ``P.ps``.

   Parent handling
   ---------------
   For each parent variable:

   - if the parent is observed in ``evidence_df``, the observed values are broadcast across
     samples,
   - otherwise, sampled values for the current batch are pulled from ``batch_samples``.

   Important
   ---------
   The module assumes each variable appears as a **child** of exactly one probability object.
   If a variable is a child in multiple objects, an ``AssertionError`` is raised.

