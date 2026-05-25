Custom probability models
=========================

When to use CPT vs custom models
--------------------------------

TBNPy represents a Bayesian network as:

1. **Variables** (:class:`~tbnpy.variable.Variable`) that define names and domains
2. **Probability objects** that define conditional relationships for each node

For **discrete, tractable** variables (small state spaces), you can use
:class:`~tbnpy.cpt.Cpt` to define probability tables.

For **continuous** variables, **non-tabular** conditionals, or **deterministic** relationships,
you should implement a **custom probability model**.

A probability object is any Python class that exposes a small interface used by
:mod:`tbnpy.inference` and :mod:`tbnpy.adaptiveMH`.

Minimal interface
-----------------

A custom probability model class should define:

.. rubric:: Required attributes

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Attribute
     - Meaning
   * - ``childs``
     - List of child variables (each a :class:`~tbnpy.variable.Variable`)
   * - ``parents``
     - List of parent variables (each a :class:`~tbnpy.variable.Variable`)
   * - ``device``
     - Torch device (``"cpu"`` or ``"cuda"``), or ``torch.device``

.. rubric:: Required methods

.. py:method:: prob.sample(Cs_pars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]

   Generate samples of the child variable(s) conditional on parent samples.

   Parameters
   ----------
   Cs_pars
      Parent realisations as a tensor of shape ``(n_samples, n_parents)``.

      - For **discrete** parents, entries are typically integer state indices.
      - For **continuous** parents, entries are floats.

   Returns
   -------
   (Cs_child, logp)
      ``Cs_child`` is a tensor of shape ``(n_samples, n_childs)`` containing generated
      child samples.

      ``logp`` is a tensor of shape ``(n_samples,)`` containing the log-probability
      of the generated samples under the model.

   Notes
   -----
   The return value ``logp`` is strongly recommended because it enables downstream
   inference routines to track likelihood contributions efficiently.

.. py:method:: prob.log_prob(Cs: torch.Tensor) -> torch.Tensor

   Evaluate log-probability for provided samples.

   Parameters
   ----------
   Cs
      Tensor of shape ``(n_samples, n_childs + n_parents)`` storing realisations of:

      - child variables first, then
      - parent variables (in the same order as ``parents``)

   Returns
   -------
   torch.Tensor
      Log-probabilities of shape ``(n_samples,)``.

Device and performance notes
----------------------------

- Implement methods using **PyTorch tensor operations** so computations can run on CPU or GPU.
- Keep shapes consistent: TBNPy relies heavily on batched tensor operations.
- Deterministic relationships can be represented by returning:
  - child samples computed by a function of parent samples, and
  - a log-probability tensor (often zeros if treated as deterministic in your factorisation).

Examples in the ABCDE tutorial
------------------------------

The ABCDE example demonstrates three common cases:

- ``C | A, B`` : continuous conditional (Gaussian with parameters depending on discrete parents)
- ``OC | C``   : noisy observation model
- ``E | C, D`` : deterministic relationship

See the ABCDE example page for working code snippets.
