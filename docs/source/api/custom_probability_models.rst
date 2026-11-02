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
you either write the node declaratively in a spec (:mod:`tbnpy.spec`), or implement a
**custom probability model**: a Python class.

A probability object is any Python class that exposes a small interface used by
:mod:`tbnpy.inference` and :mod:`tbnpy.adaptiveMH`. It need not inherit anything, but
inheriting :class:`tbnpy.Probability <tbnpy.probability.Probability>` makes it readable
when printed (see :doc:`tbnpy.probability`).

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
   :no-index:

   Generate samples of the child variable(s) conditional on parent samples. A root
   (an object without parents) is called as ``prob.sample(n_sample=N)`` instead.

   Parameters
   ----------
   Cs_pars
      Parent realisations as a tensor of shape ``(n_samples, n_parents)``, in the order
      of ``parents``.

      - For **discrete** parents, entries are state indices (whole numbers, possibly in
        a float tensor: cast with ``.long()`` before indexing).
      - For **continuous** parents, entries are floats.

   Returns
   -------
   (Cs_child, logp)
      ``Cs_child`` is a tensor of shape ``(n_samples, n_childs)``, or ``(n_samples,)``
      for a single child, containing generated child samples. Discrete children are
      state indices.

      ``logp`` is a tensor of shape ``(n_samples,)`` containing the log-probability
      of the generated samples under the model.

   Notes
   -----
   ``logp`` must equal ``log_prob`` of the same rows: :func:`tbnpy.check_model
   <tbnpy.probability.check_model>` tests this, and likelihood-weighted inference relies on it.

.. py:method:: prob.log_prob(Cs: torch.Tensor) -> torch.Tensor
   :no-index:

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

Making the object readable
--------------------------

Inherit :class:`~tbnpy.probability.Probability` and, optionally, say what the
distribution is in :meth:`~tbnpy.probability.Probability.form`:

.. code-block:: python

   import torch
   from torch.distributions import Normal
   from tbnpy.probability import Probability

   class OC(Probability):
       """OC | C ~ Normal(C, 0.5^2)"""

       def __init__(self, childs, parents, device="cpu"):
           super().__init__(childs, parents, device)
           self.std = 0.5

       def form(self):
           return f"normal(loc={self.parents[0].name}, scale={self.std:g})"

       def sample(self, Cs_pars):
           dist = Normal(Cs_pars[:, 0], self.std)
           Cs = dist.sample()
           return Cs, dist.log_prob(Cs)

       def log_prob(self, Cs):
           return Normal(Cs[:, 1], self.std).log_prob(Cs[:, 0])

Then ``probs["OC"]`` prints ``OC(P(OC | C): normal(loc=C, scale=0.5))`` rather than
``<oc.OC object at 0x...>``, and ``print(probs["OC"])`` lists each variable with its
domain. Override :meth:`~tbnpy.probability.Probability.form_details` to add more lines,
such as a table of parameters.

Checking the object
-------------------

Before running inference, check the whole model once:

.. code-block:: python

   import tbnpy

   print(tbnpy.check_model(probs))

It samples a few rows through every object and reports any that returns the wrong
shapes, samples outside its variables' domains, or has a ``log_prob`` that disagrees
with its ``sample``. See :func:`tbnpy.check_model <tbnpy.probability.check_model>`.

Device and performance notes
----------------------------

- Implement methods using **PyTorch tensor operations** so computations can run on CPU or GPU.
- Keep shapes consistent: TBNPy relies heavily on batched tensor operations.
- Deterministic relationships can be represented by returning:

  - child samples computed by a function of parent samples, and
  - a log-probability tensor (often zeros if treated as deterministic in your factorisation).

- :func:`tbnpy.inference.sample` deep-copies every object before sampling, so an object
  must survive :func:`copy.deepcopy`.

Examples in the ABCDE tutorial
------------------------------

The ABCDE example demonstrates three common cases:

- ``C | A, B`` : continuous conditional (Gaussian with parameters depending on discrete parents)
- ``OC | C``   : noisy observation model
- ``E | C, D`` : deterministic relationship

See :doc:`../examples/abcde_example` for working code, both as classes and as a spec.
