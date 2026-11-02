tbnpy.probability
=================

Overview
--------

A **probability object** quantifies one node of a Bayesian network. tbnpy
recognises one by what it has, not by what it inherits from, so a class written
for one model needs no tbnpy base class to work. This module makes the protocol
explicit and adds three things on top of it:

- :class:`~tbnpy.probability.Probability`, an *optional* base class that makes an
  object readable when it is printed;
- :func:`~tbnpy.probability.describe`, the same reading for any probability object,
  whether it inherits the base class or not;
- :func:`~tbnpy.probability.check_model`, which samples a few rows through a whole
  model to find an object that breaks the protocol before an inference run does.

:class:`~tbnpy.cpt.Cpt` and every node compiled by :mod:`tbnpy.compile` inherit
:class:`~tbnpy.probability.Probability`. The three names are also available as
``tbnpy.Probability``, ``tbnpy.describe`` and ``tbnpy.check_model``.

The protocol
------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Member
     - Meaning
   * - ``childs``
     - List of child :class:`~tbnpy.variable.Variable`. Every variable of a model is
       the child of exactly one object.
   * - ``parents``
     - List of parent :class:`~tbnpy.variable.Variable`.
   * - ``sample(n_sample=N)``
     - For a root (no parents). Returns ``(Cs, logp)``.
   * - ``sample(Cs_pars=...)``
     - For an object with parents. ``Cs_pars`` is ``(N, n_parents)``, in the order of
       ``parents``. Returns ``(Cs, logp)``.
   * - ``log_prob(Cs)``
     - ``Cs`` is ``(N, n_childs + n_parents)``: childs first, then parents. Returns
       ``(N,)`` log probabilities (or log densities).
   * - ``device``
     - Where the object's tensors live (``"cpu"``, ``"cuda"`` or a ``torch.device``).

In ``sample``'s return value, ``Cs`` is ``(N,)`` for a single child or
``(N, n_childs)``, and ``logp`` is ``(N,)``: the log probability of each sampled row
given its parents. ``log_prob`` of the same rows must return the same numbers.

Discrete values
^^^^^^^^^^^^^^^

A discrete variable's value is its **state index** into ``Variable.values``, not its
label. It may be held in any dtype that stores whole numbers exactly: inference
builds ``Cs_pars`` with :func:`torch.stack`, which promotes a mix of integer and float
columns to a common dtype. So a class receiving discrete parents should cast them
with ``.long()`` before using them as indices, and look up labels with
``Variable.values``:

.. code-block:: python

   A_values = torch.tensor(A.values, dtype=torch.float32)   # labels, by state index
   mean = A_values[Cs_pars[:, 0].long()]

Reading an object
-----------------

With the base class, typing an object's name shows the distribution it defines, and
``print`` gives each variable's domain:

.. code-block:: python

   >>> probs
   {'A': Cpt(P(A): table of 2 rows), 'B': Cpt(P(B): table of 3 rows),
    'C': C(P(C | A, B): normal(loc=A + B, scale=0.6)), ...}

   >>> print(probs['C'])
   P(C | A, B)  [C]
     form    normal(loc=A + B, scale=0.6)
     child   C  continuous (-inf, inf)
     parent  A  discrete, 2 states: -0.3, 0.3
     parent  B  discrete, 3 states: -0.2, 0.0, 0.2

The *form* is what the class says it is. Override
:meth:`~tbnpy.probability.Probability.form` to return one line, and
:meth:`~tbnpy.probability.Probability.form_details` to add lines such as a table
(as :class:`~tbnpy.cpt.Cpt` does). Both are optional:

.. code-block:: python

   from tbnpy.probability import Probability

   class C(Probability):
       def __init__(self, childs, parents, sigma=0.6, device="cpu"):
           super().__init__(childs, parents, device)    # sets childs, parents, device
           self.sigma = sigma

       def form(self):
           return f"normal(loc=A + B, scale={self.sigma:g})"

       def sample(self, Cs_pars): ...
       def log_prob(self, Cs): ...

For an object that does not inherit the base class, ``tbnpy.describe(obj)`` prints the
same reading (without the form, unless the object has a ``form()`` method).

Checking a model
----------------

.. code-block:: python

   >>> print(tbnpy.check_model(probs))
   Model check: 6 probability objects, 6 variables, 8 sampled rows: no problems found

:func:`~tbnpy.probability.check_model` raises :class:`~tbnpy.probability.ModelError`
when it finds an error, listing every issue; pass ``raise_on_error=False`` to receive
the :class:`~tbnpy.probability.ModelCheck` instead. It reports:

- **Without sampling:** a missing ``childs``, ``parents``, ``sample`` or ``log_prob``; a
  variable that is the child of two objects; a parent that is no object's child; a
  parent declared with a different domain from its owner's (a warning); a cycle.
- **By sampling** ``n_sample`` rows in topological order, as
  :func:`tbnpy.inference.sample` does: an object that cannot be deep-copied or whose
  ``sample`` raises; ``Cs`` or ``logp`` of the wrong shape; a discrete value that is not
  a state index or a continuous value outside its support; a ``logp`` that is NaN or
  infinite; and a ``log_prob`` that disagrees with ``sample``'s ``logp``. An object
  whose ``log_prob`` raises ``NotImplementedError`` gets a warning, since forward
  sampling does not need it.

The check uses its own seed and restores the random number generator's state when it
finishes, so running it never changes the results of a seeded run that follows.
Inference does not run it for you: call it once after building a model.

API
---

.. currentmodule:: tbnpy.probability

.. autoclass:: Probability
   :members: form, form_details

.. autofunction:: describe

.. autofunction:: check_model

.. autoclass:: ModelCheck
   :members: ok, errors, warnings

.. autoclass:: ModelIssue

.. autoclass:: ModelError

.. autofunction:: is_probability

.. autofunction:: statement

.. autofunction:: domain
