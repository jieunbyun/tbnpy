tbnpy.spec
==========

A model can be written as data, a JSON file of variables and nodes, instead of as
Python classes. This module defines that format and validates it;
:mod:`tbnpy.compile` turns it into the ``varis`` and ``probs`` the rest of tbnpy uses.

A spec has five node types:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - ``type``
     - Meaning
   * - ``distribution``
     - A parametric family (``normal``, ``lognormal``, ``categorical``, ...) whose
       parameters are expressions over the parents (:mod:`tbnpy.expr`).
   * - ``equation``
     - A deterministic function of the parents, with an optional ``tolerance``.
   * - ``discrete_cpt``
     - An explicit table over discrete child and parents; compiles to
       :class:`~tbnpy.cpt.Cpt`.
   * - ``discretise``
     - Bins one continuous parent into the child's discrete states.
   * - ``custom``
     - A hand-written probability class, imported by module and class name. Anything
       the other types cannot express is written in Python and referenced from here.

The ABCDE example's model as a spec is
``examples/ABCDE/abcde.tbn.json``; see :doc:`../examples/abcde_example`.

.. literalinclude:: ../../../examples/ABCDE/abcde.tbn.json
   :language: json

API
---

.. automodule:: tbnpy.spec
   :members: ModelSpec, VariableSpec, NodeSpec, DistributionNode, EquationNode, DiscreteCptNode, DiscretiseNode, CustomNode, AnalysisSpec, SpecError, load_spec, save_spec
