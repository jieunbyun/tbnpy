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

Every node owns one variable, named by ``child`` (by default, the node's own name),
except a ``custom`` node, which may own several. It lists them in ``childs``, and its
class receives them in that order, as a field drawn jointly over many sites needs:

.. code-block:: json

   {"name": "ground-motion", "type": "custom",
    "module": "my_hazard", "cls": "GroundMotionField",
    "childs": ["SA::site1", "SA::site2", "SA::site3"],
    "parents": [],
    "kwargs": {"mw": 7.0, "sites": [[36.6, -121.9], [36.7, -121.6], [36.9, -121.4]]}}

``kwargs`` are the class's constructor arguments, so a spec built from custom nodes
records each probability object as its class, its variables and its arguments. Files
using ``childs`` declare ``"spec_version": 2``; earlier files are version 1 and load
unchanged.

The ABCDE example's model as a spec is
``examples/ABCDE/abcde.tbn.json``; see :doc:`../examples/abcde_example`.

.. literalinclude:: ../../../examples/ABCDE/abcde.tbn.json
   :language: json

API
---

.. automodule:: tbnpy.spec
   :members: ModelSpec, VariableSpec, NodeSpec, DistributionNode, EquationNode, DiscreteCptNode, DiscretiseNode, CustomNode, SpecError, load_spec, save_spec
