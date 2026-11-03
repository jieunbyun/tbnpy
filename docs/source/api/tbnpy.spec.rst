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
records each probability object as its class, its variables and its arguments.

Parameter sets
--------------

Many nodes often share one set of arguments: every bridge of a class has the same
fragility curve, every edge the same closure rule. Copying the numbers into each node
would let them drift apart and suggest they are independent choices. Instead, the spec
names the set once in ``parameter_sets``, and each node says which set it uses in
``parameter_set``. Each node keeps its own probability object, so one bridge's damage
can still be observed or intervened on alone:

.. code-block:: json

   {"name": "bridges", "spec_version": 3,
    "variables": [
      {"name": "SA::b1", "kind": "continuous", "support": [0, null]},
      {"name": "SA::b2", "kind": "continuous", "support": [0, null]},
      {"name": "Damage::b1", "kind": "discrete", "states": ["none", "slight", "moderate", "extensive", "complete"]},
      {"name": "Damage::b2", "kind": "discrete", "states": ["none", "slight", "moderate", "extensive", "complete"]}],
    "parameter_sets": {
      "hazus.HWB5":  {"medians": [0.25, 0.35, 0.45, 0.70], "betas": 0.6},
      "hazus.HWB11": {"medians": [0.90, 0.90, 1.10, 1.50], "betas": 0.6}},
    "nodes": [
      {"name": "ground-motion", "type": "custom", "module": "my_hazard", "cls": "GroundMotionField",
       "childs": ["SA::b1", "SA::b2"], "parents": [],
       "kwargs": {"mw": 7.0, "sites": [[36.6, -121.9], [36.7, -121.6]]}},
      {"name": "Damage::b1", "type": "custom", "module": "my_fragility", "cls": "Fragility",
       "parents": ["SA::b1"], "parameter_set": "hazus.HWB11"},
      {"name": "Damage::b2", "type": "custom", "module": "my_fragility", "cls": "Fragility",
       "parents": ["SA::b2"], "parameter_set": "hazus.HWB5"}]}

The class receives the set's arguments together with the node's own ``kwargs``, so it
is written as usual, with the curve as constructor arguments:

.. code-block:: python

   class Fragility(Probability):
       def __init__(self, childs, parents, medians, betas, device="cpu"):
           ...

``kwargs`` hold what is particular to one node, and the set what nodes share; an
argument given in both is refused rather than silently overridden. Retrofitting one
bridge is then a change of the set its node names. A compiled node keeps the name as
``probs["Damage::b1"].spec.parameter_set``, and
:meth:`ModelSpec.arguments <tbnpy.spec.ModelSpec.arguments>` gives a node's full
arguments.

Versions
--------

A spec says which version of the format it uses in ``spec_version``. Version 2 added
``childs``, and version 3 added ``parameter_sets``. Files of earlier versions load
unchanged and keep their version when saved; a file of a later version than the
installed tbnpy understands is refused, not misread.

Prestored data
--------------

``kwargs`` are JSON, so data too large or too binary to write inline (rule sets, lookup
tables, grids, a surrogate model's weights) is named by file instead, relative to the
spec's folder. A custom class that has a ``directory`` parameter receives that folder:

.. code-block:: json

   {"name": "SystemState::n3", "type": "custom",
    "module": "my_system", "cls": "SystemState",
    "parents": ["EdgeOpen::e1", "EdgeOpen::e2"],
    "kwargs": {"origin": "n3", "rules": "rules/n3.pt"},
    "parameter_set": "travel-time"}

Here ``"travel-time": {"network": "network.json", "target_add_h": 0.5}`` in
``parameter_sets`` is shared by every origin's node, while each node keeps its own
origin and rule file. The class reads both files:

.. code-block:: python

   class SystemState(Probability):
       def __init__(self, childs, parents, origin, network, target_add_h, rules,
                    directory=None, device="cpu"):
           super().__init__(childs, parents, device)
           self.graph = load_graph(Path(directory) / network)
           self.rules = torch.load(Path(directory) / rules)

:func:`~tbnpy.compile.compile_file` passes the spec file's folder;
:func:`~tbnpy.compile.compile_model` takes it as ``directory=...`` for a spec built in
memory. A spec is then portable together with the files it names: copy the folder, not
the JSON alone. A class without a ``directory`` parameter never receives one.

The ABCDE example's model as a spec is
``examples/ABCDE/abcde.tbn.json``; see :doc:`../examples/abcde_example`.

.. literalinclude:: ../../../examples/ABCDE/abcde.tbn.json
   :language: json

API
---

.. automodule:: tbnpy.spec
   :members: ModelSpec, VariableSpec, NodeSpec, DistributionNode, EquationNode, DiscreteCptNode, DiscretiseNode, CustomNode, SpecError, load_spec, save_spec
