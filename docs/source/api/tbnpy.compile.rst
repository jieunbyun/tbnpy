tbnpy.compile
=============

Compiles a :class:`~tbnpy.spec.ModelSpec` into runnable tbnpy objects.
:func:`~tbnpy.compile.compile_file` loads and compiles a spec file in one call:

.. code-block:: python

   from tbnpy.compile import compile_file

   spec, varis, probs = compile_file("examples/ABCDE/abcde.tbn.json")

Every compiled node is a :class:`~tbnpy.probability.Probability`, so it prints as the
distribution it defines, with the form taken from the spec:

.. code-block:: python

   >>> probs["C"]
   DistributionSpecNode(P(C | A, B): normal(loc=A + B, scale=0.6))
   >>> probs["E"]
   EquationSpecNode(P(E | C, D): E = If(D == 0, C, 0))

Each node also keeps the :class:`~tbnpy.spec.NodeSpec` it came from as ``.spec``, and
that node's free-form ``metadata`` as ``.spec_metadata``, for inference algorithms
that need model-level hints.

API
---

.. automodule:: tbnpy.compile
   :members: compile_model, compile_file, CompileError, FAMILIES
