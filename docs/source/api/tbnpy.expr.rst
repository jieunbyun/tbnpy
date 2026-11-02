tbnpy.expr
==========

The expression language of :mod:`tbnpy.spec`'s ``distribution`` parameters and
``equation`` nodes. Expressions are parsed with :mod:`ast` against a whitelist (there
is no ``eval``) and evaluate on batches of tensors. The function names follow
GeNIe's equation syntax, so equations exported from GeNIe compile unchanged.

In an expression, a discrete parent stands for its **label value** (``A + B`` adds the
labels of ``A`` and ``B``); ``Choose(X, a, b, ...)`` instead selects on ``X``'s state
index. ``^`` means exponentiation, as in GeNIe.

API
---

.. automodule:: tbnpy.expr
   :members: Expression, ExpressionError, NameSpec, FUNCTION_NAMES
