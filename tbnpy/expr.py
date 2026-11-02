"""
Safe, vectorised expression compiler for declarative tbnpy models.

Turns a string such as::

    0.5 * Max(Gamma*H - 4*S_u, 0.3*Gamma*H) * H

into a callable that evaluates on batches of torch tensors. Expressions are
parsed with :mod:`ast` and compiled against a whitelist -- there is no ``eval``
and no access to Python builtins, attributes, subscripts, or comprehensions.

The function vocabulary follows GeNIe's equation syntax so that networks
exported from GeNIe can be imported without rewriting their equations.

Two details of that syntax matter and are handled here:

``^`` means exponentiation
    In Python ``^`` is bitwise XOR and binds *looser* than ``*``, so
    ``0.325 * K_a * Gamma * H^2`` would parse as ``(0.325*K_a*Gamma*H) ** 2``.
    The source is rewritten to use ``**`` before parsing, which restores the
    precedence every engineering notation assumes.

``Choose`` selects on a discrete parent's state index
    Everywhere else a discrete variable resolves to its *label value*, so
    ``A + B`` means the sum of the labels. ``Choose(Soil_Type, Pa_sand,
    Pa_clay)`` instead uses ``Soil_Type``'s state index, and is checked at
    compile time to supply exactly one option per state.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Callable, Mapping

import torch

__all__ = ["Expression", "ExpressionError", "NameSpec", "FUNCTION_NAMES"]


class ExpressionError(ValueError):
    """Raised when an expression cannot be compiled or is not permitted."""


# --------------------------------------------------------------------------
# Scope description
# --------------------------------------------------------------------------
CONTINUOUS = "continuous"
NUMERIC_STATE = "numeric_state"     # discrete, labels are numbers
CATEGORICAL = "categorical"         # discrete, labels are not numbers


@dataclass(frozen=True)
class NameSpec:
    """What the compiler knows about one name in scope.

    Attributes
    ----------
    kind:
        One of ``CONTINUOUS``, ``NUMERIC_STATE``, ``CATEGORICAL``.
    n_states:
        Number of states, for discrete names. ``None`` for continuous.
    """

    kind: str
    n_states: int | None = None

    @property
    def is_discrete(self) -> bool:
        return self.kind in (NUMERIC_STATE, CATEGORICAL)


# --------------------------------------------------------------------------
# Numeric helpers. Constants stay Python floats until they meet a tensor, so
# device and dtype are inherited from the data rather than guessed.
# --------------------------------------------------------------------------
def _pair(a, b):
    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
        return a, b
    if isinstance(b, torch.Tensor):
        return torch.as_tensor(a, dtype=b.dtype, device=b.device), b
    return torch.as_tensor(float(a)), torch.as_tensor(float(b))


def _reduce_pairwise(fn, args):
    out = args[0]
    for nxt in args[1:]:
        out = fn(*_pair(out, nxt))
    return out


def _tensorish(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(float(x))


def _choose(selector, *options):
    """Select ``options[selector[i]]`` row-wise."""
    sel = selector.long()
    shape = sel.shape
    expanded = []
    for opt in options:
        if isinstance(opt, torch.Tensor):
            expanded.append(opt.expand(shape) if opt.shape != shape else opt)
        else:
            expanded.append(torch.full(shape, float(opt)))
    ref = next((o for o in expanded if o.is_floating_point()), expanded[0])
    expanded = [o.to(dtype=ref.dtype, device=ref.device) for o in expanded]
    stacked = torch.stack(expanded, dim=-1)
    return stacked.gather(-1, sel.unsqueeze(-1)).squeeze(-1)


def _where(cond, a, b):
    a, b = _pair(a, b)
    return torch.where(cond, a, b)


def _unary(fn):
    return lambda x: fn(_tensorish(x))


# Variadic and fixed-arity functions available inside expressions.
_FUNCTIONS: dict[str, tuple[Callable, int | None]] = {
    # (implementation, required arity; None = variadic, >=1)
    "Abs": (_unary(torch.abs), 1),
    "Exp": (_unary(torch.exp), 1),
    "Ln": (_unary(torch.log), 1),
    "Log": (_unary(torch.log), 1),
    "Log10": (_unary(torch.log10), 1),
    "Log2": (_unary(torch.log2), 1),
    "Sqrt": (_unary(torch.sqrt), 1),
    "Sin": (_unary(torch.sin), 1),
    "Cos": (_unary(torch.cos), 1),
    "Tan": (_unary(torch.tan), 1),
    "ASin": (_unary(torch.asin), 1),
    "ACos": (_unary(torch.acos), 1),
    "ATan": (_unary(torch.atan), 1),
    "Sinh": (_unary(torch.sinh), 1),
    "Cosh": (_unary(torch.cosh), 1),
    "Tanh": (_unary(torch.tanh), 1),
    "Floor": (_unary(torch.floor), 1),
    "Ceil": (_unary(torch.ceil), 1),
    "Round": (_unary(torch.round), 1),
    "Sign": (_unary(torch.sign), 1),
    "ATan2": (lambda y, x: torch.atan2(*_pair(y, x)), 2),
    "Pow": (lambda a, b: a ** b, 2),
    "Min": (lambda *a: _reduce_pairwise(torch.minimum, a), None),
    "Max": (lambda *a: _reduce_pairwise(torch.maximum, a), None),
    "If": (_where, 3),
    "Pi": (lambda: math.pi, 0),
    "E": (lambda: math.e, 0),
}

#: Names callable inside an expression. ``Choose`` is handled separately
#: because its first argument is a state index rather than a value.
FUNCTION_NAMES = frozenset(_FUNCTIONS) | {"Choose"}

_CONSTANTS = {"pi": math.pi, "e": math.e, "inf": math.inf}

_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.BitXor: lambda a, b: a ** b,   # unreachable after ^ rewriting; kept for safety
    ast.Mod: lambda a, b: a % b,
    ast.FloorDiv: lambda a, b: a // b,
}

_CMPOPS = {
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
}


def _rewrite_caret(source: str) -> str:
    """Turn GeNIe's ``^`` into Python's ``**``.

    Safe as a textual substitution because expressions admit no string
    literals, so no ``^`` can occur anywhere but as an operator.
    """
    return source.replace("^", "**")


# --------------------------------------------------------------------------
class Expression:
    """A compiled, vectorised expression.

    Parameters
    ----------
    source:
        The expression text.
    scope:
        ``{name: NameSpec}`` for every name the expression may reference.
    label:
        Optional description used in error messages (e.g. the node name).

    Notes
    -----
    Compilation walks the AST once and builds a tree of closures, so
    evaluation does no parsing and no dictionary lookup of operators.
    """

    def __init__(self, source: str, scope: Mapping[str, NameSpec],
                 label: str | None = None):
        self.source = source
        self.label = label
        self._scope = dict(scope)
        self._names: set[str] = set()
        self._index_names: set[str] = set()

        rewritten = _rewrite_caret(source)
        try:
            tree = ast.parse(rewritten, mode="eval")
        except SyntaxError as exc:
            raise ExpressionError(
                f"{self._where()}cannot parse {source!r}: {exc.msg}"
            ) from exc

        self._fn = self._compile(tree.body)
        self.names = frozenset(self._names)
        self.index_names = frozenset(self._index_names)

    # -- public ------------------------------------------------------------
    def __call__(self, values: Mapping[str, torch.Tensor],
                 indices: Mapping[str, torch.Tensor] | None = None):
        """Evaluate on a batch.

        Parameters
        ----------
        values:
            ``{name: tensor}`` of shape ``(N,)``. Discrete variables carry
            their *label values*.
        indices:
            ``{name: tensor}`` of integer state indices, required only for
            names used as a ``Choose`` selector.
        """
        return self._fn(values, indices or {})

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"

    # -- compilation -------------------------------------------------------
    def _where(self) -> str:
        return f"in {self.label}: " if self.label else ""

    def _fail(self, msg: str) -> ExpressionError:
        return ExpressionError(f"{self._where()}{msg}")

    def _compile(self, node) -> Callable:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise self._fail(
                    f"only numeric constants are allowed, got {node.value!r}"
                )
            val = float(node.value)
            return lambda v, i: val

        if isinstance(node, ast.Name):
            return self._compile_name(node)

        if isinstance(node, ast.UnaryOp):
            inner = self._compile(node.operand)
            if isinstance(node.op, ast.USub):
                return lambda v, i: -inner(v, i)
            if isinstance(node.op, ast.UAdd):
                return inner
            if isinstance(node.op, ast.Not):
                return lambda v, i: ~inner(v, i)
            raise self._fail(f"unsupported unary operator {type(node.op).__name__}")

        if isinstance(node, ast.BinOp):
            op = _BINOPS.get(type(node.op))
            if op is None:
                raise self._fail(f"unsupported operator {type(node.op).__name__}")
            left, right = self._compile(node.left), self._compile(node.right)
            return lambda v, i: op(left(v, i), right(v, i))

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1:
                raise self._fail("chained comparisons are not supported")
            op = _CMPOPS.get(type(node.ops[0]))
            if op is None:
                raise self._fail(f"unsupported comparison {type(node.ops[0]).__name__}")
            left = self._compile(node.left)
            right = self._compile(node.comparators[0])
            return lambda v, i: op(*_pair(left(v, i), right(v, i)))

        if isinstance(node, ast.BoolOp):
            parts = [self._compile(x) for x in node.values]
            if isinstance(node.op, ast.And):
                return lambda v, i: _reduce_pairwise(
                    torch.logical_and, [p(v, i) for p in parts])
            return lambda v, i: _reduce_pairwise(
                torch.logical_or, [p(v, i) for p in parts])

        if isinstance(node, ast.Call):
            return self._compile_call(node)

        raise self._fail(
            f"{type(node).__name__} is not allowed in an expression"
        )

    def _compile_name(self, node: ast.Name) -> Callable:
        name = node.id
        if name in _CONSTANTS:
            val = _CONSTANTS[name]
            return lambda v, i: val

        spec = self._scope.get(name)
        if spec is None:
            known = ", ".join(sorted(self._scope)) or "(nothing in scope)"
            raise self._fail(
                f"unknown name {name!r}; expression may only reference: {known}"
            )
        if spec.kind == CATEGORICAL:
            raise self._fail(
                f"{name!r} has non-numeric states, so it has no value to compute "
                f"with; use Choose({name}, ...) to select on its state instead"
            )
        self._names.add(name)
        return lambda v, i: v[name]

    def _compile_call(self, node: ast.Call) -> Callable:
        if not isinstance(node.func, ast.Name):
            raise self._fail("only plain function names may be called")
        fname = node.func.id
        if node.keywords:
            raise self._fail(f"{fname}() does not take keyword arguments")

        if fname == "Choose":
            return self._compile_choose(node)

        entry = _FUNCTIONS.get(fname)
        if entry is None:
            options = ", ".join(sorted(FUNCTION_NAMES))
            raise self._fail(f"unknown function {fname!r}; available: {options}")

        impl, arity = entry
        args = [self._compile(a) for a in node.args]
        if arity is None:
            if not args:
                raise self._fail(f"{fname}() needs at least one argument")
        elif len(args) != arity:
            raise self._fail(
                f"{fname}() takes {arity} argument(s), got {len(args)}"
            )
        return lambda v, i: impl(*(a(v, i) for a in args))

    def _compile_choose(self, node: ast.Call) -> Callable:
        if len(node.args) < 2:
            raise self._fail(
                "Choose(selector, option0, option1, ...) needs a selector and "
                "at least one option"
            )
        sel_node = node.args[0]
        if not isinstance(sel_node, ast.Name):
            raise self._fail(
                "Choose()'s first argument must be a discrete variable name, "
                "not an expression"
            )
        sel_name = sel_node.id
        spec = self._scope.get(sel_name)
        if spec is None:
            raise self._fail(f"Choose() selector {sel_name!r} is not in scope")
        if not spec.is_discrete:
            raise self._fail(
                f"Choose() selector {sel_name!r} is continuous; a selector must "
                f"be a discrete variable"
            )

        options = [self._compile(a) for a in node.args[1:]]
        if spec.n_states is not None and len(options) != spec.n_states:
            raise self._fail(
                f"Choose({sel_name}, ...) has {len(options)} option(s) but "
                f"{sel_name!r} has {spec.n_states} states; supply exactly one "
                f"option per state, in state order"
            )

        self._names.add(sel_name)
        self._index_names.add(sel_name)

        def run(v, i):
            if sel_name not in i:
                raise ExpressionError(
                    f"state indices for {sel_name!r} were not supplied to the "
                    f"expression, but Choose() needs them"
                )
            return _choose(i[sel_name], *(o(v, i) for o in options))

        return run
