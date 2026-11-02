"""
Probability objects: the protocol, an optional base class, and a model check.

A *probability object* quantifies one node of a Bayesian network. tbnpy
recognises one by what it has, not by what it inherits from:

``childs`` and ``parents``
    Lists of :class:`~tbnpy.variable.Variable`. Every variable of a model is
    the child of exactly one object.
``sample(n_sample=...)`` for a root, ``sample(Cs_pars=...)`` otherwise
    ``Cs_pars`` is ``(N, n_parents)``, in the order of ``parents``. Returns
    ``(Cs, logp)``: ``Cs`` is ``(N,)`` for a single child or
    ``(N, n_childs)``, and ``logp`` is ``(N,)``, the log probability (or log
    density) of each sampled row given its parents.
``log_prob(Cs)``
    ``Cs`` is ``(N, n_childs + n_parents)``, childs first, then parents.
    Returns ``(N,)`` log probabilities.
``device``
    Where the object's tensors live.

A discrete variable's value is its *state index* into ``Variable.values``,
held in any dtype that stores whole numbers exactly. Inference stacks parent
columns with :func:`torch.stack`, which promotes integer and float columns to
a common dtype, so an object casts with ``.long()`` before indexing with them.

:class:`Probability` is an optional base class for such objects. Inheriting
it adds nothing to the protocol; it makes the object readable when printed::

    >>> probs["C"]
    C(P(C | A, B): normal(loc=A + B, scale=0.6))

:func:`describe` gives the same reading, at length, for any probability
object, and :func:`check_model` samples a few rows through a whole model to
find objects that break the protocol before an inference run does.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "Probability", "is_probability", "statement", "domain", "describe",
    "check_model", "ModelCheck", "ModelIssue", "ModelError",
]

#: What :mod:`tbnpy.inference` calls on every probability object.
REQUIRED = ("childs", "parents", "sample", "log_prob")

#: How many names or rows a one-line or tabular reading shows before eliding.
_SHOW_NAMES = 4
_SHOW_STATES = 6
_SHOW_ROWS = 12


# --------------------------------------------------------------------------
# Reading objects and variables
# --------------------------------------------------------------------------
def is_probability(obj) -> bool:
    """Whether ``obj`` has everything tbnpy calls on a probability object."""
    return (all(hasattr(obj, a) for a in REQUIRED)
            and callable(getattr(obj, "sample"))
            and callable(getattr(obj, "log_prob")))


def _join(names, limit=_SHOW_NAMES) -> str:
    names = [str(n) for n in names]
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit - 1]) + f", ..., {names[-1]} ({len(names)})"


def statement(obj) -> str:
    """The distribution an object defines, written ``P(C | A, B)``."""
    childs = _join(v.name for v in obj.childs)
    parents = [v.name for v in obj.parents]
    return f"P({childs} | {_join(parents)})" if parents else f"P({childs})"


def _is_discrete(var) -> bool:
    return isinstance(var.values, (list, np.ndarray))


def _is_continuous(var) -> bool:
    return isinstance(var.values, tuple)


def domain(var) -> str:
    """A variable's domain in words: its states, or its support."""
    if var.values is None:
        return "domain unspecified"
    if _is_continuous(var):
        low, high = var.values
        return f"continuous ({float(low):g}, {float(high):g})"
    states = list(var.values)
    shown = _join(states, limit=_SHOW_STATES)
    return f"discrete, {len(states)} states: {shown}"


def _form(obj) -> str:
    form = getattr(obj, "form", None)
    if not callable(form):
        return ""
    try:
        text = form()
    except Exception:          # a reading must never be the thing that fails
        return ""
    return text if isinstance(text, str) else ""


def _form_details(obj) -> list:
    details = getattr(obj, "form_details", None)
    if not callable(details):
        return []
    try:
        lines = details()
    except Exception:
        return []
    return [str(line) for line in lines or []]


def describe(obj) -> str:
    """Several lines on any probability object: what it is and what it reads.

    Works on any object with ``childs`` and ``parents``. When the object
    defines ``form()`` and ``form_details()`` (as :class:`Probability`
    subclasses may), their text is included.
    """
    lines = [f"{statement(obj)}  [{type(obj).__name__}]"]
    form = _form(obj)
    if form:
        lines.append(f"  form    {form}")
    width = max([len(v.name) for v in list(obj.childs) + list(obj.parents)] + [1])
    for role, variables in (("child ", obj.childs), ("parent", obj.parents)):
        for v in variables:
            lines.append(f"  {role}  {v.name:<{width}}  {domain(v)}")
    lines += ["  " + line for line in _form_details(obj)]
    return "\n".join(lines)


class Probability:
    """Optional base class for probability objects.

    Subclassing it is not required: tbnpy accepts any object that follows the
    protocol in this module's docstring. What it adds is a reading:

    - ``repr(obj)`` gives one line, ``ClassName(P(C | A, B): form)``, so a
      ``probs`` dictionary prints as a list of distributions;
    - ``str(obj)`` (and ``print(obj)``) gives :func:`describe`'s longer
      reading, with each variable's domain.

    Override :meth:`form` to say what the distribution is, and
    :meth:`form_details` to add lines such as a table, for both readings.

    Parameters
    ----------
    childs, parents:
        Lists of :class:`~tbnpy.variable.Variable`.
    device:
        ``"cpu"``, ``"cuda"`` or a :class:`torch.device`.

    Calling this ``__init__`` is optional too; a subclass that sets
    ``childs``, ``parents`` and ``device`` itself gets the same readings.
    """

    def __init__(self, childs, parents=None, device="cpu"):
        self.childs = list(childs)
        self.parents = list(parents or [])
        self.device = device

    def form(self) -> str:
        """One line on the distribution's form, e.g. ``"normal(loc=A + B, scale=0.6)"``."""
        return ""

    def form_details(self) -> list:
        """Extra lines for :func:`describe`, such as a probability table."""
        return []

    def __repr__(self) -> str:
        form = _form(self)
        return f"{type(self).__name__}({statement(self)}{': ' + form if form else ''})"

    def __str__(self) -> str:
        return describe(self)


# --------------------------------------------------------------------------
# Checking a model
# --------------------------------------------------------------------------
@dataclass
class ModelIssue:
    """One thing wrong (``"error"``) or doubtful (``"warning"``) in a model."""

    node: str
    severity: str
    issue: str
    detail: str = ""

    def __str__(self) -> str:
        text = f"[{self.severity}] {self.node}: {self.issue}"
        return text + (f" -- {self.detail}" if self.detail else "")


@dataclass
class ModelCheck:
    """What :func:`check_model` found."""

    issues: list = field(default_factory=list)
    n_objects: int = 0
    n_variables: int = 0
    n_sample: int = 0

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        """No errors. Warnings are allowed."""
        return not self.errors

    def __str__(self) -> str:
        head = (f"Model check: {self.n_objects} probability objects, "
                f"{self.n_variables} variables, {self.n_sample} sampled rows")
        if not self.issues:
            return head + ": no problems found"
        counts = f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        return "\n".join([f"{head}: {counts}"] + [f"  {i}" for i in self.issues])


class ModelError(ValueError):
    """Raised by :func:`check_model` when a model has errors."""

    def __init__(self, check: ModelCheck):
        super().__init__(str(check))
        self.check = check


def _static_issues(probs) -> list:
    """Protocol and graph problems found without sampling."""
    issues = []
    owners = {}                      # variable name -> (node, Variable)
    for node, obj in probs.items():
        missing = [a for a in REQUIRED if not hasattr(obj, a)]
        if missing:
            issues.append(ModelIssue(node, "error", "not a probability object",
                                     f"it has no {', '.join(missing)}"))
            continue
        for attr in ("sample", "log_prob"):
            if not callable(getattr(obj, attr)):
                issues.append(ModelIssue(node, "error", f"{attr} is not callable"))
        for attr in ("childs", "parents"):
            value = getattr(obj, attr)
            if not isinstance(value, list):
                issues.append(ModelIssue(node, "error", f"{attr} is not a list",
                                         f"got {type(value).__name__}"))
            elif not all(hasattr(v, "name") and hasattr(v, "values") for v in value):
                issues.append(ModelIssue(node, "error",
                                         f"{attr} must hold Variables",
                                         "each needs a name and values"))
        if not hasattr(obj, "device"):
            issues.append(ModelIssue(node, "warning", "has no device attribute"))
        if any(i.node == node and i.severity == "error" for i in issues):
            continue
        if not obj.childs:
            issues.append(ModelIssue(node, "error", "has no childs"))
        for v in obj.childs:
            if v.name in owners:
                issues.append(ModelIssue(node, "error",
                                         f"{v.name!r} is a child of two objects",
                                         f"also of {owners[v.name][0]!r}"))
            else:
                owners[v.name] = (node, v)
            if v.values is None:
                issues.append(ModelIssue(node, "warning",
                                         f"{v.name!r} has no domain",
                                         "its sampled values cannot be checked"))
        own = {v.name for v in obj.childs}
        for v in obj.parents:
            if v.name in own:
                issues.append(ModelIssue(node, "error",
                                         f"{v.name!r} is both a child and a parent"))

    if any(i.severity == "error" for i in issues):
        return issues

    for node, obj in probs.items():
        for v in obj.parents:
            owner = owners.get(v.name)
            if owner is None:
                issues.append(ModelIssue(node, "error",
                                         f"parent {v.name!r} is no object's child"))
            elif not _same_domain(v, owner[1]):
                issues.append(ModelIssue(
                    node, "warning",
                    f"parent {v.name!r} has a different domain from its owner's",
                    f"here {domain(v)}; {owner[0]!r} declares {domain(owner[1])}"))
    if any(i.severity == "error" for i in issues):
        return issues

    order = _order(probs, owners)
    if order is None:
        issues.append(ModelIssue("(model)", "error", "the graph has a cycle"))
    return issues


def _same_domain(a, b) -> bool:
    if a is b or (a.values is None and b.values is None):
        return True
    if _is_discrete(a) and _is_discrete(b):
        return [str(s) for s in a.values] == [str(s) for s in b.values]
    if _is_continuous(a) and _is_continuous(b):
        return tuple(float(x) for x in a.values) == tuple(float(x) for x in b.values)
    return False


def _order(probs, owners):
    """Node names, parents first; ``None`` when the graph has a cycle."""
    needs = {node: {owners[v.name][0] for v in obj.parents} - {node}
             for node, obj in probs.items()}
    order, ready = [], [n for n, deps in needs.items() if not deps]
    while ready:
        node = ready.pop(0)
        order.append(node)
        for other, deps in needs.items():
            if node in deps:
                deps.discard(node)
                if not deps:
                    ready.append(other)
    return order if len(order) == len(probs) else None


def _check_values(node, var, column, issues):
    """Sampled values against a variable's domain."""
    if var.values is None:
        return
    column = column.detach().to("cpu", torch.float64)
    if torch.isnan(column).any():
        issues.append(ModelIssue(node, "error", f"sampled NaN for {var.name!r}"))
        return
    if _is_continuous(var):
        low, high = (float(b) for b in var.values)
        outside = (column < low) | (column > high)
        if outside.any():
            issues.append(ModelIssue(
                node, "error", f"sampled {var.name!r} outside its support",
                f"{int(outside.sum())} of {len(column)} rows outside ({low:g}, {high:g}), "
                f"e.g. {column[outside][0].item():g}"))
        return
    n_states = len(var.values)
    whole = column == column.round()
    if not whole.all():
        issues.append(ModelIssue(
            node, "error", f"sampled {var.name!r} is not a state index",
            f"e.g. {column[~whole][0].item():g}; a discrete value is an index "
            f"into {var.name}.values"))
        return
    n_composite = 2 ** n_states - 1 if n_states < 63 else float("inf")
    bad = (column < 0) | (column >= n_composite)
    if bad.any():
        issues.append(ModelIssue(
            node, "error", f"sampled {var.name!r} out of range",
            f"e.g. {int(column[bad][0])}; {var.name!r} has {n_states} states "
            f"(0..{n_states - 1})"))
    elif (column >= n_states).any():
        issues.append(ModelIssue(
            node, "warning", f"sampled a composite state of {var.name!r}",
            f"a value >= {n_states} stands for a set of states"))


def _sample_one(node, obj, columns, n, issues):
    """Sample ``obj`` once; return its child columns, or ``None`` on failure."""
    try:
        obj = copy.deepcopy(obj)
    except Exception as exc:
        issues.append(ModelIssue(node, "error", "cannot be deep-copied",
                                 f"inference.sample copies every object: "
                                 f"{type(exc).__name__}: {exc}"))
        return None

    pars = None
    try:
        if obj.parents:
            pars = torch.stack([columns[v.name] for v in obj.parents], dim=1)
            out = obj.sample(Cs_pars=pars)
        else:
            out = obj.sample(n_sample=n)
    except Exception as exc:
        issues.append(ModelIssue(node, "error", "sample() raised",
                                 f"{type(exc).__name__}: {exc}"))
        return None

    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        issues.append(ModelIssue(node, "error", "sample() must return (Cs, logp)",
                                 f"got {type(out).__name__}"))
        return None
    Cs, logp = out
    n_childs = len(obj.childs)
    if not (torch.is_tensor(Cs) and torch.is_tensor(logp)):
        issues.append(ModelIssue(node, "error", "sample() must return tensors",
                                 f"got ({type(Cs).__name__}, {type(logp).__name__})"))
        return None
    if Cs.ndim == 1 and n_childs > 1:
        issues.append(ModelIssue(node, "error", "Cs is 1-D for several childs",
                                 f"expected ({n}, {n_childs}); inference would give "
                                 f"every child the same column"))
        return None
    expected = [(n,), (n, n_childs)] if n_childs == 1 else [(n, n_childs)]
    if tuple(Cs.shape) not in expected:
        issues.append(ModelIssue(node, "error", "Cs has the wrong shape",
                                 f"got {tuple(Cs.shape)}, expected "
                                 f"{' or '.join(str(e) for e in expected)}"))
        return None
    if tuple(logp.shape) != (n,):
        issues.append(ModelIssue(node, "error", "logp has the wrong shape",
                                 f"got {tuple(logp.shape)}, expected ({n},)"))
        return None

    childs = Cs.reshape(n, n_childs)
    for j, var in enumerate(obj.childs):
        _check_values(node, var, childs[:, j], issues)

    lp = logp.detach().to("cpu", torch.float64)
    if torch.isnan(lp).any() or (lp == float("inf")).any():
        issues.append(ModelIssue(node, "error", "logp holds NaN or +inf"))
    elif (lp == -float("inf")).any():
        issues.append(ModelIssue(node, "error",
                                 "logp is -inf for some sampled rows",
                                 "the object samples values it gives zero probability"))

    full = childs if pars is None else torch.cat([childs, pars.to(childs.device)], dim=1)
    try:
        again = obj.log_prob(full)
    except NotImplementedError:
        issues.append(ModelIssue(node, "warning", "log_prob is not implemented",
                                 "evidence on this node and MCMC cannot use it"))
    except Exception as exc:
        issues.append(ModelIssue(node, "error", "log_prob() raised",
                                 f"{type(exc).__name__}: {exc}"))
    else:
        if not torch.is_tensor(again) or tuple(again.shape) != (n,):
            shape = tuple(again.shape) if torch.is_tensor(again) else type(again).__name__
            issues.append(ModelIssue(node, "error", "log_prob has the wrong shape",
                                     f"got {shape}, expected ({n},)"))
        else:
            again = again.detach().to("cpu", torch.float64)
            agree = torch.isclose(again, lp, rtol=1e-4, atol=1e-4) | (again == lp)
            if not agree.all():
                k = int((~agree).nonzero()[0])
                issues.append(ModelIssue(
                    node, "error", "log_prob disagrees with sample's logp",
                    f"row {k}: log_prob {again[k].item():.6g}, sample "
                    f"{lp[k].item():.6g}; columns are childs then parents"))
    return {var.name: childs[:, j] for j, var in enumerate(obj.childs)}


def check_model(probs, n_sample=8, seed=0, raise_on_error=True) -> ModelCheck:
    """Check that every object in a model follows the protocol, by using it.

    The check has two parts. Without sampling, it confirms that each object
    has ``childs``, ``parents``, ``sample`` and ``log_prob``; that every
    variable is the child of exactly one object; that every parent is some
    object's child, with the same domain; and that the graph has no cycle.
    Then it forward-samples ``n_sample`` rows through the whole model, as
    :func:`tbnpy.inference.sample` does, and for each object confirms:

    - ``sample`` returns ``(Cs, logp)`` of the right shapes;
    - sampled values lie in their variables' domains (state indices for
      discrete variables, the support for continuous ones);
    - ``logp`` is finite;
    - ``log_prob`` of the sampled rows equals ``logp``.

    The random number generator's state is restored afterwards, so a check
    never changes the results of a seeded run that follows it.

    Parameters
    ----------
    probs:
        ``{node name: probability object}``.
    n_sample:
        Rows to sample. A few are enough to find a broken object.
    seed:
        Seed for the check's own sampling, so that it is reproducible.
    raise_on_error:
        Raise :class:`ModelError` when there are errors. With ``False``, the
        :class:`ModelCheck` is returned either way.

    Returns
    -------
    ModelCheck
        ``print`` it to read the issues; ``.ok`` is ``True`` without errors.
    """
    if not isinstance(probs, dict):
        check = ModelCheck([ModelIssue("(model)", "error", "probs must be a dict",
                                       f"got {type(probs).__name__}")])
        if raise_on_error:
            raise ModelError(check)
        return check

    issues = _static_issues(probs)
    n_vars = sum(len(getattr(o, "childs", []) or []) for o in probs.values())
    check = ModelCheck(issues, n_objects=len(probs), n_variables=n_vars,
                       n_sample=0)
    if check.errors:
        if raise_on_error:
            raise ModelError(check)
        return check

    owners = {v.name: (node, v) for node, obj in probs.items() for v in obj.childs}
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    columns, failed = {}, set()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        for node in _order(probs, owners):
            obj = probs[node]
            blocked = sorted({owners[v.name][0] for v in obj.parents} & failed)
            if blocked:
                failed.add(node)
                issues.append(ModelIssue(node, "warning", "not sampled",
                                         f"its parent object {blocked[0]!r} failed"))
                continue
            out = _sample_one(node, obj, columns, n_sample, issues)
            if out is None:
                failed.add(node)
            else:
                columns.update(out)

    check.n_sample = n_sample
    if check.errors and raise_on_error:
        raise ModelError(check)
    return check
