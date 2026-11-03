"""
Compile a :class:`tbnpy.spec.ModelSpec` into runnable tbnpy objects.

``compile_model`` returns exactly the two things the rest of the library works
with -- a ``{name: Variable}`` table and a ``{name: node}`` table -- and
nothing else. It has no opinion about how those are then used, which is what
keeps bespoke inference algorithms unconstrained by the model format.

Every generated node follows the tbnpy protocol: ``childs``, ``parents``,
``sample(n_sample=...)`` for roots or ``sample(Cs_pars=...)`` for the rest,
and ``log_prob(Cs)`` with columns ordered childs-then-parents. Each is a
:class:`tbnpy.probability.Probability`, whose ``form()`` comes from the spec,
so it prints as the distribution it defines. Nodes also
carry ``spec`` (the originating :class:`~tbnpy.spec.NodeSpec`) and
``spec_metadata``, so an algorithm that needs model-level hints can read them
without the compiler having to anticipate it.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.distributions as D

from tbnpy import cpt as cpt_mod
from tbnpy import variable as variable_mod
from tbnpy.probability import Probability
from tbnpy.expr import (CATEGORICAL, CONTINUOUS, NUMERIC_STATE, Expression,
                        NameSpec)
from tbnpy.spec import (CustomNode, DiscreteCptNode, DiscretiseNode,
                        DistributionNode, EquationNode, ModelSpec, NodeSpec,
                        SpecError, load_spec)

__all__ = ["compile_model", "compile_file", "CompileError", "FAMILIES"]

NEG_INF = -float("inf")


class CompileError(ValueError):
    """Raised when a valid spec cannot be turned into runnable objects."""


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------
def build_variables(spec: ModelSpec) -> dict:
    """Create the ``{name: Variable}`` table described by ``spec``."""
    varis = {}
    for v in spec.variables:
        if v.is_discrete:
            values = list(v.states)
        else:
            low, high = (v.support if v.support is not None
                         else (-torch.inf, torch.inf))
            values = (low, high)
        varis[v.name] = variable_mod.Variable(name=v.name, values=values)
    return varis


def _name_spec(v) -> NameSpec:
    if not v.is_discrete:
        return NameSpec(CONTINUOUS)
    kind = NUMERIC_STATE if v.has_numeric_states else CATEGORICAL
    return NameSpec(kind, v.n_states)


def _scope_for(node: NodeSpec, variables: Mapping) -> dict:
    return {p: _name_spec(variables[p]) for p in node.parents}


def _broadcast(value, n: int, device):
    """Make an expression result a length-``n`` tensor on ``device``."""
    if isinstance(value, torch.Tensor):
        t = value.to(device)
        if t.dim() == 0:
            return t.expand(n)
        return t
    return torch.full((n,), float(value), device=device)


# --------------------------------------------------------------------------
# Distribution families
# --------------------------------------------------------------------------
def _loguniform(p):
    base = D.Uniform(torch.log(p["low"]), torch.log(p["high"]),
                     validate_args=False)
    return D.TransformedDistribution(base, D.transforms.ExpTransform(),
                                     validate_args=False)


class _TruncatedPoisson:
    """Poisson on ``{0, ..., n_states - 1}``, renormalised over that range."""

    def __init__(self, rate, n_states):
        self.rate = rate.clamp_min(1e-30)
        self.n_states = n_states
        j = torch.arange(n_states, device=rate.device, dtype=torch.float32)
        log_unnorm = (j.unsqueeze(0) * torch.log(self.rate).unsqueeze(1)
                      - self.rate.unsqueeze(1) - torch.lgamma(j + 1.0).unsqueeze(0))
        self._log_pmf = log_unnorm - torch.logsumexp(log_unnorm, dim=1, keepdim=True)

    def sample(self):
        return torch.multinomial(self._log_pmf.exp(), 1).squeeze(1)

    def log_prob(self, k):
        k = k.long()
        in_range = (k >= 0) & (k < self.n_states)
        safe = k.clamp(0, self.n_states - 1).unsqueeze(1)
        lp = self._log_pmf.gather(1, safe).squeeze(1)
        return torch.where(in_range, lp, torch.full_like(lp, NEG_INF))


#: family -> (required parameter names, builder). ``builder`` receives a dict
#: of broadcast tensors plus ``n_states`` for discrete children.
FAMILIES = {
    "normal":      (("loc", "scale"),
                    lambda p, k: D.Normal(p["loc"], p["scale"], validate_args=False)),
    "lognormal":   (("loc", "scale"),
                    lambda p, k: D.LogNormal(p["loc"], p["scale"], validate_args=False)),
    "uniform":     (("low", "high"),
                    lambda p, k: D.Uniform(p["low"], p["high"], validate_args=False)),
    "loguniform":  (("low", "high"), lambda p, k: _loguniform(p)),
    "exponential": (("rate",),
                    lambda p, k: D.Exponential(p["rate"], validate_args=False)),
    "gamma":       (("concentration", "rate"),
                    lambda p, k: D.Gamma(p["concentration"], p["rate"], validate_args=False)),
    "beta":        (("concentration1", "concentration0"),
                    lambda p, k: D.Beta(p["concentration1"], p["concentration0"], validate_args=False)),
    "weibull":     (("scale", "concentration"),
                    lambda p, k: D.Weibull(p["scale"], p["concentration"], validate_args=False)),
    "student_t":   (("df", "loc", "scale"),
                    lambda p, k: D.StudentT(p["df"], p["loc"], p["scale"], validate_args=False)),
    "bernoulli":   (("probs",),
                    lambda p, k: D.Bernoulli(probs=p["probs"], validate_args=False)),
    "categorical": (("probs",), lambda p, k: D.Categorical(probs=p["probs"],
                                                           validate_args=False)),
    "truncated_poisson": (("rate",), lambda p, k: _TruncatedPoisson(p["rate"], k)),
}

#: Families whose samples are state indices rather than continuous values.
DISCRETE_FAMILIES = frozenset({"bernoulli", "categorical", "truncated_poisson"})


# --------------------------------------------------------------------------
# Generated node base
# --------------------------------------------------------------------------
class SpecNode(Probability):
    """Common plumbing for every compiled node."""

    def __init__(self, node: NodeSpec, childs, parents, device="cpu"):
        self.spec = node
        self.spec_metadata = dict(node.metadata)
        self.name = node.name
        self.childs = list(childs)
        self.parents = list(parents)
        self.device = device

        # Label lookup tables for discrete parents, built once.
        self._labels = {}
        for var in self.parents:
            if isinstance(var.values, tuple):
                continue
            if all(isinstance(s, (int, float)) and not isinstance(s, bool)
                   for s in var.values):
                self._labels[var.name] = torch.tensor(
                    [float(s) for s in var.values],
                    dtype=torch.float32, device=device)

    # -- environment -------------------------------------------------------
    def _env(self, columns: torch.Tensor):
        """Split a (N, n_parents) tensor into expression values and indices."""
        values, indices = {}, {}
        for j, var in enumerate(self.parents):
            col = columns[:, j]
            if isinstance(var.values, tuple):          # continuous
                values[var.name] = col
                continue
            idx = col.long()
            indices[var.name] = idx
            table = self._labels.get(var.name)
            if table is not None:
                values[var.name] = table[idx]
        return values, indices

    def _split(self, Cs: torch.Tensor):
        """Split a log_prob matrix into (child column, parent env)."""
        Cs = Cs.to(self.device)
        n_child = len(self.childs)
        expected = n_child + len(self.parents)
        if Cs.dim() != 2 or Cs.shape[1] != expected:
            raise CompileError(
                f"node {self.name!r}: log_prob expects a (N, {expected}) tensor "
                f"with columns {[c.name for c in self.childs] + [p.name for p in self.parents]}, "
                f"got shape {tuple(Cs.shape)}"
            )
        return Cs[:, 0], Cs[:, n_child:]

    def _parent_columns(self, n_sample, Cs_pars):
        if self.parents:
            if Cs_pars is None:
                raise CompileError(
                    f"node {self.name!r} has parents "
                    f"{[p.name for p in self.parents]}, so sample() needs Cs_pars"
                )
            return Cs_pars.to(self.device), Cs_pars.shape[0]
        if n_sample is None:
            raise CompileError(
                f"node {self.name!r} is a root, so sample() needs n_sample"
            )
        return torch.empty((n_sample, 0), device=self.device), int(n_sample)


# --------------------------------------------------------------------------
class DistributionSpecNode(SpecNode):
    """A parametric family with expression-valued parameters."""

    def __init__(self, node: DistributionNode, childs, parents, scope, device="cpu"):
        super().__init__(node, childs, parents, device)
        if node.family not in FAMILIES:
            raise CompileError(
                f"node {node.name!r}: unknown family {node.family!r}; "
                f"available: {sorted(FAMILIES)}"
            )
        required, self._build = FAMILIES[node.family]
        self.family = node.family

        missing = [r for r in required if r not in node.parameters]
        extra = sorted(set(node.parameters) - set(required))
        if missing or extra:
            raise CompileError(
                f"node {node.name!r}: family {node.family!r} takes parameters "
                f"{list(required)}; missing {missing}, unexpected {extra}"
            )

        child = self.childs[0]
        self._child_discrete = not isinstance(child.values, tuple)
        self._n_states = len(child.values) if self._child_discrete else None
        if (node.family in DISCRETE_FAMILIES) != self._child_discrete:
            kind = "discrete" if node.family in DISCRETE_FAMILIES else "continuous"
            raise CompileError(
                f"node {node.name!r}: family {node.family!r} produces a {kind} "
                f"child, but {child.name!r} is declared "
                f"{'discrete' if self._child_discrete else 'continuous'}"
            )

        label = f"node {node.name!r}"
        self._params = {}
        self._vector_params = {}
        for key, src in node.parameters.items():
            if isinstance(src, (list, tuple)):
                self._vector_params[key] = [
                    Expression(s, scope, label=f"{label} parameter {key}[{i}]")
                    for i, s in enumerate(src)]
            else:
                self._params[key] = Expression(
                    src, scope, label=f"{label} parameter {key}")

    def _dist(self, columns, n):
        values, indices = self._env(columns)
        params = {k: _broadcast(e(values, indices), n, self.device)
                  for k, e in self._params.items()}
        for k, exprs in self._vector_params.items():
            params[k] = torch.stack(
                [_broadcast(e(values, indices), n, self.device) for e in exprs],
                dim=-1)
        return self._build(params, self._n_states)

    def form(self):
        params = []
        for key, src in self.spec.parameters.items():
            text = f"[{', '.join(src)}]" if isinstance(src, (list, tuple)) else src
            params.append(f"{key}={text}")
        return f"{self.family}({', '.join(params)})"

    def sample(self, n_sample=None, Cs_pars=None):
        columns, n = self._parent_columns(n_sample, Cs_pars)
        dist = self._dist(columns, n)
        Cs = dist.sample()
        logp = dist.log_prob(Cs)
        if self._child_discrete:
            Cs = Cs.long()
        return Cs, logp

    def log_prob(self, Cs):
        child, columns = self._split(Cs)
        dist = self._dist(columns, child.shape[0])
        if self._child_discrete:
            return dist.log_prob(child.long())
        return dist.log_prob(child)


# --------------------------------------------------------------------------
class EquationSpecNode(SpecNode):
    """A deterministic function of the parents."""

    def __init__(self, node: EquationNode, childs, parents, scope, device="cpu"):
        super().__init__(node, childs, parents, device)
        self.expression = Expression(node.equation, scope,
                                     label=f"node {node.name!r}")
        self.tolerance = float(node.tolerance)
        child = self.childs[0]
        self._child_discrete = not isinstance(child.values, tuple)
        self._n_states = len(child.values) if self._child_discrete else None

    def _evaluate(self, columns, n):
        values, indices = self._env(columns)
        out = _broadcast(self.expression(values, indices), n, self.device)
        return out.round().long() if self._child_discrete else out

    def form(self):
        tolerance = f", within {self.tolerance:g}" if self.tolerance else ""
        return f"{self.childs[0].name} = {self.expression.source}{tolerance}"

    def sample(self, n_sample=None, Cs_pars=None):
        columns, n = self._parent_columns(n_sample, Cs_pars)
        Cs = self._evaluate(columns, n)
        return Cs, torch.zeros(n, device=self.device)

    def log_prob(self, Cs):
        child, columns = self._split(Cs)
        expected = self._evaluate(columns, child.shape[0])
        if self._child_discrete:
            ok = child.long() == expected
        elif self.tolerance > 0:
            ok = (child - expected).abs() <= self.tolerance
        else:
            ok = child == expected
        zero = torch.zeros(child.shape[0], device=self.device)
        return torch.where(ok, zero, torch.full_like(zero, NEG_INF))


# --------------------------------------------------------------------------
class DiscretiseSpecNode(SpecNode):
    """Bins one continuous parent into the child's discrete states."""

    def __init__(self, node: DiscretiseNode, childs, parents, device="cpu"):
        super().__init__(node, childs, parents, device)
        self.intervals = [(float(a), float(b)) for a, b in node.intervals]
        # Interior edges; bucketize with right=False gives [low, high) bins,
        # and the final bin absorbs everything at or above the last edge.
        self._edges = torch.tensor([b for _, b in self.intervals[:-1]],
                                   dtype=torch.float32, device=device)
        self._low = self.intervals[0][0]
        self._high = self.intervals[-1][1]

    def form(self):
        bins = [f"[{a:g}, {b:g})" for a, b in self.intervals]
        bins[-1] = bins[-1][:-1] + "]"
        return f"bins of {self.parents[0].name}: {', '.join(bins)}"

    def _bin(self, parent_values):
        idx = torch.bucketize(parent_values.contiguous(), self._edges, right=False)
        outside = (parent_values < self._low) | (parent_values > self._high)
        return idx, outside

    def sample(self, n_sample=None, Cs_pars=None):
        columns, n = self._parent_columns(n_sample, Cs_pars)
        idx, _ = self._bin(columns[:, 0])
        return idx.long(), torch.zeros(n, device=self.device)

    def log_prob(self, Cs):
        child, columns = self._split(Cs)
        idx, outside = self._bin(columns[:, 0])
        ok = (child.long() == idx) & ~outside
        zero = torch.zeros(child.shape[0], device=self.device)
        return torch.where(ok, zero, torch.full_like(zero, NEG_INF))


# --------------------------------------------------------------------------
def _build_cpt(node: DiscreteCptNode, childs, parents, device):
    """Expand a CPT table into the (C, p) arrays :class:`tbnpy.cpt.Cpt` wants."""
    n_child_states = len(childs[0].values)
    cardinalities = [len(p.values) for p in parents]

    combos = (list(np.ndindex(*cardinalities)) if cardinalities else [()])
    rows, probs = [], []
    for row_idx, combo in enumerate(combos):
        for state in range(n_child_states):
            rows.append([state, *combo])
            probs.append(float(node.probabilities[row_idx][state]))

    obj = cpt_mod.Cpt(childs=list(childs), parents=list(parents),
                      C=np.array(rows, dtype=np.int64),
                      p=np.array(probs, dtype=np.float32),
                      device=device)
    obj.spec = node
    obj.spec_metadata = dict(node.metadata)
    obj.name = node.name
    return obj


# --------------------------------------------------------------------------
def _build_custom(node: CustomNode, childs, parents, device, search_path):
    added = []
    for entry in search_path or []:
        entry = str(Path(entry).resolve())
        if entry not in sys.path:
            sys.path.insert(0, entry)
            added.append(entry)
    try:
        try:
            module = importlib.import_module(node.module)
        except ImportError as exc:
            raise CompileError(
                f"node {node.name!r}: cannot import module {node.module!r}. "
                f"Pass search_path=[...] with the directory containing it. "
                f"({exc})"
            ) from exc
        try:
            klass = getattr(module, node.cls)
        except AttributeError as exc:
            raise CompileError(
                f"node {node.name!r}: module {node.module!r} has no "
                f"{node.cls!r}"
            ) from exc

        kwargs = dict(node.kwargs)
        try:
            params = inspect.signature(klass).parameters
            accepts_device = "device" in params or any(
                p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            accepts_device = True
        if accepts_device and "device" not in kwargs:
            kwargs["device"] = device

        try:
            obj = klass(childs=list(childs), parents=list(parents), **kwargs)
        except TypeError as exc:
            raise CompileError(
                f"node {node.name!r}: could not construct "
                f"{node.module}.{node.cls} -- {exc}"
            ) from exc
    finally:
        for entry in added:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass

    for attr in ("childs", "parents", "sample", "log_prob"):
        if not hasattr(obj, attr):
            raise CompileError(
                f"node {node.name!r}: {node.module}.{node.cls} does not expose "
                f"{attr!r}, so it is not a tbnpy node"
            )
    obj.spec = node
    obj.spec_metadata = dict(node.metadata)
    if not hasattr(obj, "name"):
        obj.name = node.name
    return obj


# --------------------------------------------------------------------------
def compile_model(spec: ModelSpec, device: str = "cpu",
                  search_path: "Sequence | None" = None,
                  validate: bool = True):
    """Turn a spec into ``(varis, probs)``.

    Parameters
    ----------
    spec:
        The model description.
    device:
        ``"cpu"`` or ``"cuda"``, passed to every node.
    search_path:
        Directories to make importable while constructing ``custom`` nodes,
        typically the folder holding the model file.
    validate:
        Re-check the spec before compiling. Leave on unless it was just
        validated.

    Returns
    -------
    (dict, dict)
        ``{name: Variable}`` and ``{node_name: node}``, ready for
        ``tbnpy.inference`` or for a bespoke algorithm.
    """
    if validate:
        spec.validate()

    variables = spec.variable_map
    varis = build_variables(spec)
    probs = {}

    for node in spec.nodes:
        childs = [varis[c] for c in node.child_names]
        parents = [varis[p] for p in node.parents]

        if isinstance(node, DiscreteCptNode):
            probs[node.name] = _build_cpt(node, childs, parents, device)
            continue
        if isinstance(node, CustomNode):
            probs[node.name] = _build_custom(node, childs, parents, device,
                                             search_path)
            continue

        scope = _scope_for(node, variables)
        if isinstance(node, DistributionNode):
            probs[node.name] = DistributionSpecNode(node, childs, parents,
                                                    scope, device)
        elif isinstance(node, EquationNode):
            probs[node.name] = EquationSpecNode(node, childs, parents,
                                                scope, device)
        elif isinstance(node, DiscretiseNode):
            probs[node.name] = DiscretiseSpecNode(node, childs, parents, device)
        else:
            raise CompileError(
                f"node {node.name!r}: no compiler for type {type(node).__name__}"
            )

    return varis, probs


def compile_file(path, device: str = "cpu", search_path=None):
    """Load a spec file and compile it, searching next to the file by default."""
    path = Path(path)
    spec = load_spec(path)
    if search_path is None:
        search_path = [path.parent]
    return spec, *compile_model(spec, device=device, search_path=search_path)
