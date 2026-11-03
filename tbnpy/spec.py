"""
Declarative model specification for tbnpy.

A *spec* is a plain data description of a Bayesian network -- its variables,
its graph, and how each node is quantified -- that can be written by hand,
generated from a natural-language description, exported from GeNIe, or edited
in a GUI. :mod:`tbnpy.compile` turns a spec into the ``varis`` and ``probs``
objects the rest of the library works with.

The spec deliberately describes only the *model*. It carries nothing about how
it will be queried or how inference should be run: a Bayesian network answers
any query within its scope, and tbnpy exists precisely for problems that need
bespoke inference algorithms, so a model description that presumed particular
queries or one algorithm would be a description of the wrong thing.

Node types
----------
``distribution``
    A parametric family whose parameters are expressions over the parents.
``equation``
    A deterministic function of the parents.
``discrete_cpt``
    An explicit conditional probability table; compiles to :class:`tbnpy.cpt.Cpt`.
``discretise``
    Bins a continuous parent into the child's discrete states.
``custom``
    A hand-written node class, imported by module and name. Bespoke nodes stay
    first-class citizens: anything the declarative types cannot express is
    written in Python and referenced from here. A custom node may own several
    child variables (``childs``), such as a field drawn jointly over many sites;
    every other type owns one (``child``).

Parameter sets
--------------
Constructor arguments that many custom nodes share, such as one fragility curve
for every bridge of a class, are written once in the spec's ``parameter_sets``
and named by each node's ``parameter_set``. The node's class receives them with
the node's own ``kwargs``.

Serialisation is JSON (stdlib only). YAML is accepted too when ``pyyaml``
happens to be installed, but it is never required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SpecError", "VariableSpec", "NodeSpec", "DistributionNode", "EquationNode",
    "DiscreteCptNode", "DiscretiseNode", "CustomNode",
    "ModelSpec", "load_spec", "save_spec",
]

#: 2 adds ``childs`` to custom nodes; 3 adds ``parameter_sets``. Earlier files read unchanged.
SPEC_VERSION = 3

CONTINUOUS = "continuous"
DISCRETE = "discrete"


class SpecError(ValueError):
    """Raised when a specification is malformed or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SpecError(message)


# --------------------------------------------------------------------------
# Variables
# --------------------------------------------------------------------------
@dataclass
class VariableSpec:
    """One random variable.

    Attributes
    ----------
    kind:
        ``"continuous"`` or ``"discrete"``.
    states:
        Discrete only: the state labels, worst first by tbnpy convention.
        State *indices* are ``0 .. len(states) - 1``; the labels are what
        expressions see when the variable is used as a value.
    support:
        Continuous only: ``(low, high)`` for the variable's actual support.
        Defaults to the whole real line. Either side may be ``None`` (JSON
        ``null``) for an unbounded side, e.g. ``[0, null]`` for a positive
        quantity, since JSON cannot write infinity.
    bounds:
        Optional plausibility range, used by diagnostics to flag samples that
        leave it. Not a truncation -- GeNIe bounds are frequently only a
        display range, so the two concepts are kept apart deliberately.
    """

    name: str = ""
    kind: str = CONTINUOUS
    states: "list | None" = None
    support: "list | None" = None
    bounds: "list | None" = None
    unit: str = ""
    meaning: str = ""

    def __post_init__(self) -> None:
        _require(bool(self.name), "a variable needs a name")
        _require(self.kind in (CONTINUOUS, DISCRETE),
                 f"variable {self.name!r}: kind must be "
                 f"{CONTINUOUS!r} or {DISCRETE!r}, got {self.kind!r}")
        if self.kind == DISCRETE:
            _require(bool(self.states),
                     f"discrete variable {self.name!r} needs a non-empty "
                     f"'states' list")
            _require(len(self.states) == len(set(map(str, self.states))),
                     f"discrete variable {self.name!r} has duplicate state labels")
            _require(self.support is None,
                     f"variable {self.name!r}: 'support' applies to continuous "
                     f"variables; discrete variables use 'states'")
        else:
            _require(not self.states,
                     f"continuous variable {self.name!r} must not declare 'states'")
        for attr in ("support", "bounds"):
            rng = getattr(self, attr)
            if rng is not None:
                _require(len(rng) == 2 and (None in rng or rng[0] <= rng[1]),
                         f"variable {self.name!r}: {attr} must be "
                         f"[low, high] with low <= high (null for an unbounded "
                         f"side), got {rng!r}")

    # -- derived -----------------------------------------------------------
    @property
    def is_discrete(self) -> bool:
        return self.kind == DISCRETE

    @property
    def n_states(self) -> "int | None":
        return len(self.states) if self.is_discrete else None

    @property
    def has_numeric_states(self) -> bool:
        """Whether the state labels can be used as values in an expression."""
        if not self.is_discrete:
            return False
        return all(isinstance(s, (int, float)) and not isinstance(s, bool)
                   for s in self.states)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
@dataclass
class NodeSpec:
    """Base for every node type. ``name`` is the key in the ``probs`` dict."""

    name: str = ""
    child: str = ""
    parents: "list" = field(default_factory=list)
    meaning: str = ""
    references: "list" = field(default_factory=list)
    #: Free-form, carried through to the compiled object as ``.spec_metadata``.
    #: Bespoke inference algorithms can stash whatever they need here without
    #: the spec format having to know about it.
    metadata: dict = field(default_factory=dict)

    type: str = ""          # set by subclasses

    def __post_init__(self) -> None:
        _require(bool(self.name), "a node needs a name")
        if not self.child and not getattr(self, "childs", None):
            self.child = self.name
        _require(len(self.parents) == len(set(self.parents)),
                 f"node {self.name!r} lists a parent more than once")
        for child in self.child_names:
            _require(child not in self.parents,
                     f"node {self.name!r} lists its own child {child!r} "
                     f"as a parent")

    @property
    def child_names(self) -> "list":
        """The variables this node owns: ``[child]``, or a custom node's ``childs``."""
        return [self.child]

    def validate_against(self, variables: "dict") -> None:
        """Type-specific checks that need the variable table. Overridden below."""


@dataclass
class DistributionNode(NodeSpec):
    """A parametric family; each parameter is an expression over the parents."""

    family: str = ""
    parameters: dict = field(default_factory=dict)
    type: str = "distribution"

    def __post_init__(self) -> None:
        super().__post_init__()
        _require(bool(self.family),
                 f"node {self.name!r}: 'family' is required for a "
                 f"distribution node")
        for key, val in self.parameters.items():
            if isinstance(val, (list, tuple)):
                # Vector parameters (e.g. categorical probs) are one
                # expression per child state.
                _require(bool(val) and all(isinstance(s, str) for s in val),
                         f"node {self.name!r}: parameter {key!r} is a list, so "
                         f"every entry must be an expression string")
            else:
                _require(isinstance(val, str),
                         f"node {self.name!r}: parameter {key!r} must be an "
                         f"expression string (write 0.6 as \"0.6\")")


@dataclass
class EquationNode(NodeSpec):
    """A deterministic function of the parents.

    ``tolerance`` widens the equality that ``log_prob`` accepts. It defaults to
    zero, matching exact deterministic nodes. Note that a zero tolerance makes
    the node impossible to condition on directly when the child is continuous:
    every particle gets ``-inf``. Conditioning on such a node needs a tolerance,
    an explicit observation node, or discretisation.
    """

    equation: str = ""
    tolerance: float = 0.0
    type: str = "equation"

    def __post_init__(self) -> None:
        super().__post_init__()
        _require(bool(self.equation),
                 f"node {self.name!r}: 'equation' is required for an "
                 f"equation node")
        _require(self.tolerance >= 0,
                 f"node {self.name!r}: tolerance must be >= 0")


@dataclass
class DiscreteCptNode(NodeSpec):
    """An explicit CPT over discrete child and discrete parents.

    ``probabilities`` has one row per parent-state combination and one column
    per child state. Rows are ordered with the **last** parent varying fastest
    (C order over the parents), which matches GeNIe's flat CPT layout. A root
    node has a single row.
    """

    probabilities: "list" = field(default_factory=list)
    type: str = "discrete_cpt"

    def __post_init__(self) -> None:
        super().__post_init__()
        _require(bool(self.probabilities),
                 f"node {self.name!r}: 'probabilities' is required for a "
                 f"discrete_cpt node")
        # Accept a bare vector for a root and normalise it to one row.
        if self.probabilities and not isinstance(self.probabilities[0], (list, tuple)):
            self.probabilities = [list(self.probabilities)]
        else:
            self.probabilities = [list(r) for r in self.probabilities]
        widths = {len(r) for r in self.probabilities}
        _require(len(widths) == 1,
                 f"node {self.name!r}: every CPT row must have the same length, "
                 f"got lengths {sorted(widths)}")

    def validate_against(self, variables: "dict") -> None:
        child = variables[self.child]
        _require(child.is_discrete,
                 f"node {self.name!r}: discrete_cpt needs a discrete child, "
                 f"but {self.child!r} is continuous")
        n_rows_expected = 1
        for p in self.parents:
            pv = variables[p]
            _require(pv.is_discrete,
                     f"node {self.name!r}: discrete_cpt needs discrete parents, "
                     f"but {p!r} is continuous -- discretise it first")
            n_rows_expected *= pv.n_states
        _require(len(self.probabilities) == n_rows_expected,
                 f"node {self.name!r}: expected {n_rows_expected} CPT row(s) "
                 f"for parents {self.parents}, got {len(self.probabilities)}")
        _require(len(self.probabilities[0]) == child.n_states,
                 f"node {self.name!r}: CPT rows have "
                 f"{len(self.probabilities[0])} entries but {self.child!r} has "
                 f"{child.n_states} states")
        for i, row in enumerate(self.probabilities):
            total = float(sum(row))
            _require(all(p >= 0 for p in row),
                     f"node {self.name!r}: CPT row {i} has a negative probability")
            _require(abs(total - 1.0) < 1e-6,
                     f"node {self.name!r}: CPT row {i} sums to {total:.6g}, not 1")


@dataclass
class DiscretiseNode(NodeSpec):
    """Bins one continuous parent into the child's discrete states.

    ``intervals`` is a list of ``[low, high]`` pairs, one per child state, in
    state order. Bins are half-open ``[low, high)`` except the last, which
    includes its upper edge.
    """

    intervals: "list" = field(default_factory=list)
    type: str = "discretise"

    def __post_init__(self) -> None:
        super().__post_init__()
        _require(len(self.parents) == 1,
                 f"node {self.name!r}: discretise takes exactly one parent, "
                 f"got {self.parents}")
        _require(bool(self.intervals),
                 f"node {self.name!r}: 'intervals' is required")
        for iv in self.intervals:
            _require(len(iv) == 2 and iv[0] < iv[1],
                     f"node {self.name!r}: each interval must be [low, high] "
                     f"with low < high, got {iv!r}")
        for a, b in zip(self.intervals, self.intervals[1:]):
            _require(a[1] <= b[0],
                     f"node {self.name!r}: intervals must be ordered and "
                     f"non-overlapping; {a!r} overlaps {b!r}")

    def validate_against(self, variables: "dict") -> None:
        child = variables[self.child]
        _require(child.is_discrete,
                 f"node {self.name!r}: discretise needs a discrete child")
        _require(len(self.intervals) == child.n_states,
                 f"node {self.name!r}: {len(self.intervals)} interval(s) for "
                 f"{child.n_states} state(s) of {self.child!r}")
        _require(not variables[self.parents[0]].is_discrete,
                 f"node {self.name!r}: discretise needs a continuous parent, "
                 f"but {self.parents[0]!r} is discrete")


@dataclass
class CustomNode(NodeSpec):
    """A hand-written node class, imported and constructed by the compiler.

    The class must follow the tbnpy node protocol: ``childs``, ``parents``,
    ``sample``, ``log_prob``. It is given ``childs`` and ``parents`` (and
    ``device`` when its signature accepts one) plus everything in ``kwargs``.

    ``kwargs`` are JSON, so data too large or too binary to write inline (rule
    sets, tables, grids, weights) is named by file, relative to the spec's
    folder. A class with a ``directory`` parameter receives that folder.

    A node owning several variables, such as a field drawn jointly over many
    sites, lists them in ``childs`` instead of naming one ``child``; they reach
    the class in that order.

    ``parameter_set`` names an entry of the spec's ``parameter_sets``: arguments
    shared by many nodes, written once. The class receives that entry together
    with ``kwargs``, which hold what is particular to this node; an argument
    may come from one or the other, not both.
    """

    module: str = ""
    cls: str = ""
    kwargs: dict = field(default_factory=dict)
    childs: "list" = field(default_factory=list)
    parameter_set: str = ""
    type: str = "custom"

    def __post_init__(self) -> None:
        if self.childs:
            _require(not self.child,
                     f"node {self.name!r}: give 'child' or 'childs', not both")
            _require(len(self.childs) == len(set(self.childs)),
                     f"node {self.name!r} lists a child more than once")
        super().__post_init__()
        _require(bool(self.module) and bool(self.cls),
                 f"node {self.name!r}: 'module' and 'cls' are required for a "
                 f"custom node")
        _require(isinstance(self.parameter_set, str),
                 f"node {self.name!r}: 'parameter_set' is the name of an entry "
                 f"of the spec's parameter_sets")

    @property
    def child_names(self) -> "list":
        return list(self.childs) if self.childs else [self.child]


_NODE_TYPES = {
    "distribution": DistributionNode,
    "equation": EquationNode,
    "discrete_cpt": DiscreteCptNode,
    "discretise": DiscretiseNode,
    "custom": CustomNode,
}


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
@dataclass
class ModelSpec:
    """A whole model: its variables, its nodes, and the parameter sets nodes share.

    ``parameter_sets`` maps a name to constructor arguments, ``{argument: value}``,
    that every custom node naming it in ``parameter_set`` receives. Shared
    parameters are then one entry, not a copy per node, and the file says which
    nodes share them.
    """

    name: str = ""
    description: str = ""
    variables: "list" = field(default_factory=list)
    nodes: "list" = field(default_factory=list)
    open_questions: "list" = field(default_factory=list)
    source: str = ""
    spec_version: int = SPEC_VERSION
    parameter_sets: dict = field(default_factory=dict)

    # -- lookups -----------------------------------------------------------
    @property
    def variable_map(self) -> dict:
        return {v.name: v for v in self.variables}

    @property
    def node_map(self) -> dict:
        return {n.name: n for n in self.nodes}

    def owner_of(self, variable_name: str) -> "str | None":
        for n in self.nodes:
            if variable_name in n.child_names:
                return n.name
        return None

    def arguments(self, node: "CustomNode") -> dict:
        """A custom node's constructor arguments: its parameter set's, then its own ``kwargs``."""
        shared = self.parameter_sets[node.parameter_set] if node.parameter_set else {}
        return {**shared, **node.kwargs}

    # -- validation --------------------------------------------------------
    def validate(self) -> "ModelSpec":
        """Check the spec is internally consistent. Returns ``self``."""
        _require(bool(self.variables), "a model needs at least one variable")
        _require(bool(self.nodes), "a model needs at least one node")

        variables = self.variable_map
        _require(len(variables) == len(self.variables),
                 "two variables share a name")
        nodes = self.node_map
        _require(len(nodes) == len(self.nodes), "two nodes share a name")

        # Every variable is the child of exactly one node.
        owners: dict = {}
        for n in self.nodes:
            for child in n.child_names:
                _require(child in variables,
                         f"node {n.name!r} has child {child!r}, which is not a "
                         f"declared variable")
                _require(child not in owners,
                         f"variable {child!r} is the child of both "
                         f"{owners.get(child)!r} and {n.name!r}")
                owners[child] = n.name
        orphans = sorted(set(variables) - set(owners))
        _require(not orphans,
                 f"these variables are not produced by any node: {orphans}")

        # Parents resolve.
        for n in self.nodes:
            for p in n.parents:
                _require(p in variables,
                         f"node {n.name!r} has parent {p!r}, which is not a "
                         f"declared variable")

        self._check_acyclic(owners)

        for set_name, arguments in self.parameter_sets.items():
            _require(isinstance(arguments, Mapping),
                     f"parameter set {set_name!r} must be an object of "
                     f"constructor arguments, got {type(arguments).__name__}")
        for n in self.nodes:
            if not getattr(n, "parameter_set", ""):
                continue
            _require(n.parameter_set in self.parameter_sets,
                     f"node {n.name!r} names parameter set {n.parameter_set!r}, "
                     f"which is not in the spec's parameter_sets")
            both = sorted(set(self.parameter_sets[n.parameter_set]) & set(n.kwargs))
            _require(not both,
                     f"node {n.name!r} gives {both} in its kwargs and in its "
                     f"parameter set {n.parameter_set!r}; give each argument once")

        for n in self.nodes:
            n.validate_against(variables)

        return self

    def _check_acyclic(self, owners: dict) -> None:
        """Kahn's algorithm over nodes, reporting the cycle if there is one."""
        incoming = {n.name: set() for n in self.nodes}
        outgoing: dict = {n.name: set() for n in self.nodes}
        for n in self.nodes:
            for p in n.parents:
                owner = owners[p]
                if owner == n.name:
                    continue
                incoming[n.name].add(owner)
                outgoing[owner].add(n.name)

        ready = [name for name, deps in incoming.items() if not deps]
        seen = 0
        pending = {k: set(v) for k, v in incoming.items()}
        while ready:
            cur = ready.pop()
            seen += 1
            for nxt in outgoing[cur]:
                pending[nxt].discard(cur)
                if not pending[nxt]:
                    ready.append(nxt)
        if seen != len(self.nodes):
            stuck = sorted(k for k, v in pending.items() if v)
            raise SpecError(
                f"the graph has a cycle; these nodes are involved: {stuck}"
            )

    def topological_order(self) -> "list":
        """Node names ordered so parents precede children."""
        self.validate()
        owners = {child: n.name for n in self.nodes for child in n.child_names}
        pending = {n.name: {owners[p] for p in n.parents if owners[p] != n.name}
                   for n in self.nodes}
        order: list = []
        ready = sorted(k for k, v in pending.items() if not v)
        while ready:
            cur = ready.pop(0)
            order.append(cur)
            newly = []
            for name, deps in pending.items():
                if cur in deps:
                    deps.discard(cur)
                    if not deps and name not in order and name not in ready:
                        newly.append(name)
            ready.extend(sorted(newly))
        return order

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        out = asdict(self)
        # Drop empties so hand-written files stay readable on round-trip.
        for node in out["nodes"]:
            for key in ("meaning", "references", "metadata", "childs", "child", "parameter_set"):
                if not node.get(key):
                    node.pop(key, None)
        for var in out["variables"]:
            for key in ("states", "support", "bounds", "unit", "meaning"):
                if not var.get(key):
                    var.pop(key, None)
        for key in ("description", "open_questions", "source", "parameter_sets"):
            if not out.get(key):
                out.pop(key, None)
        # Shared parameters read before the nodes that name them.
        order = ["name", "description", "spec_version", "variables", "parameter_sets", "nodes"]
        return {**{k: out[k] for k in order if k in out}, **out}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ModelSpec":
        _require(isinstance(data, Mapping), "a spec must be a JSON object")
        version = data.get("spec_version", SPEC_VERSION)
        _require(version <= SPEC_VERSION,
                 f"this spec declares version {version}, but this tbnpy "
                 f"understands up to version {SPEC_VERSION}")

        variables = [VariableSpec(**v) for v in data.get("variables", [])]

        nodes = []
        for raw in data.get("nodes", []):
            raw = dict(raw)
            ntype = raw.pop("type", None)
            _require(ntype in _NODE_TYPES,
                     f"node {raw.get('name')!r} has type {ntype!r}; expected one "
                     f"of {sorted(_NODE_TYPES)}")
            klass = _NODE_TYPES[ntype]
            allowed = {f for f in klass.__dataclass_fields__ if f != "type"}
            unknown = sorted(set(raw) - allowed)
            _require(not unknown,
                     f"node {raw.get('name')!r} ({ntype}) has unexpected "
                     f"field(s) {unknown}; allowed: {sorted(allowed)}")
            nodes.append(klass(**raw))

        # Files from earlier versions may carry an "analysis" section of intended
        # queries. A model does not fix its queries, so it is read past, not kept.
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            variables=variables,
            nodes=nodes,
            open_questions=list(data.get("open_questions", [])),
            source=data.get("source", ""),
            spec_version=version,
            parameter_sets={k: dict(v) if isinstance(v, Mapping) else v
                            for k, v in data.get("parameter_sets", {}).items()},
        )


# --------------------------------------------------------------------------
def load_spec(path, validate: bool = True) -> ModelSpec:
    """Load a spec from ``.json`` (always) or ``.yaml`` (if pyyaml is present)."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import]
        except ImportError as exc:
            raise SpecError(
                f"{path.name} is YAML, which needs the optional 'pyyaml' "
                f"package. Install it, or convert the file to JSON."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    spec = ModelSpec.from_dict(data)
    return spec.validate() if validate else spec


def save_spec(spec: ModelSpec, path) -> Path:
    """Write a spec as formatted JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path
