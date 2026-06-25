"""Structural contract shared by every node/probability object in tbnpy.

The inference engine (:mod:`tbnpy.inference`) is deliberately duck-typed: it
never checks for a concrete :class:`~tbnpy.cpt.Cpt`. Instead, any object that
satisfies :class:`CptLike` can act as a node in the Bayesian network. This is
what lets a native tensor-CPT and a custom, externally-backed node (e.g. an
RSR-driven system-state classifier) coexist behind a single interface.

This module pins that previously-implicit contract down in one place so it can
be documented, type-checked, and validated at runtime.

Log-space convention
--------------------
**All probabilities exchanged with the engine are log-probabilities.**

* ``sample(...)`` returns ``(Cs, log_ps)`` — the second element is the natural
  log of the sampling probability of each drawn event.
* ``log_prob(Cs)`` returns log-probabilities directly.

By convention a deterministic outcome has log-probability ``0.0`` (prob 1) and
an impossible outcome has ``-inf`` (prob 0). Keeping everything additive in
log-space is what allows native CPTs and custom nodes to be combined uniformly.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CptLike(Protocol):
    """Minimal interface an object must expose to be used as a BN node.

    Attributes:
        childs: list of child :class:`~tbnpy.variable.Variable` objects. Every
            variable must be a child of exactly one node.
        parents: list of parent :class:`~tbnpy.variable.Variable` objects.

    Methods:
        sample: draw events from the node.
            * No-parent nodes are called as ``sample(n_sample=...)``.
            * Nodes with parents are called as ``sample(Cs_pars=...)`` where
              ``Cs_pars`` holds composite parent states.
            Returns ``(Cs, log_ps)`` with ``log_ps`` in log-space (see module
            docstring).
        log_prob: return the log-probability of each composite event row in
            ``Cs``. Used by evidence-aware inference.
    """

    # Data attributes (presence is what ``runtime_checkable`` verifies).
    childs: list
    parents: list

    def sample(self, *args, **kwargs):  # pragma: no cover - structural only
        ...

    def log_prob(self, Cs):  # pragma: no cover - structural only
        ...


def validate_prob_objects(probs: dict) -> None:
    """Assert every value in ``probs`` satisfies :class:`CptLike`.

    Raises a ``TypeError`` naming the offending node and the missing members,
    turning a previously-silent contract violation (e.g. a node that forgot
    ``log_prob`` or returns raw instead of log probabilities) into a loud,
    actionable error at the entry point of inference.

    Note:
        ``runtime_checkable`` Protocols verify *member presence*, not call
        signatures or the log-space convention. This catches the common
        "wrong shape of object" mistakes; the log-space convention itself
        still relies on the documented contract.
    """
    required = ("childs", "parents", "sample", "log_prob")
    for name, obj in probs.items():
        missing = [m for m in required if not hasattr(obj, m)]
        if missing:
            raise TypeError(
                f"Probability object for node '{name}' ({type(obj).__name__}) "
                f"does not satisfy the CptLike contract; missing: "
                f"{', '.join(missing)}. See tbnpy.protocols.CptLike."
            )
