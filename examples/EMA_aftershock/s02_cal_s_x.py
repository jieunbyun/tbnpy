"""
Estimate ``P(S_t | X_{0,n} = 0)`` for every edge ``n`` and every time
slice ``t = 0, 1, ..., K_max``.

Approach
--------
Since ``X_{0,n} = 1`` iff ``D_{0,n} < 0.4``, conditioning on
``X_{0,n} = 0`` is equivalent to placing ``D_{0,n}`` strictly above the
threshold. We do so by overwriting the CPT of ``D_{0,n}`` with a
degenerate "delta-at-one" distribution: ``f_{D_{0,n}}(D = 1) = 1.0``
and zero elsewhere. The rest of the model is left untouched. Forward
sampling under the modified model then yields draws from
``P(. | X_{0,n} = 0)``.

The script:

1. Loads the full BN once via ``s1_define_model``.
2. For each edge ``n``, swaps in ``DeltaD`` for ``D_{0,n}`` and runs
   forward sampling.
3. Extracts ``S_t`` for every ``t`` and accumulates empirical
   probabilities ``P(S_t = s | X_{0,n} = 0)``.
4. Saves the results as a long-form CSV in ``results/``.
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def _binomial_cov(p, n):
    """Coefficient of variation of an empirical proportion ``p`` over
    ``n`` independent draws.

    ``CoV = sqrt(Var / E) = sqrt((1 - p) / (n * p))``.

    Returns ``inf`` when ``p == 0`` (the estimator is dominated by
    sampling noise) and ``0`` when ``p == 1``.
    """
    if p <= 0:
        return float("inf")
    return math.sqrt(max(0.0, 1.0 - p) / (n * p))

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

REPO_ROOT = BASE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RSR_REPO = REPO_ROOT.parent / "rsr"
if RSR_REPO.exists() and str(RSR_REPO) not in sys.path:
    sys.path.insert(0, str(RSR_REPO))

from tbnpy import inference  # noqa: E402

from s01_define_model import (  # noqa: E402
    define_probs,
    define_variables,
    load_rsr_refs,
    load_topology,
    region_from_nodes,
)


RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


class DeltaD:
    """Degenerate distribution ``f(D = value) = 1``.

    Parents are kept (with the same Variable objects as the original
    ``D_{0,n}`` node) so that the ancestor-ordering machinery in
    ``inference.sample_evidence`` continues to work — they're just
    ignored during sampling and log-prob evaluation.
    """

    def __init__(self, childs, parents, value=1.0, device="cpu"):
        self.childs = childs
        self.parents = parents
        self.device = device
        self.value = float(value)

    def sample(self, Cs_pars):
        n = Cs_pars.shape[0]
        Cs = torch.full((n,), self.value, device=self.device)
        logp = torch.zeros(n, device=self.device)
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        d = Cs[:, 0]
        return torch.where(
            d == self.value,
            torch.zeros_like(d),
            torch.full_like(d, -float("inf")),
        )


def condition_on_X0_zero(probs, varis, edge_id, device="cpu"):
    """Return a shallow copy of ``probs`` with ``D_{0,edge_id}`` pinned to 1.0.

    Keeping the same parents on the surrogate node preserves the BN
    graph structure so the topological sort still terminates.
    """
    probs_mod = dict(probs)
    orig = probs[f"D_0_{edge_id}"]
    probs_mod[f"D_0_{edge_id}"] = DeltaD(
        childs=[varis[f"D_0_{edge_id}"]],
        parents=orig.parents,
        value=1.0,
        device=device,
    )
    return probs_mod


def estimate_S_marginals(probs, varis, K_max, n_sample, max_st=2,
                         batch_size=None, verbose=True):
    """Forward-sample and return ``P(S_t = s)`` for every ``t``.

    Valid system states are ``{0, 1, ..., max_st}``. Samples classified
    as ``-1`` (the ``S`` class's default ``unknown_state``) are tallied
    across batches and reported once at the end if any appear.

    The work is split into chunks of ``batch_size`` draws (default: a
    single batch of ``n_sample``). Counts are accumulated across batches
    and only normalised at the end, so the result is exact — batching
    affects peak memory but not statistical accuracy. Each batch
    triggers a fresh ``inference.sample_evidence`` call, so the per-node
    ``(1, batch_size, ...)`` allocations are bounded by ``batch_size``
    rather than by the total sample budget.

    When ``verbose`` is True (default) and there is more than one batch,
    each batch prints its wall-clock time, cumulative elapsed time and
    a linear ETA. A single-batch call stays quiet.

    Returns
    -------
    np.ndarray of shape (K_max + 1, max_st + 1)
        Row ``t`` is the empirical probability vector for ``S_t`` over
        the valid states ``0..max_st``.
    """
    if batch_size is None or batch_size >= n_sample:
        batches = [n_sample]
    else:
        batches = []
        remaining = n_sample
        while remaining > 0:
            batches.append(min(batch_size, remaining))
            remaining -= batches[-1]

    n_states = max_st + 1
    counts = np.zeros((K_max + 1, n_states), dtype=np.int64)
    totals = np.zeros(K_max + 1, dtype=np.int64)
    unknowns = np.zeros(K_max + 1, dtype=np.int64)

    query_nodes = [f"S_{t}" for t in range(K_max + 1)]
    evidence_df = pd.DataFrame(index=[0])  # one dummy row, no columns

    show_progress = verbose and len(batches) > 1
    t_start = time.perf_counter()

    for bi, b in enumerate(batches, start=1):
        t_batch = time.perf_counter()
        filled = inference.sample_evidence(
            probs=probs,
            query_nodes=query_nodes,
            n_sample=b,
            evidence_df=evidence_df,
        )
        for t in range(K_max + 1):
            Cs = filled[f"S_{t}"].Cs  # (1, b, 1 + n_parents) — child first
            s_samples = Cs[0, :, 0].detach().cpu().numpy().astype(int)
            unknowns[t] += int((s_samples < 0).sum())
            valid = s_samples[s_samples >= 0]
            counts[t] += np.bincount(valid, minlength=n_states)
            totals[t] += len(valid)

        if show_progress:
            elapsed = time.perf_counter() - t_start
            dt = time.perf_counter() - t_batch
            eta = elapsed / bi * (len(batches) - bi)
            print(f"    batch {bi}/{len(batches)} ({b} samples) — "
                  f"{dt:.2f}s, elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

    if show_progress:
        total = time.perf_counter() - t_start
        print(f"    done: {total:.1f}s for {n_sample} samples "
              f"across {len(batches)} batches")

    for t in range(K_max + 1):
        if unknowns[t] > 0:
            print(f"  WARN  S_{t}: {unknowns[t]}/{n_sample} samples are "
                  f"unknown (-1); consider passing s_fun to s_mod.S.")

    out = counts.astype(np.float64) / np.maximum(totals[:, None], 1)
    return out


def main(K_max=2, n_sample=2000, edge_subset=None, max_st=2,
         device=None, force_recompute=False, batch_size=None):
    """Sweep edges, computing ``P(S_t | X_{0,n} = 0)``.

    The BN itself **always covers all edges in the topology** because the
    RSR refs encode the full system function and the classifier would
    error out on a partial edge set. ``edge_subset`` only restricts which
    edges the conditioning sweep iterates over (i.e. which `n` we
    compute ``P(S_t | X_{0,n} = 0)`` for).

    The result is cached on disk at
    ``results/S_given_X0_zero_Kmax{K_max}_maxst{max_st}_n{n_sample}.csv``.
    Subsequent runs with the same ``(K_max, max_st, n_sample)`` skip any
    edge already present in the cache. Pass ``force_recompute=True`` to
    ignore the cache. Results are appended to the file after each edge,
    so an interrupted run still keeps the partial work.

    ``batch_size`` caps the peak memory of each ``inference.sample_evidence``
    call. ``None`` uses a single batch of ``n_sample``; set it to a value
    smaller than ``n_sample`` for large sample budgets that would otherwise
    OOM. Batching is statistically exact (raw counts are accumulated and
    only normalised at the end).
    """
    if device is None:
        device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

    out_path = RESULTS / (
        f"S_given_X0_zero_Kmax{K_max}_maxst{max_st}_n{n_sample}.csv"
    )

    # ---- load any cached rows ----
    if out_path.exists() and not force_recompute:
        cached = pd.read_csv(out_path)
        # backfill cov if the file pre-dates the column
        if "cov" not in cached.columns:
            cached["cov"] = cached["P(S_t=S | X_0_n=0)"].apply(
                lambda p: _binomial_cov(float(p), n_sample)
            )
        done = set(cached["edge"].unique())
        rows = cached.to_dict("records")
        print(f"Loaded cache with {len(done)} edges from {out_path}.")
    else:
        done = set()
        rows = []

    # ---- build the BN over the FULL edge set ----
    nodes, edges, midpoints = load_topology()
    region = region_from_nodes(nodes)
    refs_upper, refs_lower = load_rsr_refs(max_st=max_st, device=device)

    edge_names = list(edges.keys())

    # ---- decide which edges to sweep over for conditioning ----
    if edge_subset is not None:
        unknown = [e for e in edge_subset if e not in set(edge_names)]
        if unknown:
            raise ValueError(f"Unknown edges in edge_subset: {unknown}")
        target_edges = [e for e in edge_names if e in set(edge_subset)]
    else:
        target_edges = edge_names

    to_process = [n for n in target_edges if n not in done]
    if not to_process:
        print(f"All {len(target_edges)} target edges already cached; "
              "nothing to do.")
        return pd.DataFrame(rows)

    varis = define_variables(edge_names, K_max=K_max, max_st=max_st)
    probs = define_probs(
        varis, edges, midpoints, region,
        refs_upper, refs_lower, K_max=K_max, device=device,
    )

    already = len(target_edges) - len(to_process)
    print(f"Processing {len(to_process)} target edges "
          f"(skipping {already} already cached) over a BN of "
          f"{len(edge_names)} edges.")
    t_sweep = time.perf_counter()
    for i, n in enumerate(to_process):
        t_edge = time.perf_counter()
        print(f"edge {i + 1}/{len(to_process)} ({n}) — sampling...")

        probs_mod = condition_on_X0_zero(probs, varis, n, device=device)
        marg = estimate_S_marginals(probs_mod, varis, K_max, n_sample,
                                    max_st=max_st, batch_size=batch_size)

        for t in range(K_max + 1):
            for s in range(marg.shape[1]):
                p = float(marg[t, s])
                rows.append({
                    "edge": n,
                    "t": t,
                    "S": s,
                    "P(S_t=S | X_0_n=0)": p,
                    "cov": _binomial_cov(p, n_sample),
                })

        # incremental save so partial progress survives an interruption
        pd.DataFrame(rows).to_csv(out_path, index=False)

        dt = time.perf_counter() - t_edge
        elapsed = time.perf_counter() - t_sweep
        eta = elapsed / (i + 1) * (len(to_process) - (i + 1))
        print(f"  done {i + 1}/{len(to_process)} — "
              f"{dt:.1f}s, elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

    print(f"Saved: {out_path}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Small demonstration run. For the full network, drop ``edge_subset``
    # (and expect a long wall-clock — 129 conditional runs).
    # ``batch_size`` caps peak memory; raise/lower to fit your hardware.
    main(
        K_max=10,
        n_sample=1000000,
        edge_subset=None, # If None, calculate for all edges; otherwise, a list of edge names to process.
        batch_size=50000,
    )
