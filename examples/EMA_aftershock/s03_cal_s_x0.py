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

1. Loads the full BN once via ``s01_define_model``.
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
    """Coefficient of variation of an empirical proportion ``p``."""
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
    make_s_fun,
    region_from_nodes,
)


RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


def _sync_cuda(device):
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


class DeltaD:
    """Degenerate distribution ``f(D = value) = 1``.

    Parents are kept so that the ancestor-ordering machinery in
    ``inference.sample`` continues to work. They are ignored during
    sampling and log-prob evaluation.
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
    """Return a shallow copy of ``probs`` with ``D_{0,edge_id}`` pinned to 1.0."""
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
    as ``-1`` are tallied and reported once at the end if any appear.

    ``batch_size`` is used as an external chunk size to avoid keeping all
    sampled tensors on GPU at once. If CUDA OOM occurs, this function
    halves the chunk size and retries.
    """
    n_states = max_st + 1
    counts = np.zeros((K_max + 1, n_states), dtype=np.int64)
    totals = np.zeros(K_max + 1, dtype=np.int64)
    unknowns = np.zeros(K_max + 1, dtype=np.int64)

    query_nodes = [f"S_{t}" for t in range(K_max + 1)]
    if batch_size is None:
        batch_size = min(2_000, n_sample)
    else:
        batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    _sync_cuda(probs[query_nodes[0]].device)
    t_start = time.perf_counter()

    n_done = 0
    curr_batch = batch_size
    oom_retries = 0
    while n_done < n_sample:
        n_this = min(curr_batch, n_sample - n_done)
        try:
            filled = inference.sample(
                probs=probs,
                query_nodes=query_nodes,
                n_sample=n_this,
                batch_size=n_this,
            )
        except torch.OutOfMemoryError:
            dev = torch.device(probs[query_nodes[0]].device)
            if dev.type != "cuda":
                raise
            torch.cuda.empty_cache()
            next_batch = curr_batch // 2
            if next_batch < 100:
                raise
            curr_batch = next_batch
            oom_retries += 1
            if verbose:
                print(f"    CUDA OOM, retry with chunk={curr_batch:,}")
            continue

        for t in range(K_max + 1):
            Cs = filled[f"S_{t}"].Cs
            if Cs.ndim == 1:
                s_values = Cs
            else:
                s_values = Cs[:, 0]
            s_samples = s_values.detach().cpu().numpy().astype(int)
            unknowns[t] += int((s_samples < 0).sum())
            valid = s_samples[s_samples >= 0]
            counts[t] += np.bincount(valid, minlength=n_states)
            totals[t] += len(valid)

        n_done += n_this
        del filled
        dev = torch.device(probs[query_nodes[0]].device)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        if verbose and (n_done == n_sample or n_done % max(curr_batch * 5, 1) == 0):
            print(f"    progress {n_done:,}/{n_sample:,} (chunk={curr_batch:,})")

    _sync_cuda(probs[query_nodes[0]].device)
    sample_sec = time.perf_counter() - t_start
    if verbose:
        print(
            f"    sampled {n_sample} draws in {sample_sec:.1f}s "
            f"(final chunk={curr_batch:,}, oom_retries={oom_retries})"
        )

    t_calc_by_t = np.zeros(K_max + 1, dtype=np.float64)

    for t in range(K_max + 1):
        t_calc_start = time.perf_counter()
        # Count extraction is performed during chunked sampling above.
        t_calc_by_t[t] = time.perf_counter() - t_calc_start

    for t in range(K_max + 1):
        if unknowns[t] > 0:
            print(f"  WARN  S_{t}: {unknowns[t]}/{n_sample} samples are "
                  f"unknown (-1); consider passing s_fun to s_mod.S.")

    out = counts.astype(np.float64) / np.maximum(totals[:, None], 1)
    timing = {
        "sample_sec": float(sample_sec),
        "calc_t_sec": t_calc_by_t,
        "calc_total_sec": float(t_calc_by_t.sum()),
        "total_sec": float(sample_sec + t_calc_by_t.sum()),
    }
    return out, timing


def main(K_max=2, n_sample=2000, edge_subset=None, max_st=2,
         device=None, force_recompute=False, batch_size=None):
    """Sweep edges, computing ``P(S_t | X_{0,n} = 0)``.

    The BN itself always covers all edges in the topology because the
    RSR refs encode the full system function. ``edge_subset`` only
    restricts which edges the conditioning sweep iterates over.

    ``batch_size`` is passed to ``inference.sample`` and controls that
    routine's internal per-node sample chunks. ``None`` uses the
    default internal batch size.
    """
    if device is None:
        device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

    out_path = RESULTS / (
        f"S_given_X0_zero_Kmax{K_max}_maxst{max_st}_n{n_sample}.csv"
    )
    timing_path = RESULTS / (
        f"S_given_X0_zero_timing_Kmax{K_max}_maxst{max_st}_n{n_sample}.csv"
    )

    if out_path.exists() and not force_recompute:
        cached = pd.read_csv(out_path)
        if "cov" not in cached.columns:
            cached["cov"] = cached["P(S_t=S | X_0_n=0)"].apply(
                lambda p: _binomial_cov(float(p), n_sample)
            )
        for col, default in (
            ("edge_total_sec", np.nan),
            ("sample_sec", np.nan),
            ("calc_t_sec", np.nan),
            ("prob_calc_sec", np.nan),
        ):
            if col not in cached.columns:
                cached[col] = default
        done = set(cached["edge"].unique())
        rows = cached.to_dict("records")
        print(f"Loaded cache with {len(done)} edges from {out_path}.")
    else:
        done = set()
        rows = []

    nodes, edges, midpoints = load_topology()
    region = region_from_nodes(nodes)
    refs_upper, refs_lower = load_rsr_refs(max_st=max_st, device=device)
    edge_names = list(edges.keys())

    if edge_subset is not None:
        unknown = [e for e in edge_subset if e not in set(edge_names)]
        if unknown:
            raise ValueError(f"Unknown edges in edge_subset: {unknown}")
        target_edges = [e for e in edge_names if e in set(edge_subset)]
    else:
        target_edges = edge_names

    to_process = [n for n in target_edges if n not in done]
    if not to_process:
        print(f"All {len(target_edges)} target edges already cached; nothing to do.")
        return pd.DataFrame(rows)

    varis = define_variables(edge_names, K_max=K_max, max_st=max_st)
    probs = define_probs(
        varis, edges, midpoints, region,
        refs_upper, refs_lower, K_max=K_max,
        s_fun=make_s_fun(), device=device,
    )

    already = len(target_edges) - len(to_process)
    print(f"Processing {len(to_process)} target edges "
          f"(skipping {already} already cached) over a BN of "
          f"{len(edge_names)} edges.")
    for i, n in enumerate(to_process):
        t_edge = time.perf_counter()
        print(f"edge {i + 1}/{len(to_process)} ({n}) - sampling...")

        probs_mod = condition_on_X0_zero(probs, varis, n, device=device)
        marg, timing = estimate_S_marginals(
            probs_mod,
            varis,
            K_max,
            n_sample,
            max_st=max_st,
            batch_size=batch_size,
        )

        for t in range(K_max + 1):
            for s in range(marg.shape[1]):
                p = float(marg[t, s])
                rows.append({
                    "edge": n,
                    "t": t,
                    "S": s,
                    "P(S_t=S | X_0_n=0)": p,
                    "cov": _binomial_cov(p, n_sample),
                    "edge_total_sec": timing["total_sec"],
                    "sample_sec": timing["sample_sec"],
                    "calc_t_sec": float(timing["calc_t_sec"][t]),
                    "prob_calc_sec": float(timing["calc_t_sec"][t]) / marg.shape[1],
                })

        pd.DataFrame(rows).to_csv(out_path, index=False)

        dt = time.perf_counter() - t_edge
        print(f"  done {i + 1}/{len(to_process)} - {dt:.1f}s")

    timing_cols = [
        "edge", "edge_total_sec", "sample_sec", "calc_t_sec", "prob_calc_sec"
    ]
    timing_df = pd.DataFrame(rows)[timing_cols].drop_duplicates()
    timing_df.to_csv(timing_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Timing: {timing_path}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(
        K_max=10,
        n_sample=1_000_000,
        edge_subset=None,
        batch_size=100_000,
        force_recompute=False,
    )
