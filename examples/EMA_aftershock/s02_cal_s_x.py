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

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

REPO_ROOT = BASE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RSR_REPO = REPO_ROOT.parent / "rsr"
if RSR_REPO.exists() and str(RSR_REPO) not in sys.path:
    sys.path.insert(0, str(RSR_REPO))

from tbnpy import inference  # noqa: E402

from s1_define_model import (  # noqa: E402
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


def estimate_S_marginals(probs, varis, K_max, n_sample, device="cpu"):
    """Forward-sample and return ``P(S_t = s)`` for every ``t``.

    Returns
    -------
    np.ndarray of shape (K_max + 1, n_states)
        Row ``t`` is the empirical probability vector for ``S_t``.
    """
    query_nodes = [f"S_{t}" for t in range(K_max + 1)]
    evidence_df = pd.DataFrame(index=[0])  # one dummy row, no columns

    filled = inference.sample_evidence(
        probs=probs,
        query_nodes=query_nodes,
        n_sample=n_sample,
        evidence_df=evidence_df,
    )

    # S_t variables are 3-state {0, 1, 2}
    n_states = 3
    out = np.zeros((K_max + 1, n_states), dtype=np.float64)
    for t in range(K_max + 1):
        Cs = filled[f"S_{t}"].Cs  # (1, n_sample, 1 + n_parents) — child first
        s_samples = Cs[0, :, 0].detach().cpu().numpy().astype(int)
        counts = np.bincount(s_samples, minlength=n_states)
        out[t] = counts / counts.sum()
    return out


def main(K_max=2, n_sample=2000, edge_subset=None, device=None):
    if device is None:
        device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

    nodes, edges, midpoints = load_topology(edge_subset=edge_subset)
    region = region_from_nodes(nodes)
    refs_upper, refs_lower = load_rsr_refs(max_st=2, device=device)

    edge_names = list(edges.keys())
    varis = define_variables(edge_names, K_max=K_max)
    probs = define_probs(
        varis, edges, midpoints, region,
        refs_upper, refs_lower, K_max=K_max, device=device,
    )

    rows = []
    for i, n in enumerate(edge_names):
        probs_mod = condition_on_X0_zero(probs, varis, n, device=device)
        marg = estimate_S_marginals(probs_mod, varis, K_max, n_sample, device)

        for t in range(K_max + 1):
            for s in range(marg.shape[1]):
                rows.append({
                    "edge": n,
                    "t": t,
                    "S": s,
                    "P(S_t=S | X_0_n=0)": float(marg[t, s]),
                })

        if (i + 1) % 10 == 0 or (i + 1) == len(edge_names):
            print(f"  processed {i + 1}/{len(edge_names)} edges")

    df = pd.DataFrame(rows)
    out_path = RESULTS / f"S_given_X0_zero_Kmax{K_max}_n{n_sample}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return df


if __name__ == "__main__":
    # Small demonstration run. For the full network, drop ``edge_subset``
    # (and expect a long wall-clock — 129 conditional runs).
    main(
        K_max=2,
        n_sample=2000,
        edge_subset=["e0001", "e0002", "e0003"],
    )
