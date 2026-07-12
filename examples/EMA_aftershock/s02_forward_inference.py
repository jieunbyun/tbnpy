"""
Unconditional forward inference of the full EMA BN.

Samples all variables with no evidence and saves:
  results/forward_stats_Kmax{K}_maxst{M}_n{N}.csv - per-variable summary stats
  results/histograms/Kmax{K}_maxst{M}_n{N}/ - one PNG per variable,
      organised into sub-directories: global/, S/, A/, D/, X/, and t{i}/
      for the per-slot variables (R, V, L, M).
"""

import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

N_BINS = 60


def _resolve_device(device):
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    raise RuntimeError(
        "CUDA is not available. Set device='cpu' explicitly only if you want to run on CPU."
    )


def _sync_cuda(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


class _DiscAcc:
    """Accumulate state counts for a discrete variable."""

    def __init__(self, states):
        self.states = list(states)
        self.counts = np.zeros(len(states), dtype=np.int64)

    def update(self, arr):
        for i, s in enumerate(self.states):
            self.counts[i] += int((arr == s).sum())

    def probs(self):
        total = self.counts.sum()
        return self.counts / max(total, 1)


class _ContAcc:
    """Parallel Welford mean/variance + running histogram counts."""

    def __init__(self, n_bins=N_BINS):
        self.n_bins = n_bins
        self.n = 0
        self._mean = 0.0
        self._M2 = 0.0
        self.bin_edges = None
        self.hist_counts = None

    def update(self, arr):
        n_new = arr.size
        if n_new == 0:
            return
        mean_new = float(arr.mean())
        var_new = float(arr.var())

        if self.n == 0:
            self.n = n_new
            self._mean = mean_new
            self._M2 = var_new * n_new
        else:
            n_total = self.n + n_new
            delta = mean_new - self._mean
            self._mean = (self.n * self._mean + n_new * mean_new) / n_total
            self._M2 += var_new * n_new + delta ** 2 * self.n * n_new / n_total
            self.n = n_total

        if self.bin_edges is None:
            lo, hi = float(arr.min()), float(arr.max())
            pad = max(abs(lo), abs(hi)) * 1e-6 + 1e-30
            self.bin_edges = np.linspace(lo - pad, hi + pad, self.n_bins + 1)
            self.hist_counts = np.zeros(self.n_bins, dtype=np.int64)
        self.hist_counts += np.histogram(arr, bins=self.bin_edges)[0]

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return math.sqrt(max(self._M2 / max(self.n - 1, 1), 0.0))


def _make_accs(varis):
    return {
        name: (_DiscAcc(var.values) if isinstance(var.values, list)
               else _ContAcc())
        for name, var in varis.items()
    }


def _update_accs(filled, accs):
    for name, acc in accs.items():
        if name not in filled:
            continue
        Cs = filled[name].Cs
        if Cs.ndim == 1:
            values = Cs
        elif Cs.ndim == 2:
            values = Cs[:, 0]
        elif Cs.ndim == 3:
            values = Cs[0, :, 0]
        else:
            raise ValueError(f"Unexpected Cs shape for {name}: {Cs.shape}")
        arr = values.detach().cpu().numpy()
        acc.update(arr)


def _update_accs_batch(node_name, Cs_batch, accs):
    if node_name not in accs:
        return
    acc = accs[node_name]
    if Cs_batch.ndim == 1:
        values = Cs_batch
    elif Cs_batch.ndim == 2:
        values = Cs_batch[:, 0]
    elif Cs_batch.ndim == 3:
        values = Cs_batch[0, :, 0]
    else:
        raise ValueError(f"Unexpected Cs shape for {node_name}: {Cs_batch.shape}")
    arr = values.detach().cpu().numpy()
    acc.update(arr)


def _save_stats(accs, path):
    rows = []
    for name, acc in accs.items():
        if isinstance(acc, _DiscAcc):
            for s, p, c in zip(acc.states, acc.probs(), acc.counts):
                rows.append(dict(variable=name, type="discrete",
                                 state=s, prob=round(float(p), 8),
                                 count=int(c)))
        else:
            rows.append(dict(variable=name, type="continuous",
                             mean=acc.mean, std=acc.std, n=acc.n))
    pd.DataFrame(rows).to_csv(path, index=False)


def _var_subdir(name, K_max):
    """Map a variable name to a histogram sub-directory."""
    if name in ("L_x_0", "L_y_0", "M_0", "K"):
        return "global"
    for prefix in ("A", "D", "X"):
        if name.startswith(f"{prefix}_"):
            return prefix
    for t in range(K_max + 1):
        if name == f"S_{t}":
            return "S"
        if name in (f"R_{t}", f"V_{t}", f"L_x_{t}", f"L_y_{t}", f"M_{t}"):
            return f"t{t}"
    return "other"


def _save_histograms(accs, hist_dir, K_max):
    for name, acc in accs.items():
        subdir = hist_dir / _var_subdir(name, K_max)
        subdir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(4, 3))

        if isinstance(acc, _DiscAcc):
            ax.bar(range(len(acc.states)), acc.probs(),
                   tick_label=[str(s) for s in acc.states],
                   color="steelblue", alpha=0.85)
            ax.set_ylabel("P")
        else:
            if acc.bin_edges is None:
                plt.close(fig)
                continue
            widths = np.diff(acc.bin_edges)
            density = acc.hist_counts / max(acc.n, 1) / np.where(widths > 0, widths, 1)
            ax.bar(acc.bin_edges[:-1], density, width=widths, align="edge",
                   color="steelblue", alpha=0.85)
            ax.set_ylabel("Density")

        ax.set_xlabel(name)
        ax.set_title(name, fontsize=8)
        fig.tight_layout()
        fig.savefig(subdir / f"{name.replace('/', '_')}.png", dpi=120)
        plt.close(fig)


def main(K_max=10, n_sample=100_000, max_st=2, device=None,
         force_recompute=False, batch_size=100_000):
    t_total_start = time.perf_counter()
    device = _resolve_device(device)

    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    tag = f"Kmax{K_max}_maxst{max_st}_n{n_sample}"
    stats_path = RESULTS / f"forward_stats_{tag}.csv"
    hist_dir = RESULTS / "histograms" / tag
    timing_path = RESULTS / f"forward_timing_{tag}.csv"

    if stats_path.exists() and hist_dir.exists() and not force_recompute:
        print(f"Results exist ({stats_path}). Pass force_recompute=True to redo.")
        return pd.read_csv(stats_path)

    nodes, edges, midpoints = load_topology()
    region = region_from_nodes(nodes)
    refs_upper, refs_lower = load_rsr_refs(max_st=max_st, device=device)
    edge_names = list(edges.keys())

    _sync_cuda(device)
    t_build_start = time.perf_counter()
    varis = define_variables(edge_names, K_max=K_max, max_st=max_st)
    probs = define_probs(
        varis, edges, midpoints, region,
        refs_upper, refs_lower, K_max=K_max,
        s_fun=make_s_fun(), device=device,
    )
    _sync_cuda(device)
    build_model_sec = time.perf_counter() - t_build_start

    query_nodes = list(varis.keys())

    _sync_cuda(device)
    t_start = time.perf_counter()
    print(
        f"Forward sampling: {n_sample:,} samples, {len(query_nodes)} variables, "
        f"external batch_size={batch_size:,}."
    )

    curr_batch = int(batch_size)
    while True:
        accs = _make_accs(varis)
        n_done = 0

        def _consume_batch(node_name, Cs_batch, _ps_batch, start, end):
            nonlocal n_done
            _update_accs_batch(node_name, Cs_batch, accs)
            if node_name == query_nodes[0]:
                n_done = end

        try:
            inference.sample(
                probs=probs,
                query_nodes=query_nodes,
                n_sample=n_sample,
                batch_size=curr_batch,
                on_batch=_consume_batch,
                accumulate_query=False,
            )
        except torch.OutOfMemoryError:
            if device.type != "cuda":
                raise
            torch.cuda.empty_cache()
            next_batch = curr_batch // 2
            if next_batch < 100:
                raise
            curr_batch = next_batch
            print(f"  CUDA OOM; retrying with smaller batch_size={curr_batch:,}")
            continue
        break

    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  progress: {n_done:,}/{n_sample:,} samples")

    _sync_cuda(device)
    sampling_sec = time.perf_counter() - t_start

    print(f"Sampling done in {sampling_sec:.1f}s. "
          "Saving stats and histograms...")

    t_stats_start = time.perf_counter()
    _save_stats(accs, stats_path)
    save_stats_sec = time.perf_counter() - t_stats_start

    t_hist_start = time.perf_counter()
    _save_histograms(accs, hist_dir, K_max)
    save_histograms_sec = time.perf_counter() - t_hist_start

    total_sec = time.perf_counter() - t_total_start

    timing_df = pd.DataFrame([
        {
            "device": str(device),
            "K_max": int(K_max),
            "max_st": int(max_st),
            "n_sample": int(n_sample),
            "batch_size": int(curr_batch),
            "n_variables": int(len(query_nodes)),
            "build_model_sec": build_model_sec,
            "sampling_sec": sampling_sec,
            "save_stats_sec": save_stats_sec,
            "save_histograms_sec": save_histograms_sec,
            "total_sec": total_sec,
        }
    ])
    timing_df.to_csv(timing_path, index=False)

    n_plots = sum(1 for acc in accs.values()
                  if isinstance(acc, _DiscAcc)
                  or (isinstance(acc, _ContAcc) and acc.bin_edges is not None))
    print(f"Stats     -> {stats_path}")
    print(f"Histograms -> {hist_dir}  ({n_plots} files)")
    print(f"Timing    -> {timing_path}")
    return pd.read_csv(stats_path)


if __name__ == "__main__":
    main(
        K_max=10,
        n_sample=1_000_000,
        force_recompute=True,
    )
