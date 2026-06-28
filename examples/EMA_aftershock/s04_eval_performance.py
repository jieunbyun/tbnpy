"""
Evaluate wall time for forward sampling in the EMA example.

The benchmark builds the BN once and measures the wall time needed to
forward-sample either every EMA variable, as in ``s02_forward_inference.py``,
or one requested query node for increasing sample counts.

Outputs
-------
results/performance_sampling_Kmax{K}_maxst{M}_{query}.csv
results/performance_sampling_Kmax{K}_maxst{M}_{query}.png
"""

import argparse
import ctypes
import os
import platform
import sys
import time
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

REPO_ROOT = BASE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RSR_REPO = REPO_ROOT.parent / "rsr"
if RSR_REPO.exists() and str(RSR_REPO) not in sys.path:
    sys.path.insert(0, str(RSR_REPO))

from s01_define_model import (  # noqa: E402
    define_probs,
    define_variables,
    load_rsr_refs,
    load_topology,
    make_s_fun,
    region_from_nodes,
)
from tbnpy import inference  # noqa: E402


RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

DEFAULT_SAMPLE_SIZES = [10 ** p for p in range(1, 7)]
DEFAULT_QUERY_NODE = "all" # To change to single nodes, e.g. "x001"
PLOT_FONT_SIZE = 16


def _format_sample_label(n_sample):
    if n_sample >= 1_000_000:
        return f"{n_sample // 1_000_000}M"
    if n_sample >= 1_000:
        return f"{n_sample // 1_000}k"
    return str(n_sample)


def _x_position_for_value(value, sorted_values):
    """Map a numeric x value onto the bar-index axis."""
    values = [float(v) for v in sorted_values]
    value = float(value)
    if value <= values[0]:
        return 0.0
    if value >= values[-1]:
        return float(len(values) - 1)

    for i, (left, right) in enumerate(zip(values[:-1], values[1:])):
        if left <= value <= right:
            frac = (value - left) / (right - left)
            return i + frac

    return None


def _safe_tag(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(" ", "_")


def resolve_query_nodes(query_node, probs, varis):
    """Resolve user shorthand such as ``x001`` or ``all`` to BN node names."""
    if str(query_node).strip().lower() in ("all", "*"):
        return list(varis.keys()), "all"

    if query_node in probs:
        return [query_node], query_node

    q = str(query_node).strip()
    if q in probs:
        return [q], q

    # Accept case-insensitive exact node names, e.g. x_0_e0001.
    lower_to_node = {name.lower(): name for name in probs}
    if q.lower() in lower_to_node:
        node = lower_to_node[q.lower()]
        return [node], node

    # Accept edge ids, e.g. e0001 -> X_0_e0001.
    if q.lower().startswith("e"):
        candidate = f"X_0_{q.lower()}"
        if candidate in probs:
            return [candidate], candidate

    # Accept compact X shorthand, e.g. x001 -> X_0_e0001.
    if q.lower().startswith("x") and q[1:].isdigit():
        candidate = f"X_0_e{int(q[1:]):04d}"
        if candidate in probs:
            return [candidate], candidate

    raise ValueError(
        f"Unknown query node {query_node!r}. Use a BN node such as "
        "'X_0_e0001', an edge id such as 'e0001', shorthand 'x001', or 'all'."
    )


def _get_total_ram_gb():
    """Return total system memory in GiB when available."""
    if platform.system() == "Windows":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullTotalPhys / (1024 ** 3)
        return None

    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            return None
        return pages * page_size / (1024 ** 3)

    return None


def get_desktop_spec():
    """Build a compact desktop-spec label for benchmark plots."""
    cpu = (
        platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER")
        or platform.machine()
        or "CPU"
    )
    cpu = " ".join(cpu.split())
    cores = os.cpu_count()
    ram_gb = _get_total_ram_gb()

    parts = [cpu]
    if cores is not None:
        parts.append(f"{cores} logical cores")
    if ram_gb is not None:
        parts.append(f"{ram_gb:.0f} GiB RAM")
    return ", ".join(parts)


def build_model(K_max=10, max_st=2, query_node=DEFAULT_QUERY_NODE, device=None):
    """Build the EMA BN and resolve the requested query nodes."""
    if device is None:
        device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

    nodes, edges, midpoints = load_topology()
    edge_names = list(edges.keys())

    region = region_from_nodes(nodes)
    refs_upper, refs_lower = load_rsr_refs(max_st=max_st, device=device)
    varis = define_variables(edge_names, K_max=K_max, max_st=max_st)
    probs = define_probs(
        varis,
        edges,
        midpoints,
        region,
        refs_upper,
        refs_lower,
        K_max=K_max,
        s_fun=make_s_fun(),
        device=device,
    )
    query_nodes, query_label = resolve_query_nodes(query_node, probs, varis)
    return probs, query_nodes, query_label, device


def run_benchmark(sample_sizes=None, K_max=10, max_st=2,
                  query_node=DEFAULT_QUERY_NODE,
                  batch_size=100_000, repeats=1, device=None):
    """Return one timing row per ``n_sample`` and repeat."""
    if sample_sizes is None:
        sample_sizes = DEFAULT_SAMPLE_SIZES
    sample_sizes = [int(n) for n in sample_sizes]
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    probs, query_nodes, query_label, device = build_model(
        K_max=K_max,
        max_st=max_st,
        query_node=query_node,
        device=device,
    )

    print(
        f"Benchmarking {len(query_nodes)} query node(s) ({query_label}): "
        f"K_max={K_max}, max_st={max_st}, "
        f"device={device}, batch_size={batch_size:,}."
    )
    rows = []
    for n_sample in sample_sizes:
        for rep in range(1, repeats + 1):
            print(f"  n_sample={n_sample:,}, repeat={rep}/{repeats}")
            t_start = time.perf_counter()
            inference.sample(
                probs=probs,
                query_nodes=query_nodes,
                n_sample=n_sample,
                batch_size=batch_size,
            )
            wall_time_s = time.perf_counter() - t_start
            rows.append({
                "n_sample": n_sample,
                "repeat": rep,
                "wall_time_s": wall_time_s,
                "samples_per_s": n_sample / wall_time_s,
                "K_max": K_max,
                "max_st": max_st,
                "query_node": query_label,
                "n_query_nodes": len(query_nodes),
                "batch_size": batch_size,
                "device": device,
            })
            print(f"    wall time: {wall_time_s:.3f}s")

    return pd.DataFrame(rows)


def save_outputs(df, K_max=10, max_st=2, query_node=None, desktop_spec=None):
    """Save timing CSV and a bar chart of mean wall time by sample count."""
    query_tag = _safe_tag(query_node)
    tag = f"Kmax{K_max}_maxst{max_st}_{query_tag}"
    csv_path = RESULTS / f"performance_sampling_{tag}.csv"
    fig_path = RESULTS / f"performance_sampling_{tag}.png"

    df.to_csv(csv_path, index=False)

    summary = (
        df.groupby("n_sample", as_index=False)
        .agg(
            mean_wall_time_s=("wall_time_s", "mean"),
            std_wall_time_s=("wall_time_s", "std"),
        )
        .sort_values("n_sample")
    )
    yerr = summary["std_wall_time_s"].fillna(0.0).to_numpy()
    n_samples = summary["n_sample"].to_numpy()
    labels = [_format_sample_label(n) for n in summary["n_sample"]]
    batch_size = int(df["batch_size"].iloc[0])

    if desktop_spec is None:
        desktop_spec = get_desktop_spec()

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        labels,
        summary["mean_wall_time_s"],
        yerr=yerr,
        color="steelblue",
        alpha=0.88,
        capsize=6,
    )
    ax.set_xlabel("Number of samples", fontsize=PLOT_FONT_SIZE)
    ax.set_ylabel("Wall time (seconds)", fontsize=PLOT_FONT_SIZE)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=PLOT_FONT_SIZE)

    ymax = max(float(summary["mean_wall_time_s"].max()), 1e-12)
    ax.set_ylim(top=ymax * 3.0)
    for bar, wall_time in zip(bars, summary["mean_wall_time_s"]):
        ax.annotate(
            f"{wall_time:.1e}",
            xy=(bar.get_x() + bar.get_width() / 2, wall_time),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=PLOT_FONT_SIZE,
        )

    batch_x = _x_position_for_value(batch_size, n_samples)
    if batch_x is not None:
        ax.axvline(
            batch_x,
            color="tab:red",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
        )
        ax.text(
            batch_x + 0.04,
            ymax * 0.98,
            f"batch size = {_format_sample_label(batch_size)}",
            color="tab:red",
            rotation=90,
            va="top",
            ha="left",
            fontsize=PLOT_FONT_SIZE,
        )

    desktop_lines = textwrap.wrap(f"Desktop: {desktop_spec}", width=78)
    query_title = (
        "Forward-sampling time for all variables"
        if query_node == "all"
        else f"Forward-sampling time for {query_node}"
    )
    ax.set_title(
        "\n".join([
            query_title,
            *desktop_lines,
        ]),
        fontsize=PLOT_FONT_SIZE,
    )
    ax.grid(axis="y", which="both", color="0.9", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved CSV  -> {csv_path}")
    print(f"Saved plot -> {fig_path}")
    return csv_path, fig_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark EMA forward-sampling wall time."
    )
    parser.add_argument("--K-max", type=int, default=10)
    parser.add_argument("--max-st", type=int, default=2)
    parser.add_argument(
        "--query-node",
        default=DEFAULT_QUERY_NODE,
        help=(
            "Node to query. Accepts full names like X_0_e0001, edge ids like "
            "e0001, shorthand x001, or all for list(varis.keys()). Default: all."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_SIZES,
        help="Sample sizes to benchmark. Default: 10 100 ... 1000000.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Override device. By default USE_CUDA=1 selects cuda, otherwise cpu.",
    )
    parser.add_argument(
        "--desktop-spec",
        default=None,
        help="Override the auto-detected desktop spec used in the plot title.",
    )
    return parser.parse_args()


def main(sample_sizes=None, K_max=10, max_st=2,
         query_node=DEFAULT_QUERY_NODE,
         batch_size=100_000, repeats=1, device=None, desktop_spec=None):
    df = run_benchmark(
        sample_sizes=sample_sizes,
        K_max=K_max,
        max_st=max_st,
        query_node=query_node,
        batch_size=batch_size,
        repeats=repeats,
        device=device,
    )
    save_outputs(
        df,
        K_max=K_max,
        max_st=max_st,
        query_node=df["query_node"].iloc[0],
        desktop_spec=desktop_spec,
    )
    return df


if __name__ == "__main__":
    args = parse_args()
    main(
        sample_sizes=args.sample_sizes,
        K_max=args.K_max,
        max_st=args.max_st,
        query_node=args.query_node,
        batch_size=args.batch_size,
        repeats=args.repeats,
        device=args.device,
        desktop_spec=args.desktop_spec,
    )
