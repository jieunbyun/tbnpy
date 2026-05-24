"""
Draw the EMA highway network and overlay the bounding rectangle used as
``L_0``'s uniform support.

The region is the axis-aligned bounding box of the node coordinates,
expanded by ``margin_frac`` (default 0.30 = 30%) on each of the four
sides.

Run as::

    python plot_network.py

The figure is saved to ``results/ema_network.png``.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from s01_define_model import load_topology, region_from_nodes  # noqa: E402


RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


DESTS = ["n22", "n66"]


def plot_network(margin_frac=0.30, fname="ema_network.png",
                 fontsize_label=16, fontsize_tick=16, fontsize_legend=14,
                 dests=None):
    if dests is None:
        dests = DESTS

    nodes, edges, _ = load_topology()
    region = region_from_nodes(nodes, margin_frac=margin_frac)
    x_min, x_max, y_min, y_max = region

    fig, ax = plt.subplots(figsize=(7, 7))

    # ---- edges ----
    for e in edges.values():
        a, b = nodes[e["from"]], nodes[e["to"]]
        ax.plot([a["x"], b["x"]], [a["y"], b["y"]],
                color="0.5", linewidth=0.8, zorder=1)

    # ---- nodes ----
    dest_set = set(dests)
    node_ids = list(nodes.keys())
    xs = [nodes[n]["x"] for n in node_ids]
    ys = [nodes[n]["y"] for n in node_ids]
    pops = [nodes[n].get("population", 0.0) for n in node_ids]
    max_pop = max(pops) if max(pops) > 0 else 1.0
    sizes = [10 + 30 * (p / max_pop) for p in pops]

    reg_idx  = [i for i, n in enumerate(node_ids) if n not in dest_set]
    dest_idx = [i for i, n in enumerate(node_ids) if n in dest_set]

    ax.scatter([xs[i] for i in reg_idx], [ys[i] for i in reg_idx],
               s=[sizes[i] for i in reg_idx],
               c="tab:blue", edgecolor="tab:blue", linewidths=0.6, zorder=2,
               label="nodes\n(size $\propto$ population)")
    ax.scatter([xs[i] for i in dest_idx], [ys[i] for i in dest_idx],
               s=[sizes[i]*1.5 for i in dest_idx],
               c="tab:orange", edgecolor="tab:orange", linewidths=0.8, zorder=3,
               marker="*",
               label=f"destinations\n({', '.join(dests)})")

    # ---- region rectangle ----
    rect = Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor="tab:red", linewidth=1.8,
        linestyle="--", zorder=3,
        label=(f"L_0 support\n"
               f"x=[{x_min:.1f}, {x_max:.1f}]\n"
               f"y=[{y_min:.1f}, {y_max:.1f}]"),
    )
    ax.add_patch(rect)

    # ---- annotation: node-bbox vs region ----
    ax.axvline(min(xs), color="0.8", linewidth=0.5, zorder=0)
    ax.axvline(max(xs), color="0.8", linewidth=0.5, zorder=0)
    ax.axhline(min(ys), color="0.8", linewidth=0.5, zorder=0)
    ax.axhline(max(ys), color="0.8", linewidth=0.5, zorder=0)

    ax.set_xlabel("x (km)", fontsize=fontsize_label)
    ax.set_ylabel("y (km)", fontsize=fontsize_label)
    ax.tick_params(labelsize=fontsize_tick)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=fontsize_legend)

    out_path = RESULTS / fname
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Region: x in [{x_min:.3f}, {x_max:.3f}], "
          f"y in [{y_min:.3f}, {y_max:.3f}]")
    return region


if __name__ == "__main__":
    plot_network(margin_frac=0.20)
