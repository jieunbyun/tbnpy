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


def plot_network(margin_frac=0.30, fname="ema_network.png"):
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
    xs = [n["x"] for n in nodes.values()]
    ys = [n["y"] for n in nodes.values()]
    pops = [n.get("population", 0.0) for n in nodes.values()]
    sizes = [10 + 30 * (p / max(pops) if max(pops) > 0 else 0) for p in pops]
    ax.scatter(xs, ys, s=sizes, c="tab:blue", edgecolor="white",
               linewidths=0.6, zorder=2, label="nodes")

    # ---- region rectangle ----
    rect = Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor="tab:red", linewidth=1.8,
        linestyle="--", zorder=3,
        label=f"L_0 support ({int(margin_frac * 100)}% margin)",
    )
    ax.add_patch(rect)

    # ---- annotation: node-bbox vs region ----
    ax.axvline(min(xs), color="0.8", linewidth=0.5, zorder=0)
    ax.axvline(max(xs), color="0.8", linewidth=0.5, zorder=0)
    ax.axhline(min(ys), color="0.8", linewidth=0.5, zorder=0)
    ax.axhline(max(ys), color="0.8", linewidth=0.5, zorder=0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"EMA network ({len(nodes)} nodes, {len(edges)} edges)\n"
        f"region = ({x_min:.1f}, {x_max:.1f}, {y_min:.1f}, {y_max:.1f})"
    )

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
