"""
Draw the EMA highway network and overlay the bounding rectangle used as
``L_0``'s uniform support (the areal source).

The region is the axis-aligned bounding box of the node coordinates,
expanded by ``margin_frac`` (default 0.30 = 30%) on each of the four
sides.

Styling follows ``draw_graph_population.py``: a compact portrait figure with
node marker *diameters* measured in data units -- the same units as the edge
lengths -- so nodes stay proportionate to the graph geometry at any figure
size. A size key relates marker size to population.

Run as::

    python plot_network.py

The figure is saved to ``results/ema_network.png`` (with an SVG copy).
"""

import math
import sys
from pathlib import Path

import matplotlib

# Keep SVG text editable (e.g., in Inkscape) instead of outlining it to paths.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from s01_define_model import load_topology, region_from_nodes  # noqa: E402


RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)


DESTS = ["n22", "n66"]

EDGE_COLOR = "#9aa4ad"
NODE_FILL = "tab:blue"
DEST_COLOR = "tab:orange"
SOURCE_COLOR = "tab:red"

# Node *diameters* as a fraction of the median edge length. Area still grows
# with population (diameter ~ sqrt(pop)); the floor keeps zero-pop nodes visible.
MIN_DIAM_FRAC = 0.20
MAX_DIAM_FRAC = 0.85
# Destination-star diameter as a fraction of the median edge length.
STAR_DIAM_FRAC = 1.10
# Population values shown in the size key.
LEGEND_POPS = (500, 1500, 3000)


def _points_per_data_unit(fig, ax) -> float:
    """Length of one data unit expressed in typographic points.

    Marker sizes (scatter ``s``) are areas in points**2, independent of the
    data scale, so to size nodes in data units we convert here. Requires the
    axes limits to be final and the canvas drawn.
    """
    trans = ax.transData
    (x0, _), (x1, _) = trans.transform((0.0, 0.0)), trans.transform((1.0, 0.0))
    px_per_data = abs(x1 - x0)
    return px_per_data * 72.0 / fig.dpi


def plot_network(margin_frac=0.30, fname="ema_network.png",
                 figsize=(2.2, 2.6), dpi=300, fontsize=5, title_fontsize=6,
                 dests=None):
    if dests is None:
        dests = DESTS

    nodes, edges, _ = load_topology()
    region = region_from_nodes(nodes, margin_frac=margin_frac)
    x_min, x_max, y_min, y_max = region

    pos = {nid: (float(a["x"]), float(a["y"])) for nid, a in nodes.items()}
    pop = {nid: float(a.get("population", 0.0)) for nid, a in nodes.items()}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- edges ----
    segments = [(pos[e["from"]], pos[e["to"]]) for e in edges.values()]
    edge_lengths = [math.dist(a, b) for a, b in segments]
    median_edge = sorted(edge_lengths)[len(edge_lengths) // 2]
    for (a, b) in segments:
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=EDGE_COLOR, linewidth=0.6, zorder=1)

    # ---- region rectangle (areal source) ----
    rect = Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor=SOURCE_COLOR, linewidth=1.0,
        linestyle="--", zorder=1,
    )
    ax.add_patch(rect)

    # Diameter in data units: sqrt scaling so *area* tracks population, with a
    # floor so the smallest nodes stay visible. Reference = median edge length.
    max_pop = max(pop.values()) or 1.0
    min_d = MIN_DIAM_FRAC * median_edge
    max_d = MAX_DIAM_FRAC * median_edge

    def diam_for(p: float) -> float:
        return min_d + (max_d - min_d) * math.sqrt(p / max_pop)

    ax.margins(0.05)
    ax.autoscale_view()

    # Limits are final now; convert data-unit diameters to marker areas (pt**2).
    fig.canvas.draw()
    ppd = _points_per_data_unit(fig, ax)

    def area(diam_units: float) -> float:
        return (diam_units * ppd) ** 2

    dest_set = set(dests)
    plain = [nid for nid in pos if nid not in dest_set]
    ax.scatter(
        [pos[n][0] for n in plain],
        [pos[n][1] for n in plain],
        s=[area(diam_for(pop[n])) for n in plain],
        c=NODE_FILL,
        edgecolors="none",
        zorder=2,
        clip_on=False,
    )

    # Destinations as orange stars, sized to stand out regardless of population.
    origins = [n for n in dests if n in pos]
    if origins:
        ax.scatter(
            [pos[n][0] for n in origins],
            [pos[n][1] for n in origins],
            s=area(STAR_DIAM_FRAC * median_edge),
            marker="*",
            c=DEST_COLOR,
            edgecolors="none",
            zorder=3,
            clip_on=False,
        )

    # ---- legend: categories (destinations + areal source) ----
    cat_handles = [
        Line2D([], [], marker="*", linestyle="none",
               markerfacecolor=DEST_COLOR, markeredgecolor="none",
               markersize=STAR_DIAM_FRAC * median_edge * ppd,
               label="Destinations"),
        Line2D([], [], color=SOURCE_COLOR, linewidth=1.0, linestyle="--",
               label="Areal source"),
    ]
    cat_legend = ax.legend(
        handles=cat_handles,
        loc="upper left",
        frameon=False,
        handletextpad=0.8,
        borderpad=0.0,
        fontsize=fontsize,
    )
    ax.add_artist(cat_legend)

    # ---- size key: reference circles at known populations, drawn with the
    # exact same size mapping as the nodes (markersize is a diameter in points,
    # so it equals the data-unit diameter times points-per-data-unit). ----
    size_handles = [
        Line2D([], [], marker="o", linestyle="none",
               markerfacecolor=NODE_FILL, markeredgecolor="none",
               markersize=diam_for(p) * ppd, label=f"{p:,}")
        for p in LEGEND_POPS
    ]
    size_legend = ax.legend(
        handles=size_handles,
        title="Population\n(node size)",
        loc="lower left",
        frameon=False,
        labelspacing=1.0,
        handletextpad=0.8,
        borderpad=0.0,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
    )
    size_legend.get_title().set_ha("left")

    out_path = RESULTS / fname
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    # Also emit a vector copy (SVG) alongside, unless the primary output is
    # already a vector format.
    outputs = [out_path]
    if out_path.suffix.lower() not in (".svg", ".pdf", ".eps"):
        vector_path = out_path.with_suffix(".svg")
        fig.savefig(vector_path, bbox_inches="tight")
        outputs.append(vector_path)
    plt.close(fig)

    for out in outputs:
        print(f"Saved: {out}")
    print(f"Region: x in [{x_min:.3f}, {x_max:.3f}], "
          f"y in [{y_min:.3f}, {y_max:.3f}]")
    return region


if __name__ == "__main__":
    plot_network(margin_frac=0.20)
