from ndtools import graphs
import json, math
import matplotlib.pyplot as plt

import os
BASE = os.path.dirname(os.path.abspath(__file__))

def compute_edge_centers(nodes, edges):
    centers = {}
    for eid, e in edges.items():
        n1 = nodes[e["from"]]
        n2 = nodes[e["to"]]
        cx = 0.5 * (n1["x"] + n2["x"])
        cy = 0.5 * (n1["y"] + n2["y"])
        centers[eid] = (cx, cy)
    return centers


def compute_source_edge_center_distances(nodes, edges, sources):
    """
    Returns:
        {edge_id: {source_id: distance_km}}
    """
    centers = compute_edge_centers(nodes, edges)
    dists = {}

    for eid, (cx, cy) in centers.items():
        dists[eid] = {}
        for sid, s in sources.items():
            sx, sy = s["x"], s["y"]
            d = math.sqrt((sx - cx)**2 + (sy - cy)**2)
            dists[eid][sid] = d
    return dists

def load_data():
    edges = json.load(open(os.path.join(BASE, 'edges.json')))
    nodes = json.load(open(os.path.join(BASE, 'nodes.json')))
    eq_sources = json.load(open(os.path.join(BASE, 'eq_sources.json')))

    return nodes, edges, eq_sources

def add_len_to_edges():

    nodes, edges, eq_sources = load_data()

    # Add edge lengths (if missing)
    if 'length_km' not in list(edges.values())[0]:
        lengths = graphs.compute_edge_lengths(nodes, edges)
        for eid, length in lengths.items():
            edges[eid]['length_km'] = length

    # Add distance from source to edge *center* (if missing)
    if 'dist_s1_km' not in list(edges.values())[0]:
        dists = compute_source_edge_center_distances(nodes, edges, eq_sources)
        for eid, dist_dict in dists.items():
            for sid, dist in dist_dict.items():
                edges[eid][f'dist_{sid}_km'] = dist

    # Save updated edges
    with open(os.path.join(BASE, 'edges.json'), 'w') as f:
        json.dump(edges, f, indent=4)

def plot_system(save_path="system.png"):
    """
    Plots the network defined by nodes.json and edges.json,
    together with point sources in eq_sources.json,
    and saves the figure as system.png.
    """

    # Load data
    with open(os.path.join(BASE, 'nodes.json')) as f:
        nodes = json.load(f)
    with open(os.path.join(BASE, 'edges.json')) as f:
        edges = json.load(f)
    with open(os.path.join(BASE, 'eq_sources.json')) as f:
        sources = json.load(f)

    save_path = os.path.join(BASE, save_path)

    fig, ax = plt.subplots(figsize=(7, 7))
    fsz = 16
    ax.tick_params(axis='both', labelsize=fsz)

    # ---- Plot edges ----
    for eid, e in edges.items():
        x1, y1 = nodes[e["from"]]["x"], nodes[e["from"]]["y"]
        x2, y2 = nodes[e["to"]]["x"], nodes[e["to"]]["y"]
        ax.plot([x1, x2], [y1, y2], color="gray", linewidth=2)

    # ---- Plot nodes ----
    for nid, n in nodes.items():
        ax.scatter(n["x"], n["y"], color="black", s=50)
        ax.text(n["x"] + 0.1, n["y"] + 0.1, nid, fontsize=fsz)

    # ---- Plot sources ----
    for sid, s in sources.items():
        ax.scatter(s["x"], s["y"], color="red", s=80, marker="x")
        ax.text(s["x"] + 0.2, s["y"] + 0.2, sid, fontsize=fsz, color="red")

    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=fsz)
    ax.set_ylabel("Y", fontsize=fsz)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Save
    fig.savefig(save_path, dpi=150)
    print(f"Saved system plot to {os.path.abspath(save_path)}")

if __name__ == "__main__":
    add_len_to_edges()
    #plot_system()