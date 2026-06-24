from __future__ import annotations
import torch, copy

from tbnpy.protocols import validate_prob_objects


# Functions
def get_ancestor_order(probs, query_nodes):
    """
    Compute all ancestor nodes of the given query nodes and return them in
    a valid topological order (i.e., parents appear before their children).

    Supports probability objects that own multiple child variables: a node N
    owning childs [X, Y] is treated as a single graph vertex, and any other
    node listing X (or Y) as a parent variable induces an edge N → that node.
    When multiple parent variables of a node trace back to the same owning
    node, the edge is counted exactly once.

    Args:
        probs (dict):
            A dictionary mapping node names (strings) to probability objects.
            Each probability object must define:
                - childs: a list of child variable objects (each has `.name`)
                - parents: a list of parent variable objects (each has `.name`)
            Every variable must appear as a child of exactly one probability
            object.

        query_nodes (list[str] or set[str]):
            Node names (keys in `probs`) whose marginal distributions we want
            to infer.

    Returns:
        list[str]:
            A topologically sorted list of all ancestor node names of
            query_nodes, including the query nodes themselves. The order
            ensures that if N1 is a parent of N2 (i.e., a child variable of
            N1 appears in N2.parents), then N1 appears before N2.
    """

    # 1. Validate inputs
    assert isinstance(probs, dict), "`probs` must be a dictionary."

    assert isinstance(query_nodes, (list, set)), \
        "`query_nodes` must be a list or a set of node names."

    assert all(isinstance(q, str) for q in query_nodes), \
        "`query_nodes` must contain only strings."

    missing = [q for q in query_nodes if q not in probs]
    assert len(missing) == 0, \
        f"Query nodes not found in `probs`: {missing}"

    for name, obj in probs.items():
        assert hasattr(obj, "childs"), f"Node '{name}' must have attribute `childs`."
        assert hasattr(obj, "parents"), f"Node '{name}' must have attribute `parents`."
        assert isinstance(obj.parents, list), f"`parents` of node '{name}' must be a list."

        for child in obj.childs:
            assert hasattr(child, "name"), \
                f"Child of '{name}' does not have attribute `.name`."

        for parent in obj.parents:
            assert hasattr(parent, "name"), \
                f"Parent of '{name}' does not have attribute `.name`."

    # 2. Build variable -> owning-node map (handles multi-child nodes)
    var_to_node = {}
    for node_name, prob in probs.items():
        for child_var in prob.childs:
            vname = child_var.name
            assert vname not in var_to_node, (
                f"Variable '{vname}' appears as child of more than one probability object "
                f"(in '{var_to_node[vname]}' and '{node_name}')."
            )
            var_to_node[vname] = node_name

    # 3. DFS: identify ancestor NODES (not variables)
    visited = set(query_nodes)
    stack = list(query_nodes)

    while stack:
        node = stack.pop()

        for var in probs[node].parents:
            pname = var.name
            assert pname in var_to_node, (
                f"Parent variable '{pname}' of node '{node}' does not appear as a child "
                f"of any probability object."
            )
            parent_node = var_to_node[pname]
            if parent_node not in visited:
                visited.add(parent_node)
                stack.append(parent_node)

    ancestors = visited

    # 4. Build adjacency lists at the NODE level (deduplicated edges)
    children_of = {n: set() for n in ancestors}
    indegree = {n: 0 for n in ancestors}

    for child_node in ancestors:
        parent_nodes = set()
        for parent_var in probs[child_node].parents:
            parent_node = var_to_node[parent_var.name]
            # Skip self-loops; we don't model a node as its own parent.
            if parent_node == child_node:
                continue
            if parent_node in ancestors:
                parent_nodes.add(parent_node)

        for parent_node in parent_nodes:
            children_of[parent_node].add(child_node)
        indegree[child_node] = len(parent_nodes)

    # 5. Topological sort (Kahn's algorithm)
    ordering = []
    queue = [n for n in ancestors if indegree[n] == 0]

    assert len(queue) > 0, \
        "No root ancestor found—this typically indicates a cycle in the model."

    while queue:
        node = queue.pop(0)
        ordering.append(node)

        for child in children_of[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    assert len(ordering) == len(ancestors), \
        "Topological sorting failed; a cycle may exist in the ancestor subgraph."

    return ordering

def _validate_batch_size(batch_size):
    assert isinstance(batch_size, int), "`batch_size` must be an integer."
    assert batch_size > 0, "`batch_size` must be positive."


def sample(probs, query_nodes, n_sample, batch_size=50_000):
    """
    Forward-sample all ancestors of query_nodes, returning samples for
    query_nodes only. Intermediate ancestor samples are discarded after
    each batch to reduce peak memory usage.

    Returns:
        dict: {node_name: updated probability object} for query nodes only.
              Each object has:
                  - .Cs : tensor (n_sample, n_childs + n_parents)
                  - .ps : tensor (n_sample,)
    """

    # --- Validate input ----------------------------------------------------
    assert isinstance(probs, dict)
    assert isinstance(query_nodes, (list, set))
    assert all(q in probs for q in query_nodes)
    assert isinstance(n_sample, int), "`n_sample` must be an integer."
    assert n_sample > 0, "`n_sample` must be positive."
    _validate_batch_size(batch_size)
    validate_prob_objects(probs)

    ordered_nodes = get_ancestor_order(probs, query_nodes)
    query_set = set(query_nodes)

    probs_copy = {node: copy.deepcopy(probs[node]) for node in ordered_nodes}

    # Accumulators for query nodes only
    query_Cs_batches = {node: [] for node in query_set}
    query_ps_batches = {node: [] for node in query_set}

    # --- Forward sampling: batch-outer, node-inner -------------------------
    for start in range(0, n_sample, batch_size):
        end = min(start + batch_size, n_sample)
        n_batch = end - start

        # Temporary per-batch storage: var_name -> (n_batch,)
        batch_samples = {}

        for node in ordered_nodes:
            prob = probs_copy[node]
            parents = prob.parents

            if len(parents) == 0:
                Cs_batch, ps_batch = prob.sample(n_sample=n_batch)
            else:
                Cs_list = []
                for parent in parents:
                    pname = parent.name
                    assert pname in batch_samples, (
                        f"Parent variable '{pname}' does not appear as a child "
                        f"of any probability object."
                    )
                    Cs_list.append(batch_samples[pname])

                Cs_par = torch.stack(Cs_list, dim=1)
                Cs_batch, ps_batch = prob.sample(Cs_pars=Cs_par)

            # Store each child variable's samples for use by descendants
            for j, child_var in enumerate(prob.childs):
                if Cs_batch.ndim == 1:
                    batch_samples[child_var.name] = Cs_batch
                else:
                    batch_samples[child_var.name] = Cs_batch[:, j]

            if node in query_set:
                query_Cs_batches[node].append(Cs_batch)
                query_ps_batches[node].append(ps_batch)

        # batch_samples discarded at end of each batch

    # --- Assemble results for query nodes only -----------------------------
    result = {}
    for node in query_set:
        prob = probs_copy[node]
        prob.Cs = torch.cat(query_Cs_batches[node], dim=0)
        prob.ps = torch.cat(query_ps_batches[node], dim=0)
        result[node] = prob

    return result

def sample_evidence_v0(probs, query_nodes, n_sample, evidence_df):
    """
    Forward-sample all ancestors of `query_nodes` under multiple evidence rows.

    Parameters
    ----------
    probs : dict
        Mapping from node name -> probability object.
    query_nodes : list or set
        Variables of interest.
    n_sample : int
        Number of samples to generate for each evidence row.
    evidence_df : pd.DataFrame
        Evidence rows. Each column must match a variable name.
        Shape = (n_evi, n_vars_with_evidence)

    Returns
    -------
    dict :
        {node_name : updated probability object}
        Each object contains:
            - object.Cs : (n_evi, n_sample, n_childs + n_parents)
            - object.ps : (n_evi, n_sample)
    """

    # --- Validate ----------------------------------------------------------
    assert isinstance(probs, dict)
    assert isinstance(query_nodes, (list, set))
    assert all(q in probs for q in query_nodes)
    assert hasattr(evidence_df, "columns"), "evidence_df must be a pandas DataFrame"

    n_evi = len(evidence_df)

    # --- Build ancestor order ---------------------------------------------
    ordered_nodes = get_ancestor_order(probs, query_nodes)

    # --- Deep copy relevant nodes -----------------------------------------
    probs_copy = {node: copy.deepcopy(probs[node]) for node in ordered_nodes}

    # --- Build var_to_source lookup ---------------------------------------
    var_to_source = {}
    for node_name, prob in probs_copy.items():
        for j, child_var in enumerate(prob.childs):
            vname = child_var.name
            assert vname not in var_to_source, (
                f"Variable '{vname}' appears as child of more than one probability object."
            )
            var_to_source[vname] = (node_name, j)

    # --- Preallocate per-variable storage of samples -----------------------
    # For each variable name, store:
    #   samples[var] = tensor (n_evi, n_sample)
    samples = {}

    # --- Forward sampling ---------------------------------------------------
    for node in ordered_nodes:

        prob = probs_copy[node]
        parents = prob.parents
        n_childs = len(prob.childs)

        # ------------------------------------------------------------
        # Case 0 — Node is observed in evidence_df
        # ------------------------------------------------------------
        if node in evidence_df.columns:

            ev_vals = torch.tensor(evidence_df[node].values, dtype=torch.float32)
            child_vals = ev_vals.unsqueeze(1).expand(n_evi, n_sample)

            # ---------------------------
            # Build parent values
            # ---------------------------
            Cs_par_3d = []
            for parent in parents:
                pname = parent.name

                if pname in evidence_df.columns:
                    col = torch.tensor(evidence_df[pname].values, dtype=torch.float32)
                    col = col.unsqueeze(1).expand(n_evi, n_sample)
                else:
                    col = samples[pname]

                Cs_par_3d.append(col.unsqueeze(2))

            # ---------------------------
            # Case 0a — Evidence node with NO parents
            # ---------------------------
            if len(Cs_par_3d) == 0:
                Cs = child_vals.unsqueeze(2)     # (n_evi, n_sample, n_childs)
                Cs_flat = Cs.reshape(-1, 1) # (n_evi * n_sample, 1)

            # ---------------------------
            # Case 0b — Evidence node WITH parents
            # ---------------------------
            else:
                Cs_pars = torch.cat(Cs_par_3d, dim=2)
                Cs = torch.cat([child_vals.unsqueeze(2), Cs_pars], dim=2)
                Cs_flat = Cs.reshape(-1, Cs.shape[2])

            # ---------------------------
            # Compute log probability
            # ---------------------------
            logp_flat = prob.log_prob(Cs_flat)
            ps_log = logp_flat.reshape(n_evi, n_sample)

            # ---------------------------
            # Store in probability object
            # ---------------------------
            prob.Cs = Cs
            prob.ps = ps_log
            samples[node] = child_vals

            continue

        # ------------------------------------------------------------
        # Case 1: ROOT nodes → repeat sampling for each evidence row
        # ------------------------------------------------------------
        elif len(parents) == 0:

            # total number of samples
            total_samples = n_evi * n_sample

            # sample all at once
            Cs_flat, ps_flat = prob.sample(n_sample=total_samples)   # (total, n_childs), (total,)

            # reshape into evidence × samples
            Cs = Cs_flat.reshape(n_evi, n_sample, -1) # (n_evi, n_sample, n_childs)
            ps = ps_flat.reshape(n_evi, n_sample) # (n_evi, n_sample)

        # ------------------------------------------------------------
        # Case 2: NON-ROOT nodes → need parent samples
        # ------------------------------------------------------------
        else:

            # Build Cs_par: (n_evi, n_sample, n_parents)
            Cs_par_3d = []
            for parent in parents:
                pname = parent.name

                # If parent appears in evidence_df → override
                if pname in evidence_df.columns:
                    # evidence is static over samples
                    col = torch.tensor(evidence_df[pname].values, dtype=torch.float32)
                    col = col.unsqueeze(1).expand(n_evi, n_sample)  # (n_evi, n_sample)

                else:
                    # otherwise pull sampled values from earlier nodes
                    assert pname in samples, f"Parent '{pname}' should have been sampled already."
                    col = samples[pname]  # (n_evi, n_sample)

                Cs_par_3d.append(col.unsqueeze(2))  # add parent dimension

            # stack parents → (n_evi, n_sample, n_parents)
            Cs_pars = torch.cat(Cs_par_3d, dim=2)

            # ------------------------------------------------------------
            # Vectorized sampling with evidence-aligned parent samples
            # ------------------------------------------------------------
            Cs, ps = prob.sample_evidence(Cs_pars)

        # ------------------------------------------------------------
        # Store Cs, ps in probability object
        # And store child samples for descendant nodes
        # ------------------------------------------------------------
        prob.Cs = Cs    # (n_evi, n_sample, n_childs + n_parents) OR (n_evi, n_sample, n_childs)
        prob.ps = ps    # (n_evi, n_sample)

        # For each child variable, store its sample column 0
        for j, child_var in enumerate(prob.childs):
            samples[child_var.name] = Cs[:, :, j]   # (n_evi, n_sample)

    # -------------------------------------------------------------------------
    return {node: probs_copy[node] for node in ordered_nodes}


def sample_evidence(probs, query_nodes, n_sample, evidence_df, batch_size=50_000):
    """
    Forward-sample all ancestors of `query_nodes` under multiple evidence rows,
    returning samples for query_nodes only. Intermediate ancestor samples are
    discarded after each batch to reduce peak memory usage.

    Returns
    -------
    dict :
        {node_name : probability object} for query nodes only.
        Each object contains:
            - object.Cs : (n_evi, n_sample, n_childs + n_parents)
            - object.ps : (n_evi, n_sample)
    """

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    assert isinstance(probs, dict)
    assert isinstance(query_nodes, (list, set))
    assert hasattr(evidence_df, "columns")
    assert isinstance(n_sample, int), "`n_sample` must be an integer."
    assert n_sample > 0, "`n_sample` must be positive."
    _validate_batch_size(batch_size)
    validate_prob_objects(probs)

    n_evi = len(evidence_df)
    ordered_nodes = get_ancestor_order(probs, query_nodes)
    query_set = set(query_nodes)

    probs_copy = {node: copy.deepcopy(probs[node]) for node in ordered_nodes}

    # Accumulators for query nodes only
    query_Cs_batches = {node: [] for node in query_set}
    query_ps_batches = {node: [] for node in query_set}

    # --------------------------------------------------
    # Forward sampling: batch-outer, node-inner
    # --------------------------------------------------
    for start in range(0, n_sample, batch_size):
        end = min(start + batch_size, n_sample)
        n_batch = end - start

        # Temporary per-batch storage: var_name -> (n_evi, n_batch)
        batch_samples = {}

        for node in ordered_nodes:

            prob = probs_copy[node]
            parents = prob.parents
            n_childs = len(prob.childs)
            n_parents = len(parents)

            if node in evidence_df.columns:
                ev = torch.tensor(evidence_df[node].values, device=prob.device)
                child_vals = ev.unsqueeze(1).expand(n_evi, n_batch)

                Cs_par_cols = []
                for parent in parents:
                    pname = parent.name
                    if pname in evidence_df.columns:
                        col = torch.tensor(evidence_df[pname].values, device=prob.device)
                        col = col.unsqueeze(1).expand(n_evi, n_batch)
                    else:
                        col = batch_samples[pname]
                    Cs_par_cols.append(col.unsqueeze(2))

                if Cs_par_cols:
                    Cs_par_3d = torch.cat(Cs_par_cols, dim=2)
                    Cs_batch = torch.cat([child_vals.unsqueeze(2), Cs_par_3d], dim=2)
                else:
                    Cs_batch = child_vals.unsqueeze(2)

                Cs_flat = Cs_batch.reshape(-1, Cs_batch.shape[2])
                logp_flat = prob.log_prob(Cs_flat)
                ps_batch = logp_flat.reshape(n_evi, n_batch)

                batch_samples[node] = child_vals

            elif n_parents == 0:
                total = n_evi * n_batch
                Cs_child_flat, ps_flat = prob.sample(n_sample=total)

                Cs_batch = Cs_child_flat.reshape(n_evi, n_batch, n_childs)
                ps_batch = ps_flat.reshape(n_evi, n_batch)

                for j, child_var in enumerate(prob.childs):
                    batch_samples[child_var.name] = Cs_batch[:, :, j]

            else:
                Cs_par_cols = []
                for parent in parents:
                    pname = parent.name
                    if pname in evidence_df.columns:
                        col = torch.tensor(evidence_df[pname].values, device=prob.device)
                        col = col.unsqueeze(1).expand(n_evi, n_batch)
                    else:
                        col = batch_samples[pname]
                    Cs_par_cols.append(col.unsqueeze(2))

                Cs_par_3d = torch.cat(Cs_par_cols, dim=2)
                Cs_par_flat = Cs_par_3d.reshape(-1, n_parents)

                Cs_child_flat, ps_flat = prob.sample(Cs_pars=Cs_par_flat)

                Cs_child = Cs_child_flat.reshape(n_evi, n_batch, n_childs)
                ps_batch = ps_flat.reshape(n_evi, n_batch)
                Cs_batch = torch.cat([Cs_child, Cs_par_3d], dim=2)

                for j, child_var in enumerate(prob.childs):
                    batch_samples[child_var.name] = Cs_batch[:, :, j]

            if node in query_set:
                query_Cs_batches[node].append(Cs_batch)
                query_ps_batches[node].append(ps_batch)

        # batch_samples discarded at end of each batch

    # --------------------------------------------------
    # Assemble results for query nodes only
    # --------------------------------------------------
    result = {}
    for node in query_set:
        prob = probs_copy[node]
        prob.Cs = torch.cat(query_Cs_batches[node], dim=1)
        prob.ps = torch.cat(query_ps_batches[node], dim=1)
        result[node] = prob

    return result
