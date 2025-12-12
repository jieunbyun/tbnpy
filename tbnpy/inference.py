from __future__ import annotations
import torch, copy


# Functions
def get_ancestor_order(probs, query_nodes):
    """
    Compute all ancestor nodes of the given query nodes and return them in
    a valid topological order (i.e., parents appear before their children).

    Args:
        probs (dict):
            A dictionary mapping node names (strings) to probability objects.
            Each probability object must define:
                - childs: a list containing the child variable(s)
                - parents: a list of parent variable objects (each has `.name`)

        query_nodes (list[str] or set[str]):
            Node names whose marginal distributions we want to infer.

    Returns:
        list[str]:
            A topologically sorted list of all ancestors of query_nodes,
            including the query nodes themselves. The order ensures that
            if X is a parent of Y, then X appears before Y.
    """

    # 1. Validate inputs
    assert isinstance(probs, dict), "`probs` must be a dictionary."

    assert isinstance(query_nodes, (list, set)), \
        "`query_nodes` must be a list or a set of node names."

    # ensure all elements are strings
    assert all(isinstance(q, str) for q in query_nodes), \
        "`query_nodes` must contain only strings."

    # ensure nodes exist in the network
    missing = [q for q in query_nodes if q not in probs]
    assert len(missing) == 0, \
        f"Query nodes not found in `probs`: {missing}"

    # Validate structure of each probability object
    for name, obj in probs.items():
        assert hasattr(obj, "childs"), f"Node '{name}' must have attribute `childs`."
        assert hasattr(obj, "parents"), f"Node '{name}' must have attribute `parents`."
        assert isinstance(obj.parents, list), f"`parents` of node '{name}' must be a list."

        # each parent must have a `.name`
        for parent in obj.parents:
            assert hasattr(parent, "name"), \
                f"Parent of '{name}' does not have attribute `.name`."

    # 2. DFS: Identify ancestor nodes (recursively)
    ancestors = set()
    stack = list(query_nodes)

    while stack:
        node = stack.pop()

        # assert that this node exists in the graph
        assert node in probs, f"Node '{node}' missing from `probs` during traversal."

        parent_vars = probs[node].parents

        for var in parent_vars:
            pname = var.name
            if pname not in ancestors:
                ancestors.add(pname)
                stack.append(pname)

    # include query nodes themselves
    ancestors.update(query_nodes)

    # 3. Build adjacency lists for the induced ancestral subgraph
    children_of = {n: [] for n in ancestors}
    indegree = {n: 0 for n in ancestors}

    for child in ancestors:
        for parent in probs[child].parents:
            pname = parent.name
            if pname in ancestors:
                # parent → child edge
                children_of[pname].append(child)
                indegree[child] += 1

    # 4. Topological sort (Kahn’s algorithm)
    ordering = []

    # nodes with no parents (root ancestors)
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

    # 5. Final consistency checks
    assert len(ordering) == len(ancestors), \
        "Topological sorting failed; a cycle may exist in the ancestor subgraph."

    return ordering

def sample(probs, query_nodes, n_sample):
    """
    Forward-sample all ancestors of query_nodes into a deep-copied probability
    structure, storing for each node X:
        - X.Cs : tensor (n_sample, n_childs + n_parents)
        - X.ps : tensor (n_sample,)   probability or log-probability

    Returns:
        dict: {node_name: updated probability object} in ancestor order.
    """

    # --- Validate input ----------------------------------------------------
    assert isinstance(probs, dict)
    assert isinstance(query_nodes, (list, set))
    assert all(q in probs for q in query_nodes)

    # --- Get ancestor ordering --------------------------------------------
    ordered_nodes = get_ancestor_order(probs, query_nodes)

    # --- Create a deep copy of only the nodes we need ----------------------
    probs_copy = {node: copy.deepcopy(probs[node]) for node in ordered_nodes}

    # --- Build lookup: variable name → (prob_node_name, child_idx) ---------
    #    So we can find where any variable's samples live.
    var_to_source = {}
    for node_name, prob in probs_copy.items():
        for j, child_var in enumerate(prob.childs):
            vname = child_var.name
            assert vname not in var_to_source, (
                f"Variable '{vname}' appears as child of more than one probability object."
            )
            var_to_source[vname] = (node_name, j)

    # --- Forward sampling ---------------------------------------------------
    for node in ordered_nodes:
        prob = probs_copy[node]
        parents = prob.parents

        # Build parent sample matrix
        if len(parents) == 0:
            Cs, ps = prob.sample(n_sample)

        else:
            # Retrieve parent samples from probs_copy
            Cs_list = []
            for parent in parents:
                pname = parent.name

                # Find where this parent variable is generated
                assert pname in var_to_source, (
                    f"Parent variable '{pname}' does not appear as a child "
                    f"of any probability object."
                )

                src_node, child_idx = var_to_source[pname]
                parent_Cs = probs_copy[src_node].Cs     # shape (n_sample, n_childs+...)

                if parent_Cs.ndim == 1:
                        assert child_idx == 0, (
                            f"Variable '{pname}' expected to be the first/only child of node '{src_node}', "
                            f"but child_idx={child_idx}. Cs is 1-dimensional so only child_idx=0 is valid."
                        )
                        parent_samples = parent_Cs  # shape (n_sample,)

                elif parent_Cs.ndim == 2:
                    parent_samples = parent_Cs[:, child_idx]

                else:
                    raise ValueError(
                        f"Unexpected Cs shape {parent_Cs.shape} for node '{src_node}'. "
                        "Cs must be 1D or 2D."
                    )
                
                Cs_list.append(parent_samples)

            Cs_par = torch.stack(Cs_list, dim=1)
            Cs, ps = prob.sample(Cs_par)     # Cs: (N,1+np), ps: (N,)

        # Store inside the copied probability node
        prob.Cs = Cs
        prob.ps = ps

    # --- Return only the relevant updated probability objects -------------
    return {node: probs_copy[node] for node in ordered_nodes}

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


def sample_evidence(probs, query_nodes, n_sample, evidence_df):
    """
    Forward-sample all ancestors of `query_nodes` under multiple evidence rows,
    using ONLY prob.sample() (no prob.sample_evidence).

    Returns
    -------
    dict :
        {node_name : probability object}
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

    n_evi = len(evidence_df)

    # --------------------------------------------------
    # Ancestor ordering
    # --------------------------------------------------
    ordered_nodes = get_ancestor_order(probs, query_nodes)

    # --------------------------------------------------
    # Deep copy relevant nodes
    # --------------------------------------------------
    probs_copy = {node: copy.deepcopy(probs[node]) for node in ordered_nodes}

    # --------------------------------------------------
    # Storage for sampled variables
    # samples[var_name] : (n_evi, n_sample)
    # --------------------------------------------------
    samples = {}

    # --------------------------------------------------
    # Forward sampling
    # --------------------------------------------------
    for node in ordered_nodes:

        prob = probs_copy[node]
        parents = prob.parents
        n_childs = len(prob.childs)
        n_parents = len(parents)

        # ==================================================
        # CASE 0 — Child variable is observed (evidence)
        # ==================================================
        if node in evidence_df.columns:

            # child values: (n_evi, n_sample)
            ev = torch.tensor(evidence_df[node].values, device=prob.device)
            child_vals = ev.unsqueeze(1).expand(n_evi, n_sample)

            # build parent tensor (3D)
            Cs_par_cols = []
            for parent in parents:
                pname = parent.name
                if pname in evidence_df.columns:
                    col = torch.tensor(evidence_df[pname].values, device=prob.device)
                    col = col.unsqueeze(1).expand(n_evi, n_sample)
                else:
                    col = samples[pname]
                Cs_par_cols.append(col.unsqueeze(2))

            # assemble Cs
            if Cs_par_cols:
                Cs_par_3d = torch.cat(Cs_par_cols, dim=2)
                Cs = torch.cat([child_vals.unsqueeze(2), Cs_par_3d], dim=2)
            else:
                Cs = child_vals.unsqueeze(2)

            # log-prob only (no sampling)
            Cs_flat = Cs.reshape(-1, Cs.shape[2])
            logp_flat = prob.log_prob(Cs_flat)
            ps = logp_flat.reshape(n_evi, n_sample)

            prob.Cs = Cs
            prob.ps = ps
            samples[node] = child_vals
            continue

        # ==================================================
        # CASE 1 — Root node (no parents)
        # ==================================================
        if n_parents == 0:

            total = n_evi * n_sample
            Cs_child_flat, ps_flat = prob.sample(n_sample=total)

            Cs_child = Cs_child_flat.reshape(n_evi, n_sample, n_childs)
            ps = ps_flat.reshape(n_evi, n_sample)

            Cs = Cs_child

        # ==================================================
        # CASE 2 — Non-root node (parents exist)
        # ==================================================
        else:

            # build parent tensor (3D)
            Cs_par_cols = []
            for parent in parents:
                pname = parent.name
                if pname in evidence_df.columns:
                    col = torch.tensor(evidence_df[pname].values, device=prob.device)
                    col = col.unsqueeze(1).expand(n_evi, n_sample)
                else:
                    col = samples[pname]
                Cs_par_cols.append(col.unsqueeze(2))

            Cs_par_3d = torch.cat(Cs_par_cols, dim=2)  # (n_evi, n_sample, n_parents)
            Cs_par_flat = Cs_par_3d.reshape(-1, n_parents)

            # sample children conditionally
            Cs_child_flat, ps_flat = prob.sample(Cs_pars=Cs_par_flat)

            Cs_child = Cs_child_flat.reshape(n_evi, n_sample, n_childs)
            ps = ps_flat.reshape(n_evi, n_sample)

            Cs = torch.cat([Cs_child, Cs_par_3d], dim=2)

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------
        prob.Cs = Cs
        prob.ps = ps

        for j, child_var in enumerate(prob.childs):
            samples[child_var.name] = Cs[:, :, j]

    # --------------------------------------------------
    return probs_copy