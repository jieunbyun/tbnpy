from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional, Set, Any

from tbnpy import variable, cpt
import time

import torch

# -----------------------------
# Utility: detect variable type
# -----------------------------
def is_discrete(var) -> bool:
    return isinstance(var.values, list)

def is_continuous(var) -> bool:
    return isinstance(var.values, tuple)

def num_categories(var) -> int:
    assert is_discrete(var)
    return len(var.values)

# ---------------------------------------------------------
# Adaptive, tensorised MH sampler for hybrid BNs
# ---------------------------------------------------------
# Factor adapter: build Cs for a probability object (factor)
def build_Cs_3d(prob, state: Dict[str, torch.Tensor], evidence_1d: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Build Cs with shape (n_evi, n_chain, n_childs + n_parents) in the column order:
        [child_0, ..., child_{n_childs-1}, parent_0, ..., parent_{n_parents-1}]
    Evidence variables are broadcast to (n_evi, n_chain).
    Latent variables come from `state[var_name]` which must be (n_evi, n_chain).

    Input:
    prob : probability object
        Must have `prob.childs` and `prob.parents` lists of variable objects.
    state: dict
        state["X"] has shape (n_evi, n_chain)
    evidence_1d: dict
        evidence_1d["E"] has shape (n_evi,)
    """
    # Infer (n_evi, n_chain) from any state tensor, else from evidence.
    n_evi = None
    n_chain = None
    if len(state) > 0:
        any_t = next(iter(state.values()))
        n_evi, n_chain = any_t.shape
    else:
        # fall back to evidence
        any_e = next(iter(evidence_1d.values()))
        n_evi = any_e.shape[0]
        raise ValueError("State is empty; cannot infer n_chain. Provide initial state for latent variables.")

    cols: List[torch.Tensor] = []

    # Child columns
    for v in prob.childs:
        name = v.name
        if name in evidence_1d:
            col = evidence_1d[name].unsqueeze(1).expand(n_evi, n_chain)
        else:
            col = state[name]
        cols.append(col)

    # Parent columns
    for v in prob.parents:
        name = v.name
        if name in evidence_1d:
            col = evidence_1d[name].unsqueeze(1).expand(n_evi, n_chain)
        else:
            col = state[name]
        cols.append(col)

    Cs = torch.stack(cols, dim=2)  # (n_evi, n_chain, n_childs+n_parents)
    return Cs


def factor_logp_2d(prob, Cs_3d: torch.Tensor) -> torch.Tensor:
    """
    Evaluate prob.log_prob on Cs_3d (n_evi, n_chain, dim) by flattening.
    Returns (n_evi, n_chain).
    """
    n_evi, n_chain, dim = Cs_3d.shape
    Cs_flat = Cs_3d.reshape(n_evi * n_chain, dim)
    logp_flat = prob.log_prob(Cs_flat)
    return logp_flat.reshape(n_evi, n_chain)


# -----------------------------
# Build factor adjacency
# -----------------------------
def build_factors_by_var(probs: Dict[str, Any]) -> Dict[str, List[Any]]:
    """
    Map variable name -> list of probability objects (factors) that depend on it
    (as either child or parent).
    """
    factors_by_var: Dict[str, List[Any]] = {}
    for _, prob in probs.items():
        for v in list(prob.childs) + list(prob.parents):
            factors_by_var.setdefault(v.name, []).append(prob)
    return factors_by_var


# -----------------------------
# Proposals (symmetric)
# -----------------------------
def propose_discrete_adaptive(
    x: torch.Tensor,
    logits: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Adaptive symmetric proposal for a discrete variable.

    Parameters
    ----------
    x : torch.Tensor
        Current state, shape (n_evi, n_chain), integer in [0, K-1]
    logits : torch.Tensor
        Global adaptive logits, shape (K,)
    alpha : float
        Mixing weight for local vs global proposal (0 < alpha < 1)

    Returns
    -------
    x_new : torch.Tensor
        Proposed state, same shape as x
    """
    device = x.device
    n_evi, n_chain = x.shape
    K = logits.numel()

    # ---- Local symmetric proposal: uniform over other states ----
    r = torch.randint(0, K - 1, size=(n_evi, n_chain), device=device)
    x_local = r + (r >= x).to(r.dtype)

    # ---- Global adaptive proposal (independent of x) ----
    probs = torch.softmax(logits, dim=0)
    x_global = torch.multinomial(
        probs,
        num_samples=n_evi * n_chain,
        replacement=True,
    ).reshape(n_evi, n_chain)

    # ---- Mixture (still symmetric) ----
    use_local = torch.rand((n_evi, n_chain), device=device) < alpha
    x_new = torch.where(use_local, x_local, x_global)

    # ---- Ensure x_new != x (important for MH efficiency) ----
    same = x_new == x
    if same.any():
        r = torch.randint(0, K - 1, size=(same.sum(),), device=device)
        x_new[same] = r + (r >= x[same]).to(r.dtype)

    return x_new


def propose_continuous_rw_gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Symmetric Gaussian random-walk proposal.
    x: (n_evi, n_chain) float tensor
    """
    return x + torch.randn_like(x) * sigma


# -----------------------------
# Adaptive, Block MH Sampler
# -----------------------------
@dataclass
class AdaptConfig:
    burnin: int = 200
    gamma: float = 0.2 # Robbins–Monro exponent in (0.5, 1]
    target_accept: float = 0.234 # target acceptable rate for block updates
    min_log_sigma: float = -12.0      # clamp for stability
    max_log_sigma: float = 6.0
    alpha: float = 0.5 # for discrete vars, pi(new) = alpha * pi_local(new | old) + (1-alpha) * pi_global(new)
        # pi_local(new | old): uniform leap between states (for symmetry)
        # pi_global(new): distribution of previous sampled states


class HybridAdaptiveMH:
    """
    Tensorised Metropolis–Hastings for hybrid (discrete + continuous) BN,
    with many evidence rows and many chains.

    - Latent variable values are stored in `state[var_name]` with shape (n_evi, n_chain).
    - Evidence is stored in `evidence_1d[var_name]` with shape (n_evi,).
    - Each MH step updates ONE variable at a time (easy and robust).
      (You can extend to blocks later.)
    """

    def __init__(
        self,
        probs: Dict[str, Any],
        variables: Dict[str, Any] | Sequence[Any],            # list of Variable objects (all variables)
        evidence_df,                         # pandas DataFrame: shape (n_evi, ...)
        n_chain: int,
        device: str | torch.device = "cpu",
        adapt: AdaptConfig = AdaptConfig()
    ):
        self.probs = probs
        if isinstance(variables, dict):
            variables = list(variables.values())
        else:
            self.variables = list(variables)
        self.device = torch.device(device)
        self.adapt = adapt

        # Evidence as 1D tensors (n_evi,)
        self.n_evi = len(evidence_df)
        self.n_chain = n_chain

        self.evidence_1d: Dict[str, torch.Tensor] = {}
        for col in evidence_df.columns:
            # keep dtype flexible: discrete evidence should still be numeric indices in df if used
            t = torch.tensor(evidence_df[col].values, device=self.device)
            if t.dtype not in (torch.int64, torch.int32, torch.float32, torch.float64):
                t = t.to(torch.float32)
            self.evidence_1d[col] = t

        # Determine which vars are evidence vs latent
        self.evidence_vars: Set[str] = set(evidence_df.columns)
        self.latent_vars: List[Any] = [v for v in self.variables if v.name not in self.evidence_vars]

        # Adjacency: var_name -> factors
        self.factors_by_var = build_factors_by_var(self.probs)

        # Proposal params for continuous latent vars
        self.log_sigma: Dict[str, torch.Tensor] = {}
        for v in self.latent_vars:
            if is_continuous(v):
                self.log_sigma[v.name] = torch.tensor(0.0, device=self.device)  # sigma=1 initially

        # State placeholders
        self.state: Dict[str, torch.Tensor] = {} # var_name -> (n_evi, n_chain) tensor

        # Cache: per-factor logp (n_evi, n_chain) for current state (optional but useful)
        self.factor_logp_cache: Dict[int, torch.Tensor] = {}  # key by id(prob)

        # Cache: total logp per evidence×chain and aggregated per chain
        self.logp_evi_chain: Optional[torch.Tensor] = None
        self.logp_chain: Optional[torch.Tensor] = None

        # Proposal params for discrete latent vars (global adaptive logits)
        self.discrete_logits: Dict[str, torch.Tensor] = {}
        for v in self.latent_vars:
            if is_discrete(v):
                K = num_categories(v)
                self.discrete_logits[v.name] = torch.zeros(
                    K, device=self.device, dtype=torch.float32
                )

    # Helper: get affected factors for a block of variables
    def _affected_factors_for_block(self, var_names):
        affected = []
        seen = set()

        for name in var_names:
            for prob in self.factors_by_var.get(name, []):
                pid = id(prob)
                if pid not in seen:
                    affected.append(prob)
                    seen.add(pid)
        return affected

    # Initialisation
    def init_state_from_forward_samples(self, probs_copy: Dict[str, Any]):
        """
        Initialise latent state from a forward-sampling result compatible with your sample_evidence(),
        i.e. each prob in probs_copy has prob.Cs of shape (n_evi, n_sample, ...)
        and prob.ps of shape (n_evi, n_sample).
        We treat n_sample as n_chain here.
        """
        # Build samples[var_name] from prob.Cs columns (childs first).
        samples: Dict[str, torch.Tensor] = {}
        for prob in probs_copy.values():
            Cs = prob.Cs  # (n_evi, n_chain, dim)
            for j, child_var in enumerate(prob.childs):
                samples[child_var.name] = Cs[:, :, j]

        # Keep only latent vars (evidence vars are fixed)
        for v in self.latent_vars:
            if v.name not in samples:
                raise KeyError(f"Latent variable '{v.name}' not found in forward samples. "
                               f"Make sure ancestors include it and it is sampled.")
            x = samples[v.name].to(self.device)

            # dtype rules
            if is_discrete(v):
                x = x.to(torch.long)
            else:
                x = x.to(torch.float32)
            self.state[v.name] = x

        # Compute caches
        self._recompute_all_logps()

    def init_state_random(self, seed: Optional[int] = None):
        """
        Fallback initializer if you don't want to rely on forward sampling.
        Discrete: uniform over categories
        Continuous: standard normal
        """
        if seed is not None:
            torch.manual_seed(seed)

        for v in self.latent_vars:
            if is_discrete(v):
                K = num_categories(v)
                x = torch.randint(0, K, (self.n_evi, self.n_chain), device=self.device, dtype=torch.long)
            else:
                x = torch.randn((self.n_evi, self.n_chain), device=self.device, dtype=torch.float32)
            self.state[v.name] = x

        self._recompute_all_logps()

    # Log-probability bookkeeping
    def _recompute_all_logps(self):
        """
        Compute and cache:
        - per-factor logp (n_evi, n_chain)
        - total logp_evi_chain (n_evi, n_chain)
        - total logp_chain (n_chain,) = sum over evidence rows
        """
        logp_evi_chain = torch.zeros((self.n_evi, self.n_chain), device=self.device, dtype=torch.float32)
        self.factor_logp_cache.clear()

        for prob in self.probs.values():
            Cs = build_Cs_3d(prob, self.state, self.evidence_1d)
            logp2d = factor_logp_2d(prob, Cs).to(torch.float32)
            self.factor_logp_cache[id(prob)] = logp2d
            logp_evi_chain += logp2d

        self.logp_evi_chain = logp_evi_chain
        self.logp_chain = logp_evi_chain.sum(dim=0)  # (n_chain,)

    # One MH update for one variable
    @torch.no_grad()
    def mh_update_block(self, vars: List[variable.Variable], iteration: int) -> torch.Tensor:
        """
        Metropolis–Hastings update for a block of variables.
        vars: list[Variable]
        Returns accept mask: (n_chain,)
        """
        var_names = [v.name for v in vars]

        # 1. Propose block
        proposed = self._propose_block(vars) # proposed: Dict[str, torch.Tensor]; proposed[v.name].shape == (n_evi, n_chain)

        # 2. Find affected factors
        affected = self._affected_factors_for_block(var_names) # List[ProbabilityObject]

        delta_evi_chain = torch.zeros(
            (self.n_evi, self.n_chain),
            device=self.device,
            dtype=torch.float32,
        )

        # 3. Temporarily swap in proposed values
        old_values = {}
        for name in var_names:
            old_values[name] = self.state[name]
            self.state[name] = proposed[name]

        # 4. Compute Δ log-probability
        for prob in affected:
            logp_old_2d = self.factor_logp_cache[id(prob)] # self.factor_logp_cache: Dict[int, torch.Tensor]; 
            # self.factor_logp_cache[id(prob)].shape == (n_evi, n_chain)
            Cs_new = build_Cs_3d(prob, self.state, self.evidence_1d)
            logp_new_2d = factor_logp_2d(prob, Cs_new)
            delta_evi_chain += (logp_new_2d - logp_old_2d)

        # 5. Restore old values before accept/reject
        for name, x_old in old_values.items():
            self.state[name] = x_old

        # 6. Aggregate over evidence
        log_alpha = delta_evi_chain.sum(dim=0)  # (n_chain,)

        # 7. Accept / reject
        u = torch.log(torch.rand((self.n_chain,), device=self.device))
        accept = u < log_alpha

        # 8. Apply accepted updates and refresh caches
        if accept.any():
            for name in var_names:
                self.state[name][:, accept] = proposed[name][:, accept]

            for prob in affected:
                logp_old_2d = self.factor_logp_cache[id(prob)]
                Cs_upd = build_Cs_3d(prob, self.state, self.evidence_1d)
                logp_upd_2d = factor_logp_2d(prob, Cs_upd)
                self.factor_logp_cache[id(prob)] = logp_upd_2d
                self.logp_evi_chain += (logp_upd_2d - logp_old_2d)

            self.logp_chain = self.logp_evi_chain.sum(dim=0)

        # 9. Adaptation (continuous + discrete vars in block)
        if iteration < self.adapt.burnin:
            acc_rate = accept.float().mean().item()
            eta = (iteration + 1.0) ** (-self.adapt.gamma)

            for v in vars:
                name = v.name

                # ---- Continuous adaptation (unchanged) ----
                if is_continuous(v):
                    target = self.adapt.target_accept
                    new_log_sigma = self.log_sigma[name] + eta * (acc_rate - target)
                    self.log_sigma[name] = new_log_sigma.clamp(
                        self.adapt.min_log_sigma,
                        self.adapt.max_log_sigma,
                    )

                # ---- Discrete adaptation (NEW) ----
                elif is_discrete(v):
                    K = num_categories(v)

                    # accepted samples for this variable
                    x_acc = self.state[name][:, accept]   # (n_evi, n_accept)

                    if x_acc.numel() > 0:
                        # empirical frequencies
                        hist = torch.bincount(
                            x_acc.flatten(),
                            minlength=K
                        ).float()

                        hist = hist / hist.sum().clamp_min(1.0)

                        # Robbins–Monro update of global logits
                        pi = torch.softmax(self.discrete_logits[name], dim=0)
                        self.discrete_logits[name] += eta * (hist - pi)

        return accept # accept : torch.BoolTensor of shape (n_chain,)
    
    def _build_update_schedule(self, update_blocks):
        if update_blocks is None:
            return [[v] for v in self.latent_vars]

        name_to_var = {v.name: v for v in self.latent_vars}

        blocks = []
        used = set()

        for block_names in update_blocks:
            block_vars = [name_to_var[n] for n in block_names]
            blocks.append(block_vars)
            used.update(block_names)

        # leftover variables → singletons
        for v in self.latent_vars:
            if v.name not in used:
                blocks.append([v])

        return blocks
    
    # Helper: block proposal
    def _propose_block(self, vars):
        """
        Propose new values for a block of variables.
        Returns dict {var_name: x_new_tensor}
        """
        proposed = {}
        for v in vars:
            name = v.name
            x_old = self.state[name]

            if is_discrete(v):
                K = num_categories(v)
                x_new = propose_discrete_adaptive(x_old, self.discrete_logits[name], self.adapt.alpha)
            else:
                sigma = float(torch.exp(self.log_sigma[name]).clamp_min(1e-12).item())
                x_new = propose_continuous_rw_gaussian(x_old, sigma)

            proposed[name] = x_new

        return proposed

    # Run
    @torch.no_grad()
    def run(
        self,
        n_iter: int,
        update_blocks: Optional[List[List[str]]] = None,
        store_every: int = 0,
        progress_every: int = 100,
    ) -> Dict[str, Any]:
        """
        Run MCMC with optional block updates.

        Parameters
        ----------
        n_iter : int
            Number of MCMC iterations.
        update_blocks : list[list[str]] or None
            Variable blocks to update together.
            - None → update all latent variables one by one
            - Each inner list is a block of variable names
            - Variables not listed are updated individually
        store_every : int
            If >0, stores thinned samples every `store_every` iterations.
            Warning: storing full (n_evi, n_chain) states can be very large.

        Returns
        -------
        dict with:
            - 'accept_rate': dict[block_name -> float]
            - 'logp_chain': final (n_chain,)
            - 'logp_evi_chain': final (n_evi, n_chain)
            - 'states_thinned' (optional)
            - 'log_sigma' (continuous proposal scales)
        """
        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------
        if self.logp_chain is None or self.logp_evi_chain is None:
            raise RuntimeError(
                "State not initialised. Call init_state_from_forward_samples() "
                "or init_state_random()."
            )

        # --------------------------------------------------
        # Build update schedule (blocks)
        # --------------------------------------------------
        name_to_var = {v.name: v for v in self.latent_vars}

        if update_blocks is None:
            # default: singleton blocks
            blocks = [[v] for v in self.latent_vars]
        else:
            blocks = [] # List[variable.Variable]
            used = set()

            for block_names in update_blocks:
                block_vars = []
                for nm in block_names:
                    if nm not in name_to_var:
                        print(
                            f"[HybridAdaptiveMH] Skipping '{nm}' in update_blocks: "
                            "not a latent variable."
                        )
                        continue
                    block_vars.append(name_to_var[nm])
                blocks.append(block_vars)
                used.update(block_names)

            # add leftover latent variables as singleton blocks
            for v in self.latent_vars:
                if v.name not in used:
                    blocks.append([v])

        # --------------------------------------------------
        # Bookkeeping
        # --------------------------------------------------
        def block_name(block):
            if len(block) == 1:
                return block[0].name
            return "(" + ",".join(v.name for v in block) + ")"

        block_names = [block_name(b) for b in blocks] # List[str]

        accept_counts = {bn: 0.0 for bn in block_names}
        total_updates = {bn: 0.0 for bn in block_names}

        states_thinned = [] if (store_every and store_every > 0) else None

        # --------------------------------------------------
        # Main MCMC loop
        # --------------------------------------------------
        total_time = 0.0
        st = time.time()
        for it in range(n_iter):
            for block, bn in zip(blocks, block_names):
                accept = self.mh_update_block(block, iteration=it) # accept.shape == (n_chain,)

                accept_rate_step = float(accept.float().mean().item())
                accept_counts[bn] += accept_rate_step
                total_updates[bn] += 1.0

            if states_thinned is not None and ((it + 1) % store_every == 0):
                # WARNING: this can be huge
                snap = {k: v.detach().clone() for k, v in self.state.items()}
                states_thinned.append(snap)

            # ---- progress ----
            if progress_every > 0 and (it + 1) % progress_every == 0:
                et = time.time()
                time1 = et - st
                total_time += time1
                st = et

                # running acceptance rates
                acc_str = ", ".join(
                    f"{bn}={accept_counts[bn] / max(total_updates[bn], 1):.2f}"
                    for bn in block_names
                )

                print(
                    f"[MCMC] {it+1}/{n_iter} iterations completed "
                    f"({n_iter - it - 1} remaining) "
                    f"{time1:.2f} sec taken, total {total_time:.2f} sec\n"
                    f"        accept: {acc_str}"
                )
        # --------------------------------------------------
        # Outputs
        # --------------------------------------------------
        accept_rate = {
            bn: accept_counts[bn] / max(total_updates[bn], 1.0)
            for bn in block_names
        }

        log_sigma_out = {
            k: float(v.detach().cpu().item())
            for k, v in self.log_sigma.items()
        }

        out = {
            "accept_rate": accept_rate,
            "logp_chain": self.logp_chain.detach().clone(),
            "logp_evi_chain": self.logp_evi_chain.detach().clone(),
            "log_sigma": log_sigma_out,
        }

        if states_thinned is not None:
            out["states_thinned"] = states_thinned

        return out