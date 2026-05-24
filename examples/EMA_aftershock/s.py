"""
System state ``S_i`` derived from the operational states ``X_{i,*}``.

Multi-state classification driven by precomputed RSR reference tensors.
The logic mirrors ``rsr.get_comp_cond_sys_prob_multi``:

    for s in 1..max_st:
        classify each still-undecided sample against
        (upper_refs[s], lower_refs[s]);
        lower => S = s - 1, finalised
        upper => could be S >= s, carry to next level
        unknown => carry, will be retried at next level or via s_fun

The class accepts an optional ``s_fun(comps_dict) -> (_, sys_st, _)``
callable to resolve any samples that remain unknown after the last
level. If ``s_fun`` is ``None``, unknown samples are assigned to
``unknown_state`` (default ``-1``).

System states are 0-indexed: valid values are ``{0, 1, ..., max_st}``.
``unknown_state = -1`` is therefore deliberately outside this range, so
unresolved samples can never be mistaken for a real classification (a
common failure mode if one defaulted to ``0`` and forgot to pass
``s_fun``).
"""

from typing import Callable, Dict, Optional, Sequence

import torch

try:
    from rsr.rsr import classify_samples_with_indices  # type: ignore[import]
except ImportError:
    # Optional dependency — install with: pip install -e <path-to-rsr-repo>
    classify_samples_with_indices = None


class S:
    """Discrete system state. Parents = [X_{i,1}, ..., X_{i,N}]."""

    def __init__(
        self,
        childs,
        parents,
        refs_dict_upper: Dict[int, torch.Tensor],
        refs_dict_lower: Dict[int, torch.Tensor],
        row_names: Sequence[str],
        s_fun: Optional[Callable[[Dict[str, int]], tuple]] = None,
        unknown_state: int = -1,
        device: str = "cpu",
    ):
        assert len(childs) == 1
        assert classify_samples_with_indices is not None, (
            "rsr package is not importable; make sure the rsr repo is on sys.path"
        )

        keys_u = set(refs_dict_upper.keys())
        keys_l = set(refs_dict_lower.keys())
        assert keys_u == keys_l, "upper/lower ref dicts must share keys"
        sys_states = sorted(keys_u)
        assert sys_states == list(range(1, max(sys_states) + 1)), (
            "ref dict keys must be 1, 2, ..., max_st"
        )

        n_parents = len(parents)
        assert n_parents == len(row_names), (
            f"len(parents)={n_parents} must equal len(row_names)={len(row_names)}"
        )

        self.childs = childs
        self.parents = parents
        self.device = device
        self.refs_upper = {s: refs_dict_upper[s].to(device) for s in sys_states}
        self.refs_lower = {s: refs_dict_lower[s].to(device) for s in sys_states}
        self.row_names = list(row_names)
        self.s_fun = s_fun
        self.unknown_state = int(unknown_state)
        self.max_st = max(sys_states)

    def _to_onehot(self, X):
        """Convert (n, N) binary state matrix to (n, N, 2) one-hot."""
        X = X.to(self.device).long()
        n_state = 2
        return torch.nn.functional.one_hot(X, num_classes=n_state).to(torch.int32)

    def _classify(self, X):
        """Run the multi-state classification, returning S of shape (n,)."""
        n = X.shape[0]
        samples = self._to_onehot(X)  # (n, N, 2)
        S_out = torch.full((n,), -1, dtype=torch.long, device=self.device)

        active = torch.ones(n, dtype=torch.bool, device=self.device)
        upper_prev = torch.ones(n, dtype=torch.bool, device=self.device)

        for s in range(1, self.max_st + 1):
            if not active.any():
                break
            res = classify_samples_with_indices(
                samples[active], self.refs_upper[s], self.refs_lower[s],
                return_masks=True,
            )
            active_idx = torch.where(active)[0]
            mask_upper = torch.zeros(n, dtype=torch.bool, device=self.device)
            mask_lower = torch.zeros(n, dtype=torch.bool, device=self.device)
            mask_upper[active_idx] = res["mask_upper"]
            mask_lower[active_idx] = res["mask_lower"]

            # samples classed as lower at level s are S = s - 1, but only if
            # they were upper at every previous level (i.e. they made it to this level)
            finalise = mask_lower & upper_prev
            S_out[finalise] = s - 1

            active = active & ~finalise
            upper_prev = mask_upper

        # remaining upper samples are S = max_st
        last_upper = upper_prev & active
        S_out[last_upper] = self.max_st
        active = active & ~last_upper

        # truly unknown samples
        unknown_idx = torch.where(active)[0]
        if unknown_idx.numel() > 0:
            if self.s_fun is not None:
                for j in unknown_idx.tolist():
                    comps = {self.row_names[k]: int(X[j, k].item())
                             for k in range(len(self.row_names))}
                    _, sys_st, _ = self.s_fun(comps)
                    S_out[j] = int(sys_st)
            else:
                S_out[unknown_idx] = self.unknown_state

        return S_out

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        X = Cs_pars  # (n, N)
        S_vals = self._classify(X)
        logp = torch.zeros(S_vals.shape[0], device=self.device)
        return S_vals, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        s_val = Cs[:, 0].long()
        X = Cs[:, 1:]
        expected = self._classify(X)
        valid = (s_val == expected)
        return torch.where(
            valid,
            torch.zeros(Cs.shape[0], device=self.device),
            torch.full((Cs.shape[0],), -float("inf"), device=self.device),
        )
