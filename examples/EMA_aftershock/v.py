"""
Direction angle ``V_t`` of aftershock ``t`` relative to the mainshock epicentre.

Active slot (``t <= K``): ``V_t ~ Uniform[0, 2 pi)``.
Inactive slot (``t > K``): ``V_t = 0`` deterministically.
"""

import math

import torch


class V:
    """Direction at slot ``slot_idx``. Parents = [K]."""

    def __init__(self, childs, parents, slot_idx, device="cpu"):
        assert len(childs) == 1
        assert len(parents) == 1, "V parents = [K]"
        self.childs = childs
        self.parents = parents
        self.device = device
        self.slot_idx = int(slot_idx)

    def _active(self, K_idx):
        return self.slot_idx <= K_idx

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        K = Cs_pars[:, 0].long()
        active = self._active(K)

        u = torch.rand(Cs_pars.shape[0], device=self.device) * (2.0 * math.pi)
        Cs = torch.where(active, u, torch.zeros_like(u))

        log_unif = -math.log(2.0 * math.pi)
        logp = torch.where(
            active,
            torch.full_like(u, log_unif),
            torch.zeros_like(u),
        )
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        v = Cs[:, 0]
        K = Cs[:, 1].long()
        active = self._active(K)

        in_range = (v >= 0) & (v < 2.0 * math.pi)
        log_unif = -math.log(2.0 * math.pi)
        logp_active = torch.where(
            in_range, torch.full_like(v, log_unif), torch.full_like(v, -float("inf"))
        )
        delta_valid = (v == 0)
        logp_inactive = torch.where(
            delta_valid, torch.zeros_like(v), torch.full_like(v, -float("inf"))
        )
        return torch.where(active, logp_active, logp_inactive)
