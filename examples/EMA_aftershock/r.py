"""
Radial distance ``R_t`` from the mainshock epicentre to aftershock ``t``.

Active slot (``t <= K``):
    f(r | k) = (1 - n) / (r_max^{1-n} - r_min^{1-n}) * r^{-n}
        on ``[r_min, r_max]``.

Inactive slot (``t > K``): ``R_t = 0`` deterministically.

Inverse-CDF sampling:
    r = (r_min^{1-n} + u * (r_max^{1-n} - r_min^{1-n})) ** (1 / (1-n))
"""

import math

import torch


def _radial_sample(low, high, n, device):
    u = torch.rand_like(low)
    one_minus_n = 1.0 - n
    base = low ** one_minus_n + u * (high ** one_minus_n - low ** one_minus_n)
    return base ** (1.0 / one_minus_n)


def _radial_logpdf(r, low, high, n):
    in_range = (r >= low) & (r <= high)
    one_minus_n = 1.0 - n
    log_norm = math.log(abs(one_minus_n)) - torch.log(
        torch.abs(high ** one_minus_n - low ** one_minus_n)
    )
    val = log_norm - n * torch.log(r.clamp_min(1e-30))
    return torch.where(in_range, val, torch.full_like(r, -float("inf")))


class R:
    """Radial distance at slot ``slot_idx``. Parents = [K]."""

    def __init__(self, childs, parents, slot_idx,
                 r_min=1.0, r_max=50.0, n=1.35, device="cpu"):
        assert len(childs) == 1
        assert len(parents) == 1, "R parents = [K]"
        assert n != 1.0, "exponent n=1 produces an undefined radial law"

        self.childs = childs
        self.parents = parents
        self.device = device
        self.slot_idx = int(slot_idx)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.n = float(n)

    def _active(self, K_idx):
        return self.slot_idx <= K_idx

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        K = Cs_pars[:, 0].long()
        active = self._active(K)

        low = torch.full((Cs_pars.shape[0],), self.r_min, device=self.device)
        high = torch.full_like(low, self.r_max)
        Cs_active = _radial_sample(low, high, self.n, self.device)

        Cs = torch.where(active, Cs_active, torch.zeros_like(low))

        logp_active = _radial_logpdf(Cs, low, high, self.n)
        logp = torch.where(active, logp_active, torch.zeros_like(low))
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        r = Cs[:, 0]
        K = Cs[:, 1].long()
        active = self._active(K)

        low = torch.full_like(r, self.r_min)
        high = torch.full_like(r, self.r_max)
        logp_active = _radial_logpdf(r, low, high, self.n)

        delta_valid = (r == 0)
        logp_inactive = torch.where(
            delta_valid, torch.zeros_like(r), torch.full_like(r, -float("inf"))
        )
        return torch.where(active, logp_active, logp_inactive)
