"""
Peak ground acceleration (PGA) at edge ``n`` from shock ``t``.

Conditional on (M_t, L_x_t, L_y_t), PGA is lognormal:

    log A | M, L ~ Normal(mu(M, r), sigma)

with the Campbell-1997 / Lee-2011 mean

    mu(M, r) = ln 0.55 - 3.512 + 0.904 M
            - 1.328 ln( sqrt(r^2 + (0.149 exp(0.647 M))^2) )

and ``r = || (L_x, L_y) - edge_midpoint ||``. ``sigma = 0.52``.

If the shock is inactive (signalled by ``M == 0`` from the magnitude
node), PGA is set to zero deterministically.
"""

import math

import torch
from torch.distributions import Normal


def _mu(M, r):
    eps = 0.149 * torch.exp(0.647 * M)
    return (
        math.log(0.55)
        - 3.512
        + 0.904 * M
        - 1.328 * torch.log(torch.sqrt(r * r + eps * eps).clamp_min(1e-30))
    )


class A:
    """PGA at one fixed edge. Parents = [M_t, L_x_t, L_y_t]."""

    def __init__(self, childs, parents, edge_midpoint, sigma=0.52, device="cpu"):
        assert len(childs) == 1
        assert len(parents) == 3, "A parents = [M_t, L_x_t, L_y_t]"
        assert len(edge_midpoint) == 2

        self.childs = childs
        self.parents = parents
        self.device = device
        self.sigma = float(sigma)
        self.edge_x = float(edge_midpoint[0])
        self.edge_y = float(edge_midpoint[1])

    def _distance(self, lx, ly):
        return torch.sqrt((lx - self.edge_x) ** 2 + (ly - self.edge_y) ** 2)

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        M = Cs_pars[:, 0]
        lx = Cs_pars[:, 1]
        ly = Cs_pars[:, 2]

        active = M > 0
        r = self._distance(lx, ly)
        mu = _mu(M.clamp_min(1e-6), r)
        sigma = torch.full_like(mu, self.sigma)
        dist = Normal(mu, sigma)
        log_a = dist.sample()
        Cs_active = torch.exp(log_a)

        Cs = torch.where(active, Cs_active, torch.zeros_like(Cs_active))
        # logp of lognormal at the realised value (in log space)
        logp_active = dist.log_prob(log_a) - log_a  # change of variables
        logp = torch.where(active, logp_active, torch.zeros_like(logp_active))
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        a_val = Cs[:, 0]
        M = Cs[:, 1]
        lx = Cs[:, 2]
        ly = Cs[:, 3]
        active = M > 0

        # Active branch: lognormal density
        log_a = torch.log(a_val.clamp_min(1e-30))
        r = self._distance(lx, ly)
        mu = _mu(M.clamp_min(1e-6), r)
        sigma = torch.full_like(mu, self.sigma)
        dist = Normal(mu, sigma)
        logp_active = dist.log_prob(log_a) - log_a
        logp_active = torch.where(a_val > 0, logp_active, torch.full_like(a_val, -float("inf")))

        # Inactive branch: delta at 0
        delta_valid = (a_val == 0)
        logp_inactive = torch.where(
            delta_valid, torch.zeros_like(a_val), torch.full_like(a_val, -float("inf"))
        )
        return torch.where(active, logp_active, logp_inactive)
