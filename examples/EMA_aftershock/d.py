"""
Park-Ang damage index at edge ``n`` after shock ``i``.

Two variants per the Ghosh-2015 demand model:

* ``D0`` — mainshock damage: log D | A ~ Normal(alpha + beta * ln A, eps)
  with ``(alpha, beta, eps) = (-1.91, 2.51, 0.70)``. Parents = [A_{0,n}].

* ``Dt`` — post-aftershock damage:
  log D | A_t, D_{t-1} ~ TruncatedNormal(alpha' + beta' ln A_t + gamma' ln D_{t-1}
                                          + delta' ln A_t ln D_{t-1}, eps',
                                          lower = ln D_{t-1})
  with ``(alpha', beta', gamma', delta', eps') =
  (-1.65, 0.71, 0.19, -0.33, 0.68)``.
  The truncation at ``ln D_{t-1}`` enforces ``D_t >= D_{t-1}`` (damage cannot
  decrease). Parents = [A_{t,n}, D_{t-1,n}].

If the shock is inactive (``A_t == 0``) the damage index carries over
unchanged, i.e. ``D_t = D_{t-1}``.
"""

import math

import torch
from torch.distributions import Normal


class D0:
    """Mainshock damage. Parents = [A_{0,n}]."""

    def __init__(self, childs, parents,
                 alpha=-1.91, beta=2.51, eps=0.70, device="cpu"):
        assert len(childs) == 1
        assert len(parents) == 1, "D0 parents = [A_{0,n}]"

        self.childs = childs
        self.parents = parents
        self.device = device
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def _mu(self, A):
        return self.alpha + self.beta * torch.log(A.clamp_min(1e-30))

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        A = Cs_pars[:, 0]

        mu = self._mu(A)
        sigma = torch.full_like(mu, self.eps)
        dist = Normal(mu, sigma)
        log_d = dist.sample()
        Cs = torch.exp(log_d)
        logp = dist.log_prob(log_d) - log_d
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        d_val = Cs[:, 0]
        A = Cs[:, 1]
        log_d = torch.log(d_val.clamp_min(1e-30))
        mu = self._mu(A)
        sigma = torch.full_like(mu, self.eps)
        dist = Normal(mu, sigma)
        logp = dist.log_prob(log_d) - log_d
        return torch.where(d_val > 0, logp, torch.full_like(d_val, -float("inf")))


class Dt:
    """Post-aftershock damage. Parents = [A_{t,n}, D_{t-1,n}]."""

    def __init__(self, childs, parents,
                 alpha=-1.65, beta=0.71, gamma=0.19, delta=-0.33, eps=0.68,
                 device="cpu"):
        assert len(childs) == 1
        assert len(parents) == 2, "Dt parents = [A_{t,n}, D_{t-1,n}]"

        self.childs = childs
        self.parents = parents
        self.device = device
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        self.eps = float(eps)

    def _mu(self, A, D_prev):
        ln_a = torch.log(A.clamp_min(1e-30))
        ln_d = torch.log(D_prev.clamp_min(1e-30))
        return (self.alpha
                + self.beta * ln_a
                + self.gamma * ln_d
                + self.delta * ln_a * ln_d)

    def _log_survival(self, alpha):
        # log(1 - Phi(alpha)) = log(0.5 * erfc(alpha / sqrt(2))), numerically stable
        return torch.log(0.5 * torch.erfc(alpha / math.sqrt(2)))

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        A = Cs_pars[:, 0]
        D_prev = Cs_pars[:, 1]
        active = A > 0

        mu = self._mu(A, D_prev)
        sigma = torch.full_like(mu, self.eps)

        # Truncated normal in log-space: log D_t ~ TN(mu, sigma, lower=log D_{t-1})
        # Guarantees D_t >= D_{t-1} for every sample path.
        ln_d_prev = torch.log(D_prev.clamp_min(1e-30))
        alpha = (ln_d_prev - mu) / sigma          # standardised lower bound
        _std = Normal(0.0, 1.0)
        p_lo = _std.cdf(alpha).clamp(max=1.0 - 1e-6)
        u = (torch.rand_like(mu) * (1.0 - p_lo) + p_lo).clamp(1e-7, 1.0 - 1e-7)
        log_d = (mu + sigma * _std.icdf(u)).clamp(min=ln_d_prev)
        Cs_active = torch.exp(log_d)

        Cs = torch.where(active, Cs_active, D_prev)

        dist = Normal(mu, sigma)
        logp_active = dist.log_prob(log_d) - log_d - self._log_survival(alpha)
        logp = torch.where(active, logp_active, torch.zeros_like(logp_active))
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        d_val = Cs[:, 0]
        A = Cs[:, 1]
        D_prev = Cs[:, 2]
        active = A > 0

        log_d = torch.log(d_val.clamp_min(1e-30))
        mu = self._mu(A, D_prev)
        sigma = torch.full_like(mu, self.eps)
        ln_d_prev = torch.log(D_prev.clamp_min(1e-30))
        alpha = (ln_d_prev - mu) / sigma

        dist = Normal(mu, sigma)
        logp_active = dist.log_prob(log_d) - log_d - self._log_survival(alpha)
        logp_active = torch.where(
            (d_val > 0) & (d_val >= D_prev),
            logp_active,
            torch.full_like(d_val, -float("inf")),
        )

        carry_valid = (d_val == D_prev)
        logp_inactive = torch.where(
            carry_valid, torch.zeros_like(d_val), torch.full_like(d_val, -float("inf"))
        )
        return torch.where(active, logp_active, logp_inactive)
