"""
Park-Ang damage index at edge ``n`` after shock ``i``.

Two variants per the Ghosh-2015 demand model:

* ``D0`` — mainshock damage: log D | A ~ Normal(alpha + beta * ln A, eps)
  with ``(alpha, beta, eps) = (-1.91, 2.51, 0.70)``. Parents = [A_{0,n}].

* ``Dt`` — post-aftershock damage:
  log D | A_t, D_{t-1} ~ Normal(alpha' + beta' ln A_t + gamma' ln D_{t-1}
                                + delta' ln A_t ln D_{t-1}, eps')
  with ``(alpha', beta', gamma', delta', eps') =
  (-1.65, 0.71, 0.19, -0.33, 0.68)``.
  Parents = [A_{t,n}, D_{t-1,n}].

If the shock is inactive (``A_t == 0``) the damage index carries over
unchanged, i.e. ``D_t = D_{t-1}``.
"""

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

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        A = Cs_pars[:, 0]
        D_prev = Cs_pars[:, 1]
        active = A > 0

        mu = self._mu(A, D_prev)
        sigma = torch.full_like(mu, self.eps)
        dist = Normal(mu, sigma)
        log_d = dist.sample()
        Cs_active = torch.exp(log_d)

        Cs = torch.where(active, Cs_active, D_prev)
        logp_active = dist.log_prob(log_d) - log_d
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
        dist = Normal(mu, sigma)
        logp_active = dist.log_prob(log_d) - log_d
        logp_active = torch.where(d_val > 0, logp_active, torch.full_like(d_val, -float("inf")))

        carry_valid = (d_val == D_prev)
        logp_inactive = torch.where(
            carry_valid, torch.zeros_like(d_val), torch.full_like(d_val, -float("inf"))
        )
        return torch.where(active, logp_active, logp_inactive)
