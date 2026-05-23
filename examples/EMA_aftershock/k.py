"""
Number of aftershocks ``K`` conditional on the mainshock magnitude ``M_0``.

Modified Omori law: the instantaneous rate is
    lambda(t) = K_0 / (t + c) ** p
with ``K_0 = 10 ** (a + b * (m_0 - m_min))``. Integrated over
``[0, T]`` this gives the Poisson mean

    Lambda(T) = K_0 / (1 - p) * ((T + c) ** (1 - p) - c ** (1 - p))

``K`` is then truncated Poisson on ``{0, 1, ..., K_max}``.
"""

import math

import torch


def _lambda_mean(m0, T, c, p, a, b, m_min):
    K0 = 10.0 ** (a + b * (m0 - m_min))
    factor = ((T + c) ** (1.0 - p) - c ** (1.0 - p)) / (1.0 - p)
    return K0 * factor


def _trunc_poisson_logpmf(k, lam, K_max):
    """Log PMF of a Poisson truncated to ``{0,...,K_max}``.

    All inputs are tensors; ``k`` is integer-valued."""
    k_f = k.to(torch.float32)
    log_pmf = k_f * torch.log(lam.clamp_min(1e-30)) - lam - torch.lgamma(k_f + 1.0)

    # Normalising constant: sum_{j=0}^{K_max} pmf(j; lam)
    j = torch.arange(0, K_max + 1, device=lam.device, dtype=torch.float32)
    log_j = j.unsqueeze(0) * torch.log(lam.unsqueeze(1).clamp_min(1e-30)) \
            - lam.unsqueeze(1) - torch.lgamma(j + 1.0).unsqueeze(0)
    log_norm = torch.logsumexp(log_j, dim=1)

    in_range = (k >= 0) & (k <= K_max)
    return torch.where(in_range, log_pmf - log_norm, torch.full_like(log_pmf, -float("inf")))


class K:
    """Truncated Poisson over ``{0,...,K_max}`` with rate from modified Omori."""

    def __init__(self, childs, parents, K_max=20,
                 T=7.0, c=0.05, p=1.08, a=-1.67, b=0.91, m_min=6.0,
                 device="cpu"):
        assert len(childs) == 1
        assert len(parents) == 1, "K parents = [M_0]"
        assert p != 1.0, "modified Omori law diverges at p == 1"

        self.childs = childs
        self.parents = parents
        self.device = device
        self.K_max = int(K_max)
        self.T = float(T)
        self.c = float(c)
        self.p = float(p)
        self.a = float(a)
        self.b = float(b)
        self.m_min = float(m_min)

    def _mean(self, m0):
        return _lambda_mean(m0, self.T, self.c, self.p, self.a, self.b, self.m_min)

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        m0 = Cs_pars[:, 0]
        lam = self._mean(m0)

        # Build a (n, K_max+1) PMF table and sample categorically
        j = torch.arange(0, self.K_max + 1, device=self.device, dtype=torch.float32)
        log_pmf = (j.unsqueeze(0) * torch.log(lam.unsqueeze(1).clamp_min(1e-30))
                   - lam.unsqueeze(1)
                   - torch.lgamma(j + 1.0).unsqueeze(0))
        pmf = torch.softmax(log_pmf, dim=1)

        Cs = torch.multinomial(pmf, num_samples=1).squeeze(1)  # (n,)
        logp = _trunc_poisson_logpmf(Cs, lam, self.K_max)
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        k = Cs[:, 0].long()
        m0 = Cs[:, 1]
        lam = self._mean(m0)
        return _trunc_poisson_logpmf(k, lam, self.K_max)
