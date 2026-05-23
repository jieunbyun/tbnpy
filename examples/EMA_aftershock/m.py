"""
Earthquake magnitude variables.

* ``M0`` — mainshock magnitude. Truncated exponential on
  ``[m_min, m_max]`` with rate ``beta``. No parents.

* ``Mt`` — aftershock magnitude at slot ``t``. Conditional on
  ``(M_0, K)``. Truncated exponential on ``[m_min, m_0 - delta_m]``
  while the slot is "active" (``t <= K``), otherwise a delta at zero.

Inverse-CDF sampling is used so the resulting log probabilities are
analytic (and finite). Outside the truncation interval ``log_prob`` is
``-inf``.
"""

import math

import torch


def _trunc_exp_sample(low, high, beta, device):
    """Sample from a truncated exponential with rate ``beta`` on
    ``[low, high]`` (element-wise tensors).

    Inverse-CDF: ``x = low - log(1 - u (1 - exp(-beta (high - low)))) / beta``.
    """
    u = torch.rand_like(low)
    z = 1.0 - torch.exp(-beta * (high - low))
    # Guard against degenerate intervals where high == low
    x = torch.where(z > 0, low - torch.log1p(-u * z) / beta, low)
    return x


def _trunc_exp_logpdf(x, low, high, beta):
    """Log density of a truncated exponential on ``[low, high]``."""
    in_range = (x >= low) & (x <= high) & (high > low)
    norm = 1.0 - torch.exp(-beta * (high - low))
    # log f(x) = log(beta) - beta (x - low) - log(norm)
    val = math.log(beta) - beta * (x - low) - torch.log(norm.clamp_min(1e-30))
    return torch.where(in_range, val, torch.full_like(x, -float("inf")))


class M0:
    """Mainshock magnitude: truncated exponential."""

    def __init__(self, childs, m_min=6.0, m_max=8.5, beta=0.76, device="cpu"):
        assert len(childs) == 1
        self.childs = childs
        self.parents = []
        self.device = device
        self.m_min = float(m_min)
        self.m_max = float(m_max)
        self.beta = float(beta)

    def sample(self, Cs_pars=None, n_sample=None):
        if n_sample is None:
            assert Cs_pars is not None
            n_sample = Cs_pars.shape[0]

        low = torch.full((n_sample,), self.m_min, device=self.device)
        high = torch.full((n_sample,), self.m_max, device=self.device)
        Cs = _trunc_exp_sample(low, high, self.beta, self.device)
        logp = _trunc_exp_logpdf(Cs, low, high, self.beta)
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        x = Cs[:, 0]
        low = torch.full_like(x, self.m_min)
        high = torch.full_like(x, self.m_max)
        return _trunc_exp_logpdf(x, low, high, self.beta)


class Mt:
    """Aftershock magnitude at slot ``slot_idx``. Parents = [M_0, K]."""

    def __init__(self, childs, parents, slot_idx,
                 m_min=4.5, beta=0.76, delta_m=1.2, device="cpu"):
        assert len(childs) == 1
        assert len(parents) == 2, "Mt parents = [M_0, K]"
        self.childs = childs
        self.parents = parents
        self.device = device
        self.slot_idx = int(slot_idx)  # 1-indexed
        self.m_min = float(m_min)
        self.beta = float(beta)
        self.delta_m = float(delta_m)

    def _active(self, K_idx):
        return self.slot_idx <= K_idx

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        m0 = Cs_pars[:, 0]
        K = Cs_pars[:, 1].long()
        m_high = m0 - self.delta_m
        m_low = torch.full_like(m0, self.m_min)

        active = self._active(K)
        # When inactive or interval invalid → delta at zero
        bad = ~active | (m_high <= m_low)
        m_low_safe = torch.where(bad, torch.zeros_like(m0), m_low)
        m_high_safe = torch.where(bad, torch.zeros_like(m0), m_high)

        Cs_active = _trunc_exp_sample(m_low_safe, m_high_safe, self.beta, self.device)
        Cs = torch.where(bad, torch.zeros_like(m0), Cs_active)

        logp_active = _trunc_exp_logpdf(Cs, m_low, m_high, self.beta)
        logp = torch.where(bad, torch.zeros_like(m0), logp_active)
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        x = Cs[:, 0]
        m0 = Cs[:, 1]
        K = Cs[:, 2].long()
        m_high = m0 - self.delta_m
        m_low = torch.full_like(x, self.m_min)

        active = self._active(K)
        # Inactive slot: valid iff x == 0 (delta), logp = 0
        delta_valid = (x == 0)
        logp_inactive = torch.where(
            delta_valid, torch.zeros_like(x), torch.full_like(x, -float("inf"))
        )
        logp_active = _trunc_exp_logpdf(x, m_low, m_high, self.beta)
        return torch.where(active, logp_active, logp_inactive)
