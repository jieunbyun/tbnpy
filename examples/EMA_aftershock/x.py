"""
Operational state of edge ``n`` after shock ``i``.

Deterministic indicator on the Park-Ang damage threshold:
    X = 1 if D < threshold else 0.
"""

import torch


class X:
    """Operational state. Parents = [D_{i,n}]. Child is binary 0/1."""

    def __init__(self, childs, parents, threshold=0.4, device="cpu"):
        assert len(childs) == 1
        assert len(parents) == 1, "X parents = [D_{i,n}]"

        self.childs = childs
        self.parents = parents
        self.device = device
        self.threshold = float(threshold)

    def _operational(self, D):
        return (D < self.threshold).long()

    def sample(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        D = Cs_pars[:, 0]
        Cs = self._operational(D)
        logp = torch.zeros(Cs.shape[0], device=self.device)
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        x_val = Cs[:, 0].long()
        D = Cs[:, 1]
        expected = self._operational(D)
        valid = (x_val == expected)
        return torch.where(
            valid, torch.zeros_like(D), torch.full_like(D, -float("inf"))
        )
