"""A custom node owning several childs, for tests of multi-child spec nodes."""

import torch
from torch.distributions import Normal

from tbnpy.probability import Probability


class JointNormal(Probability):
    """X_i ~ Normal(loc_i, scale), independent, one per child: a root owning every X_i."""

    def __init__(self, childs, parents, loc, scale=1.0, device="cpu"):
        """
        childs : [Variable X_1, ..., Variable X_n]
        loc    : n numbers, the mean of each child
        scale  : the standard deviation of every child
        """
        super().__init__(childs, parents, device)
        assert not parents and len(loc) == len(childs)
        self.loc = torch.tensor(loc, dtype=torch.float32)
        self.scale = float(scale)

    def form(self):
        return f"independent normals, loc={self.loc.tolist()}, scale={self.scale:g}"

    def sample(self, Cs_pars=None, n_sample=None):
        dist = Normal(self.loc.expand(n_sample, -1), self.scale)
        Cs = dist.sample()
        return Cs, dist.log_prob(Cs).sum(1)

    def log_prob(self, Cs):
        x = Cs[:, :len(self.childs)]
        return Normal(self.loc.expand(len(x), -1), self.scale).log_prob(x).sum(1)
