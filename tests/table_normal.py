"""A custom node that reads prestored data from a file its kwargs name, for tests of `directory`."""

import json
from pathlib import Path

import torch
from torch.distributions import Normal

from tbnpy.probability import Probability


class TableNormal(Probability):
    """X ~ Normal(loc, scale), with loc and scale read from a JSON file."""

    def __init__(self, childs, parents, table, directory=None, device="cpu"):
        """
        childs    : [Variable X]
        table     : the JSON file holding {"loc": ..., "scale": ...}, relative to directory
        directory : the folder the spec's file names are relative to
        """
        super().__init__(childs, parents, device)
        self.directory = directory
        values = json.loads((Path(directory) / table).read_text(encoding="utf-8"))
        self.loc, self.scale = float(values["loc"]), float(values["scale"])

    def sample(self, Cs_pars=None, n_sample=None):
        dist = Normal(torch.full((n_sample,), self.loc), self.scale)
        Cs = dist.sample()
        return Cs, dist.log_prob(Cs)

    def log_prob(self, Cs):
        return Normal(self.loc, self.scale).log_prob(Cs[:, 0])
