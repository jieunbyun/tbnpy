"""
Epicentre/aftershock location variables.

The 2-D location ``L_*`` is modelled as two scalar Variables ``L_x_*``
and ``L_y_*`` because the tbnpy inference engine assumes one
probability object per variable (keyed by the child's name). Each
class below has exactly one child.

* ``L0`` — one coordinate of the mainshock epicentre. Uniform on
  ``[low, high]``. No parents. Build two instances (one per axis) with
  the appropriate ``low``/``high`` to recover the rectangular support.

* ``Lt`` — one coordinate of an aftershock location. Deterministic:

      L_x_t = L_x_0 + R_t * cos(V_t)        (axis="x")
      L_y_t = L_y_0 + R_t * sin(V_t)        (axis="y")

  Parents are ``[L_axis_0, R_t, V_t]``.
"""

import math

import torch


class L0:
    """Uniform on ``[low, high]`` — one mainshock-epicentre coordinate."""

    def __init__(self, childs, low, high, device="cpu"):
        assert len(childs) == 1, "L0 has one child"
        assert high >= low, "high must be >= low"

        self.childs = childs
        self.parents = []
        self.device = device
        self.low = float(low)
        self.high = float(high)

    def _log_density(self, n_sample):
        width = self.high - self.low
        if width > 0:
            return torch.full((n_sample,), -math.log(width), device=self.device)
        return torch.zeros(n_sample, device=self.device)

    def sample(self, Cs_pars=None, n_sample=None):
        if n_sample is None:
            assert Cs_pars is not None
            n_sample = Cs_pars.shape[0]
        u = torch.rand(n_sample, device=self.device)
        Cs = self.low + (self.high - self.low) * u
        return Cs, self._log_density(n_sample)

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        x = Cs[:, 0]
        in_range = (x >= self.low) & (x <= self.high)
        width = self.high - self.low
        if width > 0:
            val = torch.full_like(x, -math.log(width))
        else:
            val = torch.zeros_like(x)
        return torch.where(in_range, val, torch.full_like(x, -float("inf")))


class Lt:
    """Deterministic aftershock-location coordinate.

    Parameters
    ----------
    parents : list of Variable
        ``[L_axis_0, R_t, V_t]`` (3 parents).
    axis : str
        Either ``"x"`` (``cos`` projection) or ``"y"`` (``sin``).
    """

    def __init__(self, childs, parents, axis, device="cpu"):
        assert len(childs) == 1, "Lt has one child"
        assert len(parents) == 3, "Lt parents = [L_axis_0, R_t, V_t]"
        assert axis in ("x", "y"), "axis must be 'x' or 'y'"

        self.childs = childs
        self.parents = parents
        self.device = device
        self.axis = axis

    def _compute(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        l0 = Cs_pars[:, 0]
        r = Cs_pars[:, 1]
        v = Cs_pars[:, 2]
        trig = torch.cos(v) if self.axis == "x" else torch.sin(v)
        return l0 + r * trig

    def sample(self, Cs_pars):
        Cs = self._compute(Cs_pars)
        logp = torch.zeros(Cs.shape[0], device=self.device)
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        x_obs = Cs[:, 0]
        x_exp = self._compute(Cs[:, 1:])
        valid = (x_obs == x_exp)
        return torch.where(
            valid,
            torch.zeros_like(x_obs),
            torch.full_like(x_obs, -float("inf")),
        )
