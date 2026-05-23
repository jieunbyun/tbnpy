"""
Epicentre/aftershock location variables.

The BN uses two location nodes per time slice:
    L_x  (x-coordinate of epicentre / aftershock location)
    L_y  (y-coordinate of epicentre / aftershock location)

Two probability classes are provided:

* ``L0``  — mainshock epicentre. Uniform over a rectangular region
  ``[x_min, x_max] x [y_min, y_max]``. No parents.

* ``Lt``  — aftershock location. Deterministic:
       L_x_t = L_x_0 + R_t * cos(V_t)
       L_y_t = L_y_0 + R_t * sin(V_t)
  Parents are passed in the order ``[L_x_0, L_y_0, R_t, V_t]``.

Both classes carry two child variables so a single CPT object covers the
2-D location.
"""

import torch


class L0:
    """Uniform mainshock epicentre over a rectangular region."""

    def __init__(self, childs, region=(-22.632, 98.072, -15.088, 90.528), device="cpu"):
        assert len(childs) == 2, "L0 has two children: [L_x_0, L_y_0]"
        assert len(region) == 4, "region = (x_min, x_max, y_min, y_max)"

        self.childs = childs
        self.parents = []
        self.device = device
        self.region = tuple(float(r) for r in region)

    def sample(self, Cs_pars=None, n_sample=None):
        if n_sample is None:
            assert Cs_pars is not None
            n_sample = Cs_pars.shape[0]

        x_min, x_max, y_min, y_max = self.region
        u = torch.rand(n_sample, 2, device=self.device)
        Cs = torch.empty(n_sample, 2, device=self.device)
        Cs[:, 0] = x_min + (x_max - x_min) * u[:, 0]
        Cs[:, 1] = y_min + (y_max - y_min) * u[:, 1]

        area = (x_max - x_min) * (y_max - y_min)
        logp = torch.full((n_sample,), -float(torch.log(torch.tensor(area))),
                          device=self.device)
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        x, y = Cs[:, 0], Cs[:, 1]
        x_min, x_max, y_min, y_max = self.region

        in_region = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        area = (x_max - x_min) * (y_max - y_min)
        logp = torch.where(
            in_region,
            torch.full_like(x, -float(torch.log(torch.tensor(area)))),
            torch.full_like(x, -float("inf")),
        )
        return logp


class Lt:
    """Deterministic aftershock location given (L_x_0, L_y_0, R_t, V_t)."""

    def __init__(self, childs, parents, device="cpu"):
        assert len(childs) == 2, "Lt has two children: [L_x_t, L_y_t]"
        assert len(parents) == 4, "Lt parents = [L_x_0, L_y_0, R_t, V_t]"

        self.childs = childs
        self.parents = parents
        self.device = device

    def _compute(self, Cs_pars):
        Cs_pars = Cs_pars.to(self.device)
        lx0 = Cs_pars[:, 0]
        ly0 = Cs_pars[:, 1]
        r = Cs_pars[:, 2]
        v = Cs_pars[:, 3]
        return lx0 + r * torch.cos(v), ly0 + r * torch.sin(v)

    def sample(self, Cs_pars):
        lx, ly = self._compute(Cs_pars)
        Cs = torch.stack([lx, ly], dim=1)
        logp = torch.zeros(Cs.shape[0], device=self.device)
        return Cs, logp

    def log_prob(self, Cs):
        Cs = Cs.to(self.device)
        lx_obs = Cs[:, 0]
        ly_obs = Cs[:, 1]
        lx_exp, ly_exp = self._compute(Cs[:, 2:])

        valid = (lx_obs == lx_exp) & (ly_obs == ly_exp)
        return torch.where(
            valid,
            torch.zeros_like(lx_obs),
            torch.full_like(lx_obs, -float("inf")),
        )
