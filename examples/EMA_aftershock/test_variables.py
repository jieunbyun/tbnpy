"""
Unit tests for each variable class in this example.

Run as a script::

    python examples/EMA_aftershock/test_variables.py
"""

import math
import os
import sys
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent.parent
RSR_REPO = REPO_ROOT.parent / "rsr"

sys.path.insert(0, str(BASE))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if RSR_REPO.exists() and str(RSR_REPO) not in sys.path:
    sys.path.insert(0, str(RSR_REPO))

from tbnpy import variable  # noqa: E402

import l as l_mod  # noqa: E402
import m as m_mod  # noqa: E402
import k as k_mod  # noqa: E402
import r as r_mod  # noqa: E402
import v as v_mod  # noqa: E402
import a as a_mod  # noqa: E402
import d as d_mod  # noqa: E402
import x as x_mod  # noqa: E402
import s as s_mod  # noqa: E402

DEVICE = "cpu"
torch.manual_seed(0)


def _new_cont(name):
    return variable.Variable(name=name, values=(-torch.inf, torch.inf))


def _new_disc(name, values):
    return variable.Variable(name=name, values=values)


# ---------------------------------------------------------------------
# L0
# ---------------------------------------------------------------------
def test_L0_sample():
    obj = l_mod.L0(childs=[_new_cont("L_x_0"), _new_cont("L_y_0")],
                   region=(0.0, 10.0, 0.0, 5.0), device=DEVICE)

    # 1) samples fall in region
    Cs, _ = obj.sample(n_sample=5000)
    assert ((Cs[:, 0] >= 0.0) & (Cs[:, 0] <= 10.0)).all()
    assert ((Cs[:, 1] >= 0.0) & (Cs[:, 1] <= 5.0)).all()

    # 2) different region -> different range
    obj2 = l_mod.L0(childs=[_new_cont("x"), _new_cont("y")],
                    region=(100.0, 110.0, 0.0, 1.0), device=DEVICE)
    Cs2, _ = obj2.sample(n_sample=5000)
    assert Cs2[:, 0].min() >= 100.0 and Cs2[:, 0].max() <= 110.0

    # 3) boundary: zero-area region collapses to a point
    obj3 = l_mod.L0(childs=[_new_cont("x"), _new_cont("y")],
                    region=(2.0, 2.0, 3.0, 3.0), device=DEVICE)
    Cs3, _ = obj3.sample(n_sample=10)
    assert torch.allclose(Cs3[:, 0], torch.full((10,), 2.0))
    assert torch.allclose(Cs3[:, 1], torch.full((10,), 3.0))


def test_L0_log_prob():
    obj = l_mod.L0(childs=[_new_cont("x"), _new_cont("y")],
                   region=(0.0, 10.0, 0.0, 5.0), device=DEVICE)

    # 1) inside region: log p = -log(area) = -log(50)
    inside = torch.tensor([[3.0, 2.5], [9.9, 0.1]])
    lp = obj.log_prob(inside)
    assert torch.allclose(lp, torch.full((2,), -math.log(50.0)), atol=1e-5)

    # 2) outside region: -inf
    outside = torch.tensor([[-1.0, 1.0], [3.0, 6.0]])
    lp = obj.log_prob(outside)
    assert torch.isinf(lp).all() and (lp < 0).all()

    # 3) smaller region -> higher log density
    obj2 = l_mod.L0(childs=[_new_cont("x"), _new_cont("y")],
                    region=(0.0, 1.0, 0.0, 1.0), device=DEVICE)
    lp2 = obj2.log_prob(torch.tensor([[0.5, 0.5]]))
    assert lp2.item() > lp[0].item()  # 0.0 > -log(50)

    # 4) boundary: exactly at edge is in-region
    lp_edge = obj.log_prob(torch.tensor([[10.0, 5.0]]))
    assert torch.isfinite(lp_edge).all()


# ---------------------------------------------------------------------
# Lt
# ---------------------------------------------------------------------
def test_Lt_sample():
    obj = l_mod.Lt(
        childs=[_new_cont("L_x_t"), _new_cont("L_y_t")],
        parents=[_new_cont("L_x_0"), _new_cont("L_y_0"),
                 _new_cont("R_t"), _new_cont("V_t")],
        device=DEVICE,
    )

    # 1) zero R -> output equals epicentre
    pars = torch.tensor([[3.0, 4.0, 0.0, 1.234]])
    Cs, lp = obj.sample(pars)
    assert torch.allclose(Cs[0], torch.tensor([3.0, 4.0]))
    assert lp.item() == 0.0

    # 2) V = 0, R > 0 -> shifts along +x
    pars = torch.tensor([[0.0, 0.0, 2.0, 0.0]])
    Cs, _ = obj.sample(pars)
    assert torch.allclose(Cs[0], torch.tensor([2.0, 0.0]), atol=1e-6)

    # 3) V = pi/2 -> shifts along +y
    pars = torch.tensor([[0.0, 0.0, 2.0, math.pi / 2]])
    Cs, _ = obj.sample(pars)
    assert torch.allclose(Cs[0], torch.tensor([0.0, 2.0]), atol=1e-6)


def test_Lt_log_prob():
    obj = l_mod.Lt(
        childs=[_new_cont("L_x_t"), _new_cont("L_y_t")],
        parents=[_new_cont("L_x_0"), _new_cont("L_y_0"),
                 _new_cont("R_t"), _new_cont("V_t")],
        device=DEVICE,
    )

    # 1) consistent assignment -> 0
    Cs = torch.tensor([[3.0, 4.0, 3.0, 4.0, 0.0, 0.0]])
    assert obj.log_prob(Cs).item() == 0.0

    # 2) inconsistent x -> -inf
    Cs = torch.tensor([[5.0, 4.0, 3.0, 4.0, 0.0, 0.0]])
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) consistent shift
    Cs = torch.tensor([[5.0, 0.0, 3.0, 0.0, 2.0, 0.0]])
    assert obj.log_prob(Cs).item() == 0.0


# ---------------------------------------------------------------------
# M0
# ---------------------------------------------------------------------
def test_M0_sample():
    obj = m_mod.M0(childs=[_new_cont("M_0")],
                   m_min=6.0, m_max=8.5, beta=0.76, device=DEVICE)

    # 1) samples within bounds
    Cs, _ = obj.sample(n_sample=5000)
    assert (Cs >= 6.0).all() and (Cs <= 8.5).all()

    # 2) higher beta -> stronger mass near m_min
    obj2 = m_mod.M0(childs=[_new_cont("M_0")],
                    m_min=6.0, m_max=8.5, beta=5.0, device=DEVICE)
    Cs2, _ = obj2.sample(n_sample=10_000)
    assert Cs2.mean().item() < Cs.mean().item()

    # 3) boundary: tight interval gives roughly uniform output
    obj3 = m_mod.M0(childs=[_new_cont("M_0")],
                    m_min=6.0, m_max=6.001, beta=0.76, device=DEVICE)
    Cs3, _ = obj3.sample(n_sample=1000)
    assert ((Cs3 >= 6.0) & (Cs3 <= 6.001)).all()


def test_M0_log_prob():
    obj = m_mod.M0(childs=[_new_cont("M_0")],
                   m_min=6.0, m_max=8.5, beta=0.76, device=DEVICE)

    # 1) in-range: finite
    Cs = torch.tensor([[6.5], [7.0], [8.0]])
    lp = obj.log_prob(Cs)
    assert torch.isfinite(lp).all()

    # 2) monotone decreasing in x (exponential decay)
    assert lp[0].item() > lp[1].item() > lp[2].item()

    # 3) out-of-range -> -inf
    Cs = torch.tensor([[5.0], [9.0]])
    assert torch.isinf(obj.log_prob(Cs)).all()


# ---------------------------------------------------------------------
# Mt
# ---------------------------------------------------------------------
def test_Mt_sample():
    obj = m_mod.Mt(childs=[_new_cont("M_t")],
                   parents=[_new_cont("M_0"), _new_disc("K", list(range(5)))],
                   slot_idx=1, device=DEVICE)

    # 1) inactive slot (K=0, slot=1 > 0) -> 0
    pars = torch.tensor([[7.5, 0.0]])
    Cs, lp = obj.sample(pars)
    assert Cs.item() == 0.0
    assert lp.item() == 0.0

    # 2) active slot: within [m_min, m_0 - delta_m]
    pars = torch.tensor([[7.5, 3.0]])
    Cs, _ = obj.sample(pars)
    assert 4.5 <= Cs.item() <= 7.5 - 1.2

    # 3) boundary: m_0 - delta_m <= m_min -> bad interval, output 0
    pars = torch.tensor([[5.5, 3.0]])  # m_0 - 1.2 = 4.3 < m_min=4.5
    Cs, _ = obj.sample(pars)
    assert Cs.item() == 0.0


def test_Mt_log_prob():
    obj = m_mod.Mt(childs=[_new_cont("M_t")],
                   parents=[_new_cont("M_0"), _new_disc("K", list(range(5)))],
                   slot_idx=1, device=DEVICE)

    # 1) inactive + value 0 -> log p = 0
    Cs = torch.tensor([[0.0, 7.5, 0.0]])
    assert obj.log_prob(Cs).item() == 0.0

    # 2) inactive + non-zero -> -inf
    Cs = torch.tensor([[5.0, 7.5, 0.0]])
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) active + in range -> finite
    Cs = torch.tensor([[5.0, 7.5, 3.0]])
    assert torch.isfinite(obj.log_prob(Cs)).all()

    # 4) active + out of range (above m_0 - delta_m) -> -inf
    Cs = torch.tensor([[7.0, 7.5, 3.0]])  # 7.0 > 7.5 - 1.2 = 6.3
    assert math.isinf(obj.log_prob(Cs).item())


# ---------------------------------------------------------------------
# K
# ---------------------------------------------------------------------
def test_K_sample():
    obj = k_mod.K(childs=[_new_disc("K", list(range(11)))],
                  parents=[_new_cont("M_0")],
                  K_max=10, device=DEVICE)

    # 1) within {0, ..., K_max}
    pars = torch.tensor([[7.0]] * 1000)
    Cs, _ = obj.sample(pars)
    assert (Cs >= 0).all() and (Cs <= 10).all()

    # 2) larger m_0 -> more aftershocks on average
    Cs_low, _ = obj.sample(torch.tensor([[6.0]] * 5000))
    Cs_hi, _ = obj.sample(torch.tensor([[8.0]] * 5000))
    assert Cs_hi.float().mean().item() > Cs_low.float().mean().item()

    # 3) boundary: K_max = 0 -> always zero
    obj0 = k_mod.K(childs=[_new_disc("K", [0])], parents=[_new_cont("M_0")],
                   K_max=0, device=DEVICE)
    Cs0, _ = obj0.sample(torch.tensor([[7.0]] * 100))
    assert (Cs0 == 0).all()


def test_K_log_prob():
    obj = k_mod.K(childs=[_new_disc("K", list(range(11)))],
                  parents=[_new_cont("M_0")],
                  K_max=10, device=DEVICE)

    # 1) log p sums to 1 over {0..K_max}
    m0 = torch.tensor([7.0])
    lp_all = torch.stack([
        obj.log_prob(torch.tensor([[float(j), 7.0]])).squeeze()
        for j in range(11)
    ])
    s = torch.logsumexp(lp_all, dim=0)
    assert abs(s.item()) < 1e-4

    # 2) out-of-range -> -inf
    assert math.isinf(obj.log_prob(torch.tensor([[11.0, 7.0]])).item())

    # 3) larger m_0 shifts log p towards higher k
    lp_low = obj.log_prob(torch.tensor([[5.0, 6.0]])).item()
    lp_hi = obj.log_prob(torch.tensor([[5.0, 8.0]])).item()
    assert lp_hi > lp_low

    # 4) k=0 has the highest prob for small m_0 (rate well below 1)
    lp_0 = obj.log_prob(torch.tensor([[0.0, 6.0]])).item()
    lp_5 = obj.log_prob(torch.tensor([[5.0, 6.0]])).item()
    assert lp_0 > lp_5


# ---------------------------------------------------------------------
# R
# ---------------------------------------------------------------------
def test_R_sample():
    obj = r_mod.R(childs=[_new_cont("R_t")],
                  parents=[_new_disc("K", list(range(5)))],
                  slot_idx=1, r_min=1.0, r_max=50.0, n=1.35, device=DEVICE)

    # 1) inactive slot -> 0
    pars = torch.tensor([[0]])
    Cs, lp = obj.sample(pars)
    assert Cs.item() == 0.0 and lp.item() == 0.0

    # 2) active: within [r_min, r_max]
    pars = torch.tensor([[3]] * 2000)
    Cs, _ = obj.sample(pars)
    assert (Cs >= 1.0).all() and (Cs <= 50.0).all()

    # 3) higher exponent -> shorter mean distance
    obj_hi = r_mod.R(childs=[_new_cont("R")],
                     parents=[_new_disc("K", list(range(5)))],
                     slot_idx=1, n=2.5, device=DEVICE)
    Cs_hi, _ = obj_hi.sample(torch.tensor([[3]] * 2000))
    assert Cs_hi.mean().item() < Cs.mean().item()


def test_R_log_prob():
    obj = r_mod.R(childs=[_new_cont("R_t")],
                  parents=[_new_disc("K", list(range(5)))],
                  slot_idx=1, device=DEVICE)

    # 1) inactive + r = 0 -> 0
    Cs = torch.tensor([[0.0, 0]])
    assert obj.log_prob(Cs).item() == 0.0

    # 2) inactive + r > 0 -> -inf
    Cs = torch.tensor([[5.0, 0]])
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) active + within range -> finite, decreasing in r
    Cs_close = torch.tensor([[2.0, 3]])
    Cs_far = torch.tensor([[40.0, 3]])
    assert obj.log_prob(Cs_close).item() > obj.log_prob(Cs_far).item()

    # 4) active + out of range -> -inf
    Cs = torch.tensor([[100.0, 3]])
    assert math.isinf(obj.log_prob(Cs).item())


# ---------------------------------------------------------------------
# V
# ---------------------------------------------------------------------
def test_V_sample():
    obj = v_mod.V(childs=[_new_cont("V_t")],
                  parents=[_new_disc("K", list(range(5)))],
                  slot_idx=2, device=DEVICE)

    # 1) inactive slot (K=1 < slot=2) -> 0
    Cs, _ = obj.sample(torch.tensor([[1]] * 10))
    assert (Cs == 0).all()

    # 2) active slot: within [0, 2 pi)
    Cs, _ = obj.sample(torch.tensor([[3]] * 5000))
    assert (Cs >= 0).all() and (Cs < 2 * math.pi).all()

    # 3) active slot: distribution mean ~ pi
    assert abs(Cs.mean().item() - math.pi) < 0.2


def test_V_log_prob():
    obj = v_mod.V(childs=[_new_cont("V_t")],
                  parents=[_new_disc("K", list(range(5)))],
                  slot_idx=1, device=DEVICE)

    # 1) inactive + v = 0 -> 0
    Cs = torch.tensor([[0.0, 0]])
    assert obj.log_prob(Cs).item() == 0.0

    # 2) inactive + v != 0 -> -inf
    Cs = torch.tensor([[1.0, 0]])
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) active + in range -> -log(2 pi)
    Cs = torch.tensor([[1.0, 3]])
    assert abs(obj.log_prob(Cs).item() + math.log(2 * math.pi)) < 1e-6

    # 4) active + out of range -> -inf
    Cs = torch.tensor([[10.0, 3]])
    assert math.isinf(obj.log_prob(Cs).item())


# ---------------------------------------------------------------------
# A
# ---------------------------------------------------------------------
def test_A_sample():
    obj = a_mod.A(childs=[_new_cont("A")],
                  parents=[_new_cont("M"), _new_cont("L_x"), _new_cont("L_y")],
                  edge_midpoint=(0.0, 0.0), device=DEVICE)

    # 1) inactive shock (M = 0) -> a = 0
    Cs, lp = obj.sample(torch.tensor([[0.0, 1.0, 1.0]]))
    assert Cs.item() == 0.0 and lp.item() == 0.0

    # 2) larger magnitude -> larger mean PGA (in log space)
    pars_low = torch.tensor([[6.0, 5.0, 0.0]] * 5000)
    pars_hi = torch.tensor([[8.0, 5.0, 0.0]] * 5000)
    a_low, _ = obj.sample(pars_low)
    a_hi, _ = obj.sample(pars_hi)
    assert a_hi.mean().item() > a_low.mean().item()

    # 3) closer epicentre -> larger mean PGA
    pars_close = torch.tensor([[7.0, 0.5, 0.0]] * 5000)
    pars_far = torch.tensor([[7.0, 50.0, 0.0]] * 5000)
    a_close, _ = obj.sample(pars_close)
    a_far, _ = obj.sample(pars_far)
    assert a_close.mean().item() > a_far.mean().item()


def test_A_log_prob():
    obj = a_mod.A(childs=[_new_cont("A")],
                  parents=[_new_cont("M"), _new_cont("L_x"), _new_cont("L_y")],
                  edge_midpoint=(0.0, 0.0), device=DEVICE)

    # 1) inactive + a=0 -> 0
    Cs = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    assert obj.log_prob(Cs).item() == 0.0

    # 2) inactive + a>0 -> -inf
    Cs = torch.tensor([[0.1, 0.0, 1.0, 1.0]])
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) active + a>0 -> finite
    Cs = torch.tensor([[0.1, 7.0, 5.0, 0.0]])
    assert torch.isfinite(obj.log_prob(Cs)).all()

    # 4) active + a = 0 -> -inf (delta peak only when inactive)
    Cs = torch.tensor([[0.0, 7.0, 5.0, 0.0]])
    assert math.isinf(obj.log_prob(Cs).item())


# ---------------------------------------------------------------------
# D0 / Dt
# ---------------------------------------------------------------------
def test_D0_sample():
    obj = d_mod.D0(childs=[_new_cont("D_0")], parents=[_new_cont("A_0")],
                   device=DEVICE)

    # 1) returns positive values
    Cs, _ = obj.sample(torch.tensor([[0.3]] * 1000))
    assert (Cs > 0).all()

    # 2) larger A -> larger mean D (positive log-slope beta=2.51)
    Cs_low, _ = obj.sample(torch.tensor([[0.05]] * 5000))
    Cs_hi, _ = obj.sample(torch.tensor([[0.5]] * 5000))
    assert Cs_hi.mean().item() > Cs_low.mean().item()

    # 3) boundary: A=0 still returns finite samples (clamp avoids -inf)
    Cs0, _ = obj.sample(torch.tensor([[0.0]] * 10))
    assert torch.isfinite(Cs0).all()


def test_D0_log_prob():
    obj = d_mod.D0(childs=[_new_cont("D_0")], parents=[_new_cont("A_0")],
                   device=DEVICE)

    # 1) positive D -> finite
    Cs = torch.tensor([[0.5, 0.3]])
    assert torch.isfinite(obj.log_prob(Cs)).all()

    # 2) D = 0 -> -inf (lognormal not defined at 0)
    Cs = torch.tensor([[0.0, 0.3]])
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) larger A shifts the mode of D upward
    lp_d_small = obj.log_prob(torch.tensor([[0.05, 0.3]])).item()
    lp_d_large = obj.log_prob(torch.tensor([[2.0, 0.3]])).item()
    # At A=0.3, mu ~ -1.91 + 2.51*log(0.3) ~ -4.93 -> small D more likely
    assert lp_d_small > lp_d_large


def test_Dt_sample():
    obj = d_mod.Dt(childs=[_new_cont("D_t")],
                   parents=[_new_cont("A_t"), _new_cont("D_prev")],
                   device=DEVICE)

    # 1) inactive (A=0) -> carry previous
    pars = torch.tensor([[0.0, 0.7]])
    Cs, lp = obj.sample(pars)
    assert Cs.item() == 0.7 and lp.item() == 0.0

    # 2) positive A -> finite samples > 0
    pars = torch.tensor([[0.3, 0.2]] * 1000)
    Cs, _ = obj.sample(pars)
    assert (Cs > 0).all() and torch.isfinite(Cs).all()

    # 3) boundary: very tiny A still produces finite values
    pars = torch.tensor([[1e-6, 0.5]] * 10)
    Cs, _ = obj.sample(pars)
    assert torch.isfinite(Cs).all()


def test_Dt_log_prob():
    obj = d_mod.Dt(childs=[_new_cont("D_t")],
                   parents=[_new_cont("A_t"), _new_cont("D_prev")],
                   device=DEVICE)

    # 1) inactive + d == d_prev -> 0
    Cs = torch.tensor([[0.7, 0.0, 0.7]])
    assert obj.log_prob(Cs).item() == 0.0

    # 2) inactive + d != d_prev -> -inf
    Cs = torch.tensor([[0.5, 0.0, 0.7]])
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) active + valid d -> finite
    Cs = torch.tensor([[0.5, 0.3, 0.2]])
    assert torch.isfinite(obj.log_prob(Cs)).all()

    # 4) active + d = 0 -> -inf
    Cs = torch.tensor([[0.0, 0.3, 0.2]])
    assert math.isinf(obj.log_prob(Cs).item())


# ---------------------------------------------------------------------
# X
# ---------------------------------------------------------------------
def test_X_sample():
    obj = x_mod.X(childs=[_new_disc("X", [0, 1])], parents=[_new_cont("D")],
                  threshold=0.4, device=DEVICE)

    # 1) D below threshold -> X = 1
    Cs, _ = obj.sample(torch.tensor([[0.1], [0.3]]))
    assert (Cs == 1).all()

    # 2) D above threshold -> X = 0
    Cs, _ = obj.sample(torch.tensor([[0.5], [1.0]]))
    assert (Cs == 0).all()

    # 3) boundary: D exactly at threshold -> X = 0 (strict less-than)
    Cs, _ = obj.sample(torch.tensor([[0.4]]))
    assert Cs.item() == 0


def test_X_log_prob():
    obj = x_mod.X(childs=[_new_disc("X", [0, 1])], parents=[_new_cont("D")],
                  threshold=0.4, device=DEVICE)

    # 1) consistent (X=1, D=0.2) -> 0
    Cs = torch.tensor([[1.0, 0.2]])
    assert obj.log_prob(Cs).item() == 0.0

    # 2) inconsistent (X=0, D=0.2) -> -inf
    Cs = torch.tensor([[0.0, 0.2]])
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) consistent failure (X=0, D=0.6) -> 0
    Cs = torch.tensor([[0.0, 0.6]])
    assert obj.log_prob(Cs).item() == 0.0

    # 4) boundary: X=1, D=threshold -> -inf
    Cs = torch.tensor([[1.0, 0.4]])
    assert math.isinf(obj.log_prob(Cs).item())


# ---------------------------------------------------------------------
# S
# ---------------------------------------------------------------------
def _trivial_refs(n_var, device=DEVICE):
    """Build trivial refs that classify every sample as S=2 (upper-1, upper-2)."""
    all_ones = torch.ones(1, n_var, 2, dtype=torch.int32, device=device)
    refs_upper = {1: all_ones, 2: all_ones}
    refs_lower = {1: torch.zeros(0, n_var, 2, dtype=torch.int32, device=device),
                  2: torch.zeros(0, n_var, 2, dtype=torch.int32, device=device)}
    return refs_upper, refs_lower


def test_S_sample():
    n_var = 4
    row_names = [f"e{i}" for i in range(n_var)]
    parents = [_new_disc(nm, [0, 1]) for nm in row_names]
    refs_upper, refs_lower = _trivial_refs(n_var)

    obj = s_mod.S(
        childs=[_new_disc("S", [0, 1, 2])], parents=parents,
        refs_dict_upper=refs_upper, refs_dict_lower=refs_lower,
        row_names=row_names, device=DEVICE,
    )

    # 1) all-1 components classified as upper for sys >= 2 -> S = 2
    X = torch.ones((3, n_var), dtype=torch.long)
    Cs, lp = obj.sample(X)
    assert (Cs == 2).all() and (lp == 0).all()

    # 2) swap to refs that finalise S = 0 (lower at level 1)
    fail_ref = torch.tensor([[[1, 1]] * n_var], dtype=torch.int32)  # 1 ref accepting any sample
    refs_upper_f = {1: torch.zeros(0, n_var, 2, dtype=torch.int32),
                    2: torch.zeros(0, n_var, 2, dtype=torch.int32)}
    refs_lower_f = {1: fail_ref, 2: torch.zeros(0, n_var, 2, dtype=torch.int32)}
    obj_f = s_mod.S(
        childs=[_new_disc("S", [0, 1, 2])], parents=parents,
        refs_dict_upper=refs_upper_f, refs_dict_lower=refs_lower_f,
        row_names=row_names, device=DEVICE,
    )
    Cs_f, _ = obj_f.sample(torch.zeros((3, n_var), dtype=torch.long))
    assert (Cs_f == 0).all()

    # 3) boundary: unknowns fall back to s_fun
    refs_upper_u = {1: torch.zeros(0, n_var, 2, dtype=torch.int32),
                    2: torch.zeros(0, n_var, 2, dtype=torch.int32)}
    refs_lower_u = {1: torch.zeros(0, n_var, 2, dtype=torch.int32),
                    2: torch.zeros(0, n_var, 2, dtype=torch.int32)}
    called = []
    def s_fun(comps):
        called.append(comps)
        return None, 1, None
    obj_u = s_mod.S(
        childs=[_new_disc("S", [0, 1, 2])], parents=parents,
        refs_dict_upper=refs_upper_u, refs_dict_lower=refs_lower_u,
        row_names=row_names, s_fun=s_fun, device=DEVICE,
    )
    Cs_u, _ = obj_u.sample(torch.tensor([[1, 0, 1, 0]]))
    assert Cs_u.item() == 1 and len(called) == 1


def test_S_log_prob():
    n_var = 4
    row_names = [f"e{i}" for i in range(n_var)]
    parents = [_new_disc(nm, [0, 1]) for nm in row_names]
    refs_upper, refs_lower = _trivial_refs(n_var)

    obj = s_mod.S(
        childs=[_new_disc("S", [0, 1, 2])], parents=parents,
        refs_dict_upper=refs_upper, refs_dict_lower=refs_lower,
        row_names=row_names, device=DEVICE,
    )

    # 1) declared S matches classifier output -> 0
    Cs = torch.cat([torch.tensor([[2]]), torch.ones((1, n_var), dtype=torch.long)], dim=1)
    assert obj.log_prob(Cs).item() == 0.0

    # 2) mismatch -> -inf
    Cs = torch.cat([torch.tensor([[0]]), torch.ones((1, n_var), dtype=torch.long)], dim=1)
    assert math.isinf(obj.log_prob(Cs).item())

    # 3) boundary: empty refs + s_fun providing S=1
    refs_e = {1: torch.zeros(0, n_var, 2, dtype=torch.int32),
              2: torch.zeros(0, n_var, 2, dtype=torch.int32)}
    obj_e = s_mod.S(
        childs=[_new_disc("S", [0, 1, 2])], parents=parents,
        refs_dict_upper=refs_e, refs_dict_lower=refs_e,
        row_names=row_names,
        s_fun=lambda c: (None, 1, None),
        device=DEVICE,
    )
    Cs = torch.tensor([[1, 1, 0, 1, 0]])
    assert obj_e.log_prob(Cs).item() == 0.0


# ---------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------
ALL_TESTS = [
    test_L0_sample, test_L0_log_prob,
    test_Lt_sample, test_Lt_log_prob,
    test_M0_sample, test_M0_log_prob,
    test_Mt_sample, test_Mt_log_prob,
    test_K_sample, test_K_log_prob,
    test_R_sample, test_R_log_prob,
    test_V_sample, test_V_log_prob,
    test_A_sample, test_A_log_prob,
    test_D0_sample, test_D0_log_prob,
    test_Dt_sample, test_Dt_log_prob,
    test_X_sample, test_X_log_prob,
    test_S_sample, test_S_log_prob,
]


if __name__ == "__main__":
    failed = []
    for t in ALL_TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed.append((t.__name__, repr(e)))
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed.append((t.__name__, repr(e)))
            print(f"  ERROR {t.__name__}: {e!r}")

    print()
    print(f"{len(ALL_TESTS) - len(failed)}/{len(ALL_TESTS)} passed")
    if failed:
        sys.exit(1)
