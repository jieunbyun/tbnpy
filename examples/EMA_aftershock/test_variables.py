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
    # ---- one-axis uniform on [0, 10] ----
    obj_x = l_mod.L0(childs=[_new_cont("L_x_0")], low=0.0, high=10.0, device=DEVICE)
    Cs, _ = obj_x.sample(n_sample=5000)
    assert ((Cs >= 0.0) & (Cs <= 10.0)).all()

    # 2) different range
    obj_y = l_mod.L0(childs=[_new_cont("L_y_0")], low=100.0, high=110.0, device=DEVICE)
    Cs2, _ = obj_y.sample(n_sample=5000)
    assert Cs2.min().item() >= 100.0 and Cs2.max().item() <= 110.0

    # 3) boundary: zero-width interval collapses to a point
    obj_pt = l_mod.L0(childs=[_new_cont("x")], low=2.0, high=2.0, device=DEVICE)
    Cs3, _ = obj_pt.sample(n_sample=10)
    assert torch.allclose(Cs3, torch.full((10,), 2.0))


def test_L0_log_prob():
    obj = l_mod.L0(childs=[_new_cont("L_x_0")], low=0.0, high=10.0, device=DEVICE)

    # 1) inside: log p = -log(width) = -log(10)
    lp = obj.log_prob(torch.tensor([[3.0], [9.9]]))
    assert torch.allclose(lp, torch.full((2,), -math.log(10.0)), atol=1e-5)

    # 2) outside: -inf
    lp_out = obj.log_prob(torch.tensor([[-1.0], [11.0]]))
    assert torch.isinf(lp_out).all() and (lp_out < 0).all()

    # 3) smaller width -> higher log density
    obj2 = l_mod.L0(childs=[_new_cont("x")], low=0.0, high=1.0, device=DEVICE)
    lp2 = obj2.log_prob(torch.tensor([[0.5]]))
    assert lp2.item() > lp[0].item()  # 0.0 > -log(10)

    # 4) boundary: exactly at edge is in-range
    lp_edge = obj.log_prob(torch.tensor([[10.0]]))
    assert torch.isfinite(lp_edge).all()


# ---------------------------------------------------------------------
# Lt
# ---------------------------------------------------------------------
def test_Lt_sample():
    obj_x = l_mod.Lt(
        childs=[_new_cont("L_x_t")],
        parents=[_new_cont("L_x_0"), _new_cont("R_t"), _new_cont("V_t")],
        axis="x", device=DEVICE,
    )
    obj_y = l_mod.Lt(
        childs=[_new_cont("L_y_t")],
        parents=[_new_cont("L_y_0"), _new_cont("R_t"), _new_cont("V_t")],
        axis="y", device=DEVICE,
    )

    # 1) zero R -> output equals epicentre on both axes
    Cs_x, lp = obj_x.sample(torch.tensor([[3.0, 0.0, 1.234]]))
    assert abs(Cs_x.item() - 3.0) < 1e-6 and lp.item() == 0.0
    Cs_y, _ = obj_y.sample(torch.tensor([[4.0, 0.0, 1.234]]))
    assert abs(Cs_y.item() - 4.0) < 1e-6

    # 2) V = 0, R > 0 -> x-axis shifts by R, y-axis stays
    Cs_x, _ = obj_x.sample(torch.tensor([[0.0, 2.0, 0.0]]))
    assert abs(Cs_x.item() - 2.0) < 1e-6
    Cs_y, _ = obj_y.sample(torch.tensor([[0.0, 2.0, 0.0]]))
    assert abs(Cs_y.item() - 0.0) < 1e-6

    # 3) V = pi/2 -> x-axis stays, y-axis shifts by R
    Cs_x, _ = obj_x.sample(torch.tensor([[0.0, 2.0, math.pi / 2]]))
    assert abs(Cs_x.item() - 0.0) < 1e-6
    Cs_y, _ = obj_y.sample(torch.tensor([[0.0, 2.0, math.pi / 2]]))
    assert abs(Cs_y.item() - 2.0) < 1e-6


def test_Lt_log_prob():
    obj_x = l_mod.Lt(
        childs=[_new_cont("L_x_t")],
        parents=[_new_cont("L_x_0"), _new_cont("R_t"), _new_cont("V_t")],
        axis="x", device=DEVICE,
    )

    # Cs columns = [L_x_t, L_x_0, R, V]
    # 1) consistent: L_x_t = 3, L_x_0 = 3, R = 0  -> 0
    Cs = torch.tensor([[3.0, 3.0, 0.0, 0.0]])
    assert obj_x.log_prob(Cs).item() == 0.0

    # 2) inconsistent: L_x_t = 5 but L_x_0 = 3, R = 0 -> -inf
    Cs = torch.tensor([[5.0, 3.0, 0.0, 0.0]])
    assert math.isinf(obj_x.log_prob(Cs).item())

    # 3) consistent shift along x: L_x_t = 5, L_x_0 = 3, R = 2, V = 0
    Cs = torch.tensor([[5.0, 3.0, 2.0, 0.0]])
    assert obj_x.log_prob(Cs).item() == 0.0


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


def test_Mt_log_prob_ex():
    """Pre-calculated density: ``f(M_t = 5 | M_0 = 7.5, K active)``.

    With ``beta = 0.76``, ``m_min = 4.5`` and ``delta_m = 1.2``, the
    aftershock distribution at an active slot is truncated exponential on
    ``[m_min, M_0 - delta_m] = [4.5, 6.3]``::

        f(m) = beta * exp(-beta (m - m_min))
             / (1 - exp(-beta (m_high - m_min)))

    Plugging in ``m = 5``:

        Z   = 1 - exp(-0.76 * 1.8) = 1 - exp(-1.368) ~ 0.74537
        f(5) = 0.76 * exp(-0.76 * 0.5) / Z
             = 0.76 * exp(-0.38) / 0.74537
             ~ 0.6975

    so ``log f(5) ~ -0.3604``.
    """
    beta = 0.76
    m_min = 4.5
    delta_m = 1.2

    obj = m_mod.Mt(
        childs=[_new_cont("M_t")],
        parents=[_new_cont("M_0"), _new_disc("K", list(range(5)))],
        slot_idx=1, m_min=m_min, beta=beta, delta_m=delta_m, device=DEVICE,
    )

    # ---- Pre-calculation (done in plain Python) ----
    m_0 = 7.5
    m_t = 5.0
    m_high = m_0 - delta_m
    Z = 1.0 - math.exp(-beta * (m_high - m_min))
    expected_pdf = beta * math.exp(-beta * (m_t - m_min)) / Z
    expected_logp = math.log(expected_pdf)

    assert abs(expected_pdf - 0.6975) < 1e-3, (
        f"sanity check on hand calc failed: {expected_pdf:.4f}"
    )

    # ---- Compare to Mt.log_prob ----
    # Cs columns = [M_t value, M_0, K]; K=3 >= slot_idx=1 -> active slot
    Cs = torch.tensor([[m_t, m_0, 3.0]])
    logp = obj.log_prob(Cs).item()

    assert abs(logp - expected_logp) < 1e-5, (
        f"log_prob mismatch: expected {expected_logp:.6f}, got {logp:.6f}"
    )
    assert abs(math.exp(logp) - expected_pdf) < 1e-5


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


def test_K_log_prob_ex():
    """Pre-calculated: ``P(K = 2 | M_0 = 7.0, K_max = 5)``.

    Modified Omori law gives the Poisson mean::

        K_0   = 10 ** (a + b (m_0 - m_min)) = 10 ** (-0.76) ~ 0.17378
        Lam   = K_0 / (1 - p) * ((T + c) ** (1 - p) - c ** (1 - p))
              ~ 0.17378 * 5.195 ~ 0.9028

    Untruncated Poisson PMF at k = 2::

        log pmf = 2 log Lam - Lam - log(2!) ~ -1.8003

    Truncating to ``k in {0,...,5}`` barely shifts the normaliser (mass
    beyond 5 is tiny), so ``log P(K = 2) ~ -1.8000`` and ``P ~ 0.1653``.
    """
    K_max = 5
    T, c, p_par = 7.0, 0.05, 1.08
    a_par, b_par, m_min = -1.67, 0.91, 6.0
    m_0 = 7.0

    obj = k_mod.K(
        childs=[_new_disc("K", list(range(K_max + 1)))],
        parents=[_new_cont("M_0")],
        K_max=K_max, T=T, c=c, p=p_par,
        a=a_par, b=b_par, m_min=m_min, device=DEVICE,
    )

    # ---- Pre-calculation (plain Python) ----
    K0 = 10.0 ** (a_par + b_par * (m_0 - m_min))
    factor = ((T + c) ** (1.0 - p_par) - c ** (1.0 - p_par)) / (1.0 - p_par)
    lam = K0 * factor

    def _log_pmf(k):
        return k * math.log(lam) - lam - math.lgamma(k + 1)

    log_norm = math.log(sum(math.exp(_log_pmf(j)) for j in range(K_max + 1)))
    expected_logp = _log_pmf(2) - log_norm
    expected_pmf = math.exp(expected_logp)

    assert abs(expected_pmf - 0.1653) < 1e-3, (
        f"sanity check on hand calc failed: {expected_pmf:.4f}"
    )

    Cs = torch.tensor([[2.0, m_0]])
    logp = obj.log_prob(Cs).item()
    assert abs(logp - expected_logp) < 1e-5, (
        f"log_prob mismatch: expected {expected_logp:.6f}, got {logp:.6f}"
    )


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


def test_A_log_prob_ex():
    """Pre-calculated: lognormal density ``f_A(a | M, L_x, L_y)`` at one point.

    Setup: edge midpoint ``(0, 0)``, ``M = 7.0``, ``L = (10, 0)``
    => ``r = 10``::

        eps   = 0.149 * exp(0.647 * 7)                  ~ 13.279
        mu    = ln 0.55 - 3.512 + 0.904 * 7
                - 1.328 * ln(sqrt(10^2 + eps^2))        ~ -1.5145
        sigma = 0.52

    PDF at ``a = 0.5`` via lognormal change of variables::

        log f_A(a) = log Normal(log a | mu, sigma) - log a   ~ -0.8194
    """
    sigma = 0.52
    obj = a_mod.A(
        childs=[_new_cont("A")],
        parents=[_new_cont("M"), _new_cont("L_x"), _new_cont("L_y")],
        edge_midpoint=(0.0, 0.0), sigma=sigma, device=DEVICE,
    )

    # ---- Pre-calculation ----
    M, lx, ly = 7.0, 10.0, 0.0
    a_val = 0.5
    r = math.sqrt(lx ** 2 + ly ** 2)             # = 10.0
    eps = 0.149 * math.exp(0.647 * M)
    mu = (math.log(0.55) - 3.512 + 0.904 * M
          - 1.328 * math.log(math.sqrt(r * r + eps * eps)))

    log_a = math.log(a_val)
    log_normal = (
        -0.5 * math.log(2 * math.pi * sigma * sigma)
        - (log_a - mu) ** 2 / (2 * sigma * sigma)
    )
    expected_logp = log_normal - log_a

    Cs = torch.tensor([[a_val, M, lx, ly]])
    logp = obj.log_prob(Cs).item()
    assert abs(logp - expected_logp) < 1e-5, (
        f"log_prob mismatch: expected {expected_logp:.6f}, got {logp:.6f}"
    )


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


def test_D0_log_prob_ex():
    """Pre-calculated: ``f_{D_0}(d | A_0)`` under the Ghosh-2015 single-shock model.

    With ``(alpha, beta, eps) = (-1.91, 2.51, 0.70)`` and ``A_0 = 0.3``::

        mu    = alpha + beta * ln A_0 = -1.91 + 2.51 * ln(0.3) ~ -4.932
        sigma = eps = 0.70

    PDF at ``d = 0.1`` via lognormal change of variables::

        log f_D(d) = log Normal(log d | mu, sigma) - log d   ~ -5.313
    """
    alpha, beta_par, eps = -1.91, 2.51, 0.70
    obj = d_mod.D0(
        childs=[_new_cont("D_0")], parents=[_new_cont("A_0")],
        alpha=alpha, beta=beta_par, eps=eps, device=DEVICE,
    )

    # ---- Pre-calculation ----
    A = 0.3
    d_val = 0.1
    mu = alpha + beta_par * math.log(A)
    sigma = eps

    log_d = math.log(d_val)
    log_normal = (
        -0.5 * math.log(2 * math.pi * sigma * sigma)
        - (log_d - mu) ** 2 / (2 * sigma * sigma)
    )
    expected_logp = log_normal - log_d

    Cs = torch.tensor([[d_val, A]])
    logp = obj.log_prob(Cs).item()
    assert abs(logp - expected_logp) < 1e-5, (
        f"log_prob mismatch: expected {expected_logp:.6f}, got {logp:.6f}"
    )


def test_Dt_sample():
    obj = d_mod.Dt(childs=[_new_cont("D_t")],
                   parents=[_new_cont("A_t"), _new_cont("D_prev")],
                   device=DEVICE)

    # 1) inactive (A=0) -> carry previous
    pars = torch.tensor([[0.0, 0.7]])
    Cs, lp = obj.sample(pars)
    assert abs(Cs.item() - 0.7) < 1e-6 and abs(lp.item()) < 1e-6

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


def test_X_log_prob_ex():
    """Pre-calculated: ``P(X | D)`` is deterministic, so log P is 0 or -inf.

    With ``threshold = 0.4`` and ``D = 0.2``::

        expected X = 1   (since 0.2 < 0.4)
        => P(X = 1 | D = 0.2) = 1   => log P = 0
        => P(X = 0 | D = 0.2) = 0   => log P = -inf
    """
    threshold = 0.4
    obj = x_mod.X(
        childs=[_new_disc("X", [0, 1])], parents=[_new_cont("D")],
        threshold=threshold, device=DEVICE,
    )

    # ---- Pre-calculation ----
    D = 0.2
    expected_X = 1 if D < threshold else 0      # = 1
    expected_logp_match = math.log(1.0)         # = 0.0

    Cs = torch.tensor([[float(expected_X), D]])
    logp = obj.log_prob(Cs).item()
    assert abs(logp - expected_logp_match) < 1e-6, (
        f"log_prob mismatch: expected {expected_logp_match}, got {logp}"
    )

    # mismatch -> -inf
    Cs_bad = torch.tensor([[float(1 - expected_X), D]])
    assert math.isinf(obj.log_prob(Cs_bad).item())


# ---------------------------------------------------------------------
# S
# ---------------------------------------------------------------------
def _trivial_refs(n_var, max_st=2, device=DEVICE):
    """Build trivial refs that classify every sample as ``S = max_st``.

    Two index spaces coexist and should not be confused:

    * Ref-dict **keys** are 1-indexed cascade levels: key ``s`` holds
      the refs used to test whether ``S >= s`` (upper) or
      ``S <= s - 1`` (lower). The rsr convention requires keys
      ``1, 2, ..., max_st``.
    * Resulting **S values** are 0-indexed: ``0, 1, ..., max_st``.

    With all-ones upper refs at every level and no lower refs, every
    sample passes upper at each level and is finalised at the last as
    ``S = max_st``.
    """
    all_ones = torch.ones(1, n_var, 2, dtype=torch.int32, device=device)
    empty = torch.zeros(0, n_var, 2, dtype=torch.int32, device=device)
    refs_upper = {s: all_ones for s in range(1, max_st + 1)}
    refs_lower = {s: empty for s in range(1, max_st + 1)}
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

    # 4) boundary: empty refs and no s_fun -> default unknown_state = -1.
    # Valid system states are 0..max_st, so -1 is deliberately outside the
    # valid range to signal "unresolved".
    obj_d = s_mod.S(
        childs=[_new_disc("S", [0, 1, 2])], parents=parents,
        refs_dict_upper=refs_upper_u, refs_dict_lower=refs_lower_u,
        row_names=row_names, s_fun=None, device=DEVICE,
    )
    Cs_d, _ = obj_d.sample(torch.tensor([[1, 0, 1, 0]]))
    assert Cs_d.item() == -1


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


def test_S_log_prob_ex():
    """Pre-calculated: ``P(S | X)`` is deterministic given the refs.

    Setup: 2 edges, refs that accept any sample as upper at both levels.
    Multi-state classification logic::

        level 1: sample is in upper_refs[1]   -> carry to level 2
        level 2: sample is in upper_refs[2]   -> finalise S = max_st = 2

    So with ``X = [1, 1]``::

        expected S = 2   => log P(S = 2 | X) = 0
        any other S      => log P = -inf
    """
    n_var = 2
    row_names = ["e0", "e1"]
    parents = [_new_disc(nm, [0, 1]) for nm in row_names]

    accept_all = torch.ones(1, n_var, 2, dtype=torch.int32, device=DEVICE)
    empty = torch.zeros(0, n_var, 2, dtype=torch.int32, device=DEVICE)
    refs_upper = {1: accept_all, 2: accept_all}
    refs_lower = {1: empty, 2: empty}

    obj = s_mod.S(
        childs=[_new_disc("S", [0, 1, 2])], parents=parents,
        refs_dict_upper=refs_upper, refs_dict_lower=refs_lower,
        row_names=row_names, device=DEVICE,
    )

    # ---- Pre-calculation ----
    X = [1, 1]
    expected_S = 2                                  # classifier walks up to max_st
    expected_logp_match = math.log(1.0)             # = 0.0

    Cs = torch.tensor([[float(expected_S)] + [float(x) for x in X]])
    logp = obj.log_prob(Cs).item()
    assert abs(logp - expected_logp_match) < 1e-6, (
        f"log_prob mismatch: expected {expected_logp_match}, got {logp}"
    )

    # mismatch -> -inf
    Cs_bad = torch.tensor([[0.0] + [float(x) for x in X]])
    assert math.isinf(obj.log_prob(Cs_bad).item())


# ---------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------
ALL_TESTS = [
    test_L0_sample, test_L0_log_prob,
    test_Lt_sample, test_Lt_log_prob,
    test_M0_sample, test_M0_log_prob,
    test_Mt_sample, test_Mt_log_prob, test_Mt_log_prob_ex,
    test_K_sample, test_K_log_prob, test_K_log_prob_ex,
    test_R_sample, test_R_log_prob,
    test_V_sample, test_V_log_prob,
    test_A_sample, test_A_log_prob, test_A_log_prob_ex,
    test_D0_sample, test_D0_log_prob, test_D0_log_prob_ex,
    test_Dt_sample, test_Dt_log_prob,
    test_X_sample, test_X_log_prob, test_X_log_prob_ex,
    test_S_sample, test_S_log_prob, test_S_log_prob_ex,
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
