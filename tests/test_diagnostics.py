import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tbnpy import diagnostics
from tbnpy.compile import compile_file

ABCDE = Path(__file__).resolve().parents[1] / "examples" / "ABCDE" / "abcde.tbn.json"


@pytest.fixture(scope="module")
def model():
    spec, varis, probs = compile_file(ABCDE)
    return spec, varis, probs


# ---------------------------------------------------------------------------
# Effective sample size
# ---------------------------------------------------------------------------
def test_uniform_weights_give_full_ess():
    lw = torch.zeros(1000)
    assert diagnostics.effective_sample_size(lw).item() == pytest.approx(1000.0)


def test_one_dominant_weight_gives_ess_near_one():
    lw = torch.full((1000,), -50.0)
    lw[0] = 0.0
    assert diagnostics.effective_sample_size(lw).item() == pytest.approx(1.0, abs=1e-3)


def test_all_impossible_weights_give_zero_not_nan():
    """An impossible observation must be reportable, not a silent NaN."""
    lw = torch.full((500,), -float("inf"))
    ess = diagnostics.effective_sample_size(lw)
    assert ess.item() == 0.0 and not math.isnan(ess.item())


def test_ess_handles_a_batch_of_evidence_rows():
    lw = torch.stack([torch.zeros(100), torch.full((100,), -float("inf"))])
    assert diagnostics.effective_sample_size(lw).tolist() == pytest.approx([100.0, 0.0])


def test_partially_impossible_weights_count_only_the_survivors():
    lw = torch.full((100,), -float("inf"))
    lw[:4] = 0.0
    assert diagnostics.effective_sample_size(lw).item() == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Weighted quantiles
# ---------------------------------------------------------------------------
def test_weighted_quantile_matches_unweighted_when_weights_are_equal():
    x = np.random.default_rng(0).normal(size=10_000)
    w = np.ones_like(x)
    got = diagnostics.weighted_quantile(x, w, [0.25, 0.5, 0.75])
    want = np.quantile(x, [0.25, 0.5, 0.75])
    assert np.allclose(got, want, atol=0.02)


def test_weighted_quantile_follows_the_weights():
    x = np.array([0.0, 1.0])
    lo = diagnostics.weighted_quantile(x, np.array([0.99, 0.01]), [0.5])[0]
    hi = diagnostics.weighted_quantile(x, np.array([0.01, 0.99]), [0.5])[0]
    assert lo < 0.5 < hi


# ---------------------------------------------------------------------------
# Posterior correctness
# ---------------------------------------------------------------------------
def _analytic_C(y):
    """Exact E[C | OC = y] for the ABCDE model, by enumerating A and B."""
    s2 = 1 / (1 / 0.36 + 1 / 0.25)
    terms = []
    for a, pa in ((-0.3, 0.5), (0.3, 0.5)):
        for b, pb in ((-0.2, 0.3), (0.0, 0.4), (0.2, 0.3)):
            mu = a + b
            lik = math.exp(-(y - mu) ** 2 / (2 * 0.61)) / math.sqrt(2 * math.pi * 0.61)
            terms.append((pa * pb * lik, s2 * (mu / 0.36 + y / 0.25)))
    z = sum(w for w, _ in terms)
    return sum(w * m for w, m in terms) / z


@pytest.mark.parametrize("y", [-0.3, 0.4])
def test_weighted_posterior_matches_the_analytic_mean(model, y):
    _, varis, probs = model
    r = diagnostics.posterior(probs, ["C"], {"OC": y}, n_sample=200_000,
                              varis=varis, seed=1)
    got = r.summary(varis).loc["C", "mean"]
    assert got == pytest.approx(_analytic_C(y), abs=5e-3)


def test_posterior_pmf_of_a_discrete_parent_shifts_with_evidence(model):
    _, varis, probs = model

    def p_high(y):
        r = diagnostics.posterior(probs, ["A"], {"OC": y}, n_sample=50_000,
                                  varis=varis, seed=0)
        return r.pmf("A", varis["A"]).loc[1, "probability"]

    assert p_high(-1.0) < 0.5 < p_high(1.0)


def test_pmf_sums_to_one(model):
    _, varis, probs = model
    r = diagnostics.posterior(probs, ["B"], {"OC": 0.2}, n_sample=20_000,
                              varis=varis, seed=0)
    assert r.pmf("B", varis["B"])["probability"].sum() == pytest.approx(1.0)


def test_no_evidence_reproduces_the_prior(model):
    _, varis, probs = model
    r = diagnostics.posterior(probs, ["C"], {}, n_sample=50_000, varis=varis, seed=0)
    assert r.summary(varis).loc["C", "mean"] == pytest.approx(0.0, abs=0.02)
    assert r.ess.item() == pytest.approx(50_000.0)


def test_several_evidence_rows_are_independent_scenarios(model):
    _, varis, probs = model
    r = diagnostics.posterior(probs, ["C"], {"OC": [-0.5, 0.0, 0.5]},
                              n_sample=20_000, varis=varis, seed=0)
    means = [r.summary(varis, row=i).loc["C", "mean"] for i in range(3)]
    assert means[0] < means[1] < means[2]


def test_ess_collapses_on_tail_evidence(model):
    """The number that stops a smooth-looking histogram being believed."""
    _, varis, probs = model
    central = diagnostics.posterior(probs, ["C"], {"OC": -0.3}, n_sample=20_000,
                                    varis=varis, seed=0)
    tail = diagnostics.posterior(probs, ["C"], {"OC": -4.0}, n_sample=20_000,
                                 varis=varis, seed=0)
    assert central.ess_fraction[0] > 0.5
    assert tail.ess_fraction[0] < 0.01
    assert tail.is_degenerate()[0] and not central.is_degenerate()[0]


# ---------------------------------------------------------------------------
# Evidence validation
# ---------------------------------------------------------------------------
def test_sound_evidence_raises_no_issues(model):
    _, varis, probs = model
    assert diagnostics.check_evidence(probs, varis, {"OC": -0.3}) == []


def test_deterministic_continuous_node_is_refused_with_remedies(model):
    _, varis, probs = model
    issues = diagnostics.check_evidence(probs, varis, {"E": 1.7})
    assert len(issues) == 1
    assert issues[0].severity == "error"
    assert len(issues[0].suggestions) == 3


def test_posterior_refuses_impossible_evidence_by_default(model):
    _, varis, probs = model
    with pytest.raises(ValueError, match="deterministic"):
        diagnostics.posterior(probs, ["C"], {"E": 1.7}, n_sample=100, varis=varis)


def test_out_of_range_discrete_state_is_caught(model):
    _, varis, probs = model
    issues = diagnostics.check_evidence(probs, varis, {"D": 7})
    assert "out of range" in issues[0].issue


def test_unknown_evidence_node_is_caught(model):
    _, varis, probs = model
    issues = diagnostics.check_evidence(probs, varis, {"Missing": 1.0})
    assert "not a node" in issues[0].issue


def test_hand_written_deterministic_node_is_caught_at_runtime():
    """A custom class has no spec to inspect, so the ESS check is the backstop."""
    from examples.ABCDE import s1_define_model as hand

    varis = hand.define_variables()
    probs = hand.define_probs(varis)
    assert diagnostics.check_evidence(probs, varis, {"E": 1.7}) == []   # static blind

    r = diagnostics.posterior(probs, ["C"], {"E": 1.7}, n_sample=2000,
                              varis=varis, validate=False)
    assert r.ess.item() == 0.0
    assert r.is_degenerate()[0]
    assert any("no particle is compatible" in i.issue for i in r.issues)


# ---------------------------------------------------------------------------
# Sense check
# ---------------------------------------------------------------------------
def test_sense_check_reports_every_variable(model):
    _, varis, probs = model
    report = diagnostics.run_sense_check(probs, varis, n_sample=4000, seed=0)
    assert set(report.marginals.index) == set(varis)
    assert report.correlation.loc["C", "OC"] > 0.5


def test_sense_check_flags_a_gate_that_correlation_misses(model):
    """D deterministically gates E, yet rho(D, E) is ~0: only the conditional
    spread reveals the dependence."""
    _, varis, probs = model
    report = diagnostics.run_sense_check(probs, varis, n_sample=8000, seed=0)
    assert abs(report.correlation.loc["D", "E"]) < 0.1
    assert report.conditionals["D"].loc["E", "effect_std"] > 1.0
    hidden = report.flags[report.flags["issue"] == "dependence invisible to correlation"]
    assert "D -> E" in set(hidden["variable"])


def test_sense_check_flags_bound_violations(model):
    _, varis, probs = model
    report = diagnostics.run_sense_check(probs, varis, n_sample=4000, seed=0,
                                         bounds={"C": (-0.5, 0.5)})
    breaches = report.flags[report.flags["issue"].str.contains("declared bound")]
    assert set(breaches["variable"]) == {"C"}


def test_report_saves_markdown_and_csvs(model, tmp_path):
    _, varis, probs = model
    report = diagnostics.run_sense_check(probs, varis, n_sample=1000, seed=0)
    report.save(tmp_path / "sc")
    assert (tmp_path / "sc.md").exists()
    assert (tmp_path / "sc_marginals.csv").exists()
    assert "Forward-sampling sense check" in (tmp_path / "sc.md").read_text(
        encoding="utf-8")


def test_markdown_table_renders_non_finite_values_readably():
    df = pd.DataFrame({"v": [1.0, float("inf"), float("nan")]})
    df.index.name = "i"
    out = diagnostics.md_table(df)
    assert "inf" in out and "nan" not in out
