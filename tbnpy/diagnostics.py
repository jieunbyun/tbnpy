"""
Model diagnostics: prior sense checks, evidence validation, and weighted
posterior summaries.

Everything here is plain torch/numpy/pandas and lives in core deliberately.
A ``pip install tbnpy`` user gets the same checks the GUI shows, and the GUI
stays a thin presentation layer over this module rather than the place where
the real work happens.

Three groups of functions:

Prior sense check
    :func:`run_sense_check` forward-samples the whole model and reports
    marginals, bound violations, rank correlations, and how each discrete
    variable moves the rest. Catches unit errors, sign flips, dead variables,
    and NaN paths.

Evidence validation
    :func:`check_evidence` refuses, with an explanation, the conditioning
    requests that cannot work -- most importantly a hard observation of a
    deterministic continuous node, which has probability zero and yields a
    NaN posterior rather than an error.

Weighted posterior
    :func:`posterior` wraps :func:`tbnpy.inference.sample_evidence`, which is
    likelihood weighting, and returns summaries *together with* the effective
    sample size. ESS is not decoration: at 20k particles a tail observation
    can leave a few hundred real ones behind a perfectly smooth-looking
    histogram, and nothing else in the output reveals it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tbnpy import inference

__all__ = [
    "SenseCheckReport", "run_sense_check", "forward_sample_all", "plot_report",
    "EvidenceIssue", "check_evidence",
    "PosteriorResult", "posterior", "effective_sample_size",
    "weighted_quantile",
]

NEG_INF = -float("inf")


def _is_discrete(var) -> bool:
    return not isinstance(var.values, tuple)


def _as_var_map(varis):
    if isinstance(varis, dict):
        return varis
    return {v.name: v for v in varis}


# ==========================================================================
# Forward sampling
# ==========================================================================
def forward_sample_all(probs, n_sample=20_000, batch_size=50_000, seed=0):
    """Prior-sample every node into a tidy ``(n_sample, n_vars)`` DataFrame.

    ``inference.sample`` returns each node's child samples only, shaped
    ``(N,)`` for a generated or custom single-child node and ``(N, n_childs)``
    for a tabular :class:`~tbnpy.cpt.Cpt`, so both layouts are unpacked here.
    """
    if seed is not None:
        torch.manual_seed(seed)

    out = inference.sample(probs, list(probs.keys()), n_sample=n_sample,
                           batch_size=batch_size)

    columns = {}
    for obj in out.values():
        Cs = obj.Cs.detach().cpu()
        if Cs.ndim == 1:
            Cs = Cs.unsqueeze(1)
        for j, child in enumerate(obj.childs):
            columns[child.name] = Cs[:, j].numpy()
    return pd.DataFrame(columns)


# ==========================================================================
# Prior sense check
# ==========================================================================
@dataclass
class SenseCheckReport:
    samples: pd.DataFrame
    marginals: pd.DataFrame
    discrete_pmfs: dict
    correlation: pd.DataFrame
    conditionals: dict
    flags: pd.DataFrame
    n_sample: int

    def save(self, stem):
        stem = Path(stem)
        stem.parent.mkdir(parents=True, exist_ok=True)
        self.marginals.to_csv(stem.with_name(stem.name + "_marginals.csv"))
        self.correlation.to_csv(stem.with_name(stem.name + "_correlation.csv"))
        self.flags.to_csv(stem.with_name(stem.name + "_flags.csv"), index=False)
        for g, table in self.conditionals.items():
            table.to_csv(stem.with_name(stem.name + f"_given_{g}.csv"))
        stem.with_name(stem.name + ".md").write_text(self.to_markdown(),
                                                     encoding="utf-8")
        return stem

    def to_markdown(self, top_k=25, corr_cut=0.15):
        lines = [f"# Forward-sampling sense check ({self.n_sample:,} prior samples)",
                 "", "## Marginals", "",
                 md_table(self.marginals, fmt="{:.4g}"), ""]

        if self.discrete_pmfs:
            lines += ["## Discrete marginals (pmf)", ""]
            for name, pmf in self.discrete_pmfs.items():
                lines += [f"**{name}**", "", md_table(pmf, fmt="{:.4f}"), ""]

        lines += ["## Strongest pairwise correlations (Spearman)", ""]
        pairs = ranked_pairs(self.correlation)
        shown = pairs[pairs["rho"].abs() >= corr_cut].head(top_k)
        lines += [md_table(shown, index=False, fmt="{:.3f}"), ""]

        if self.conditionals:
            lines += ["## Conditional behaviour, given each discrete variable", "",
                      "`effect` is the larger of `effect_mean` and `effect_std`, "
                      "in unconditional standard deviations. A large `effect` "
                      "beside a near-zero rho means a switch or gate, not an "
                      "absence of dependence.", ""]
            for g, table in self.conditionals.items():
                top = table[table["effect"].abs() >= 0.05]
                if len(top):
                    lines += [f"**given {g}**", "",
                              md_table(top.head(top_k), fmt="{:.4g}"), ""]

        lines += ["## Flags", ""]
        lines += ["None." if not len(self.flags)
                  else md_table(self.flags, index=False), ""]
        lines += ["## Full correlation matrix (Spearman)", "",
                  md_table(self.correlation, fmt="{:.2f}"), ""]
        return "\n".join(lines)


def md_table(df, index=True, fmt="{:.4g}"):
    """Minimal markdown table writer, so reports need no ``tabulate``."""
    def cell(v):
        if v is None:
            return ""
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            if np.isinf(v):
                return "inf" if v > 0 else "-inf"
            return fmt.format(v)
        return str(v)

    header = ([str(df.index.name or "")] if index else []) + [str(c) for c in df.columns]
    rows = [([str(idx)] if index else []) + [cell(v) for v in row]
            for idx, row in df.iterrows()]
    widths = [max([len(header[j])] + [len(r[j]) for r in rows])
              for j in range(len(header))]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |",
           "| " + " | ".join("-" * w for w in widths) + " |"]
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |"
            for r in rows]
    return "\n".join(out)


def ranked_pairs(corr):
    names = list(corr.columns)
    rows = [{"var_a": names[a], "var_b": names[b], "rho": corr.iloc[a, b]}
            for a in range(len(names)) for b in range(a + 1, len(names))
            if np.isfinite(corr.iloc[a, b])]
    if not rows:
        return pd.DataFrame(columns=["var_a", "var_b", "rho", "abs_rho"])
    out = pd.DataFrame(rows)
    out["abs_rho"] = out["rho"].abs()
    return out.sort_values("abs_rho", ascending=False).reset_index(drop=True)


def summarise_marginals(df, varis, bounds=None):
    bounds = bounds or {}
    varis = _as_var_map(varis)
    rows = []
    for name in df.columns:
        x = df[name].to_numpy(dtype=float)
        finite = np.isfinite(x)
        xf = x[finite]
        var = varis.get(name)
        row = {
            "variable": name,
            "kind": ("discrete" if var is not None and _is_discrete(var)
                     else "continuous" if var is not None else "?"),
            "mean": xf.mean() if xf.size else np.nan,
            "std": xf.std(ddof=1) if xf.size > 1 else np.nan,
            "min": xf.min() if xf.size else np.nan,
            "q05": np.quantile(xf, 0.05) if xf.size else np.nan,
            "median": np.quantile(xf, 0.50) if xf.size else np.nan,
            "q95": np.quantile(xf, 0.95) if xf.size else np.nan,
            "max": xf.max() if xf.size else np.nan,
            "n_nonfinite": int((~finite).sum()),
        }
        lo, hi = bounds.get(name, (None, None))
        row["bound_lo"], row["bound_hi"] = lo, hi
        row["frac_below"] = (float((xf < lo).mean())
                             if lo is not None and xf.size else np.nan)
        row["frac_above"] = (float((xf > hi).mean())
                             if hi is not None and xf.size else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("variable")


def discrete_pmfs(df, varis):
    varis = _as_var_map(varis)
    out = {}
    for name in df.columns:
        var = varis.get(name)
        if var is None or not _is_discrete(var):
            continue
        counts = (df[name].astype(int).value_counts(normalize=True)
                  .reindex(range(len(var.values)), fill_value=0.0).sort_index())
        out[name] = pd.DataFrame({
            "state": range(len(var.values)),
            "label": [str(v) for v in var.values],
            "probability": counts.to_numpy(),
        }).set_index("state")
    return out


def _spread(values):
    v = np.asarray(values, dtype=float)
    return np.nanmax(v) - np.nanmin(v) if np.isfinite(v).any() else np.nan


def conditional_summaries(df, varis, max_states=6):
    """How much each low-cardinality discrete variable moves everything else.

    Correlation is blind to switch nodes: a variable that selects between two
    branches, or gates a quantity to zero, changes another variable's spread
    or branch without changing its rank ordering, so rho sits near zero while
    the dependence is total. ``effect_std`` catches exactly that case, which
    ``effect_mean`` alone misses.
    """
    varis = _as_var_map(varis)
    out = {}
    for g in df.columns:
        var = varis.get(g)
        if var is None or not _is_discrete(var) or len(var.values) > max_states:
            continue
        gi = df[g].astype(int)
        rows = {}
        for name in df.columns:
            if name == g:
                continue
            x = df[name].to_numpy(dtype=float)
            fin = np.isfinite(x)
            overall = x[fin].std(ddof=1) if fin.sum() > 1 else np.nan
            scale = overall if overall and np.isfinite(overall) and overall > 0 else np.nan

            row, means, stds = {}, [], []
            for s in range(len(var.values)):
                xs = x[(gi == s).to_numpy() & fin]
                mu = xs.mean() if xs.size else np.nan
                sd = xs.std(ddof=1) if xs.size > 1 else np.nan
                means.append(mu)
                stds.append(sd)
                row[f"mean|{var.values[s]}"] = mu
                row[f"std|{var.values[s]}"] = sd
            row["effect_mean"] = _spread(means) / scale
            row["effect_std"] = _spread(stds) / scale
            rows[name] = row

        table = pd.DataFrame(rows).T
        table.index.name = "affected"
        table["effect"] = table[["effect_mean", "effect_std"]].max(axis=1)
        out[g] = table.sort_values("effect", ascending=False)
    return out


def collect_flags(marginals, corr, conditionals=None, near_deterministic=0.999,
                  hidden_effect=0.5, weak_rho=0.1):
    flags = []
    for name, r in marginals.iterrows():
        if r["n_nonfinite"] > 0:
            flags.append({"severity": "error", "variable": name,
                          "issue": "non-finite samples",
                          "detail": f"{int(r['n_nonfinite'])} NaN/inf values"})
        if np.isfinite(r["std"]) and r["std"] == 0:
            flags.append({"severity": "warning", "variable": name,
                          "issue": "degenerate marginal",
                          "detail": "zero variance - is this meant to be constant?"})
        for side, key in (("below", "frac_below"), ("above", "frac_above")):
            frac = r[key]
            if frac is not None and np.isfinite(frac) and frac > 0:
                flags.append({"severity": "error" if frac > 0.01 else "warning",
                              "variable": name,
                              "issue": f"samples {side} declared bound",
                              "detail": f"{frac:.2%} of samples"})

    for _, p in ranked_pairs(corr).iterrows():
        if p["abs_rho"] >= near_deterministic:
            flags.append({"severity": "warning",
                          "variable": f"{p['var_a']} ~ {p['var_b']}",
                          "issue": "near-deterministic pair",
                          "detail": f"rho = {p['rho']:.4f}; the same quantity twice?"})

    for g, table in (conditionals or {}).items():
        for name, r in table.iterrows():
            eff = r.get("effect", np.nan)
            rho = (corr.loc[g, name]
                   if g in corr.index and name in corr.columns else np.nan)
            if (np.isfinite(eff) and eff >= hidden_effect
                    and np.isfinite(rho) and abs(rho) < weak_rho):
                kind = ("shifts the mean"
                        if r.get("effect_mean", 0) >= r.get("effect_std", 0)
                        else "changes the spread")
                flags.append({"severity": "info", "variable": f"{g} -> {name}",
                              "issue": "dependence invisible to correlation",
                              "detail": (f"{kind} by {eff:.2f} sd but rho = "
                                         f"{rho:.3f} - likely a switch or gate")})

    return pd.DataFrame(flags, columns=["severity", "variable", "issue", "detail"])


def run_sense_check(probs, varis, n_sample=20_000, bounds=None, seed=0,
                    batch_size=50_000, method="spearman"):
    """Forward-sample the prior and assemble the full diagnostic report."""
    varis = _as_var_map(varis)
    df = forward_sample_all(probs, n_sample=n_sample, batch_size=batch_size,
                            seed=seed)
    marg = summarise_marginals(df, varis, bounds=bounds)
    conds = conditional_summaries(df, varis)
    # Rank correlation by default: deterministic nodes are routinely monotone
    # but not linear, and discrete states are ordinal by library convention.
    corr = df.replace([np.inf, -np.inf], np.nan).corr(method=method)
    return SenseCheckReport(samples=df, marginals=marg,
                            discrete_pmfs=discrete_pmfs(df, varis),
                            correlation=corr, conditionals=conds,
                            flags=collect_flags(marg, corr, conditionals=conds),
                            n_sample=n_sample)


def plot_report(report, varis, stem, max_vars=30):
    """Correlation heatmap plus a grid of marginal histograms.

    matplotlib is imported here rather than at module scope: it is not a
    dependency of tbnpy, so the rest of this module works without it.
    """
    import matplotlib.pyplot as plt

    varis = _as_var_map(varis)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    corr = report.correlation

    fig, ax = plt.subplots(figsize=(0.45 * len(corr) + 3, 0.45 * len(corr) + 2))
    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Spearman correlation, prior samples")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(stem.with_name(stem.name + "_corr.png"), dpi=200)
    plt.close(fig)

    cols = list(report.samples.columns)[:max_vars]
    ncol = 4
    nrow = int(np.ceil(len(cols) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             squeeze=False)
    flat = np.ravel(axes)
    for ax, name in zip(flat, cols):
        x = report.samples[name].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        var = varis.get(name)
        if var is not None and _is_discrete(var):
            K = len(var.values)
            ax.hist(x, bins=np.arange(K + 1) - 0.5, density=True, color="tab:blue")
            ax.set_xticks(range(K))
            ax.set_xticklabels([str(v) for v in var.values], fontsize=7)
        else:
            ax.hist(x, bins=50, density=True, color="gray")
        ax.set_title(name, fontsize=9)
    for ax in flat[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(stem.with_name(stem.name + "_marginals.png"), dpi=200)
    plt.close(fig)


# ==========================================================================
# Evidence validation
# ==========================================================================
@dataclass
class EvidenceIssue:
    node: str
    severity: str          # "error" | "warning"
    issue: str
    detail: str
    suggestions: list = field(default_factory=list)

    def __str__(self) -> str:
        head = f"[{self.severity}] {self.node}: {self.issue} -- {self.detail}"
        if self.suggestions:
            head += "\n  " + "\n  ".join(f"- {s}" for s in self.suggestions)
        return head


_DETERMINISTIC_ADVICE = [
    "observe it with a tolerance instead of an exact value "
    "(set 'tolerance' on the equation node, or condition on an interval)",
    "add an explicit observation node -- a noisy measurement of this "
    "quantity -- and condition on that instead",
    "discretise the node, so its states carry positive probability",
]


def _looks_deterministic(node) -> bool:
    """Whether conditioning on this node exactly has probability zero."""
    spec = getattr(node, "spec", None)
    if spec is not None and getattr(spec, "type", None) == "equation":
        return float(getattr(spec, "tolerance", 0.0)) == 0.0
    return False


def check_evidence(probs, varis, evidence) -> list:
    """Check a conditioning request before spending any sampling on it.

    Parameters
    ----------
    evidence:
        A DataFrame of evidence rows, or ``{node: value}``/``{node: [values]}``.

    Returns
    -------
    list[EvidenceIssue]
        Empty when the request is sound. Errors mean the posterior would be
        undefined, not merely poor.
    """
    varis = _as_var_map(varis)
    evidence = _as_evidence_frame(evidence)
    issues = []

    for node_name in evidence.columns:
        node = probs.get(node_name)
        if node is None:
            issues.append(EvidenceIssue(
                node_name, "error", "not a node in this model",
                f"known nodes: {', '.join(sorted(probs))}"))
            continue

        var = varis.get(node_name)
        values = evidence[node_name].to_numpy()

        if var is not None and _is_discrete(var):
            n_states = len(var.values)
            bad = [v for v in values
                   if not float(v).is_integer() or not 0 <= int(v) < n_states]
            if bad:
                issues.append(EvidenceIssue(
                    node_name, "error", "state index out of range",
                    f"{node_name!r} has {n_states} states (0..{n_states - 1}); "
                    f"got {bad[:5]}",
                    [f"use the state index, not the label; labels are "
                     f"{list(var.values)}"]))
            continue

        if _looks_deterministic(node):
            issues.append(EvidenceIssue(
                node_name, "error",
                "cannot condition on a deterministic continuous node",
                f"{node_name!r} is an exact function of its parents, so an "
                f"observed value has probability zero: every particle would "
                f"receive -inf and the posterior would be NaN",
                list(_DETERMINISTIC_ADVICE)))
            continue

        if var is not None and isinstance(var.values, tuple):
            low, high = var.values
            outside = [float(v) for v in values if not (low <= v <= high)]
            if outside:
                issues.append(EvidenceIssue(
                    node_name, "error", "value outside the declared support",
                    f"{node_name!r} is supported on [{low}, {high}]; "
                    f"got {outside[:5]}"))

    return issues


def _as_evidence_frame(evidence) -> pd.DataFrame:
    if isinstance(evidence, pd.DataFrame):
        return evidence
    if not evidence:
        return pd.DataFrame(index=[0])
    normalised = {k: (v if isinstance(v, (list, tuple, np.ndarray)) else [v])
                  for k, v in evidence.items()}
    return pd.DataFrame(normalised)


# ==========================================================================
# Weighted posterior
# ==========================================================================
def effective_sample_size(log_weights) -> torch.Tensor:
    """Kish effective sample size from unnormalised log weights.

    ``log_weights`` is ``(n_sample,)`` or ``(n_evi, n_sample)``. Rows whose
    weights are all ``-inf`` -- an impossible observation -- give 0.0 rather
    than NaN, so the caller can report the cause instead of a silent NaN.
    """
    lw = torch.as_tensor(log_weights, dtype=torch.float64)
    single = lw.dim() == 1
    if single:
        lw = lw.unsqueeze(0)

    out = torch.zeros(lw.shape[0], dtype=torch.float64)
    for i in range(lw.shape[0]):
        row = lw[i]
        if not torch.isfinite(row).any():
            continue
        norm = row - torch.logsumexp(row[torch.isfinite(row)], 0)
        w = norm.exp()
        w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
        denom = (w ** 2).sum()
        out[i] = (1.0 / denom) if denom > 0 else 0.0
    return out[0] if single else out


def weighted_quantile(values, weights, quantiles):
    """Quantiles of a weighted particle set."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not keep.any():
        return np.full(len(quantiles), np.nan)
    v, w = v[keep], w[keep]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cdf = np.cumsum(w) - 0.5 * w
    cdf /= w.sum()
    return np.interp(quantiles, cdf, v)


@dataclass
class PosteriorResult:
    """Weighted particles, their summaries, and the ESS that qualifies them."""

    samples: dict                 # node -> (n_evi, n_sample) numpy array
    log_weights: torch.Tensor     # (n_evi, n_sample)
    weights: np.ndarray           # (n_evi, n_sample), normalised
    ess: torch.Tensor             # (n_evi,)
    n_sample: int
    evidence: pd.DataFrame
    issues: list = field(default_factory=list)

    @property
    def ess_fraction(self):
        return (self.ess / self.n_sample).numpy()

    def is_degenerate(self, threshold=0.01):
        """Rows whose ESS has collapsed below ``threshold`` of the particles."""
        return self.ess_fraction < threshold

    def summary(self, varis=None, row=0) -> pd.DataFrame:
        """Weighted marginal summary for one evidence row."""
        varis = _as_var_map(varis) if varis is not None else {}
        w = self.weights[row]
        rows = []
        for name, arr in self.samples.items():
            x = np.asarray(arr[row], dtype=float)
            var = varis.get(name)
            q05, q50, q95 = weighted_quantile(x, w, [0.05, 0.50, 0.95])
            keep = np.isfinite(x) & (w > 0)
            mean = float((x[keep] * w[keep]).sum() / w[keep].sum()) if keep.any() else np.nan
            var_ = (float((w[keep] * (x[keep] - mean) ** 2).sum() / w[keep].sum())
                    if keep.any() else np.nan)
            rows.append({
                "variable": name,
                "kind": ("discrete" if var is not None and _is_discrete(var)
                         else "continuous" if var is not None else "?"),
                "mean": mean, "std": float(np.sqrt(var_)),
                "q05": q05, "median": q50, "q95": q95,
            })
        out = pd.DataFrame(rows).set_index("variable")
        return out

    def pmf(self, name, var, row=0) -> pd.DataFrame:
        """Weighted posterior pmf of a discrete variable."""
        x = np.asarray(self.samples[name][row], dtype=float).astype(int)
        w = self.weights[row]
        probs = np.zeros(len(var.values))
        for s in range(len(var.values)):
            probs[s] = w[x == s].sum()
        total = probs.sum()
        if total > 0:
            probs /= total
        return pd.DataFrame({"state": range(len(var.values)),
                             "label": [str(v) for v in var.values],
                             "probability": probs}).set_index("state")


def posterior(probs, query_nodes, evidence, n_sample=20_000, varis=None,
              seed=0, batch_size=50_000, validate=True) -> PosteriorResult:
    """Posterior particles for ``query_nodes`` given ``evidence``.

    ``inference.sample_evidence`` is likelihood weighting: it clamps the
    observed nodes, resamples everything else, and returns each node's
    log-density. The importance weight of a particle is the sum of those
    log-densities **over the evidence nodes only** -- the sampled nodes'
    densities are the proposal and cancel.

    Raises
    ------
    ValueError
        If ``validate`` and the evidence cannot be conditioned on at all.
        Pass ``validate=False`` to get the degenerate result anyway.
    """
    evidence = _as_evidence_frame(evidence)
    evidence_nodes = [c for c in evidence.columns]

    issues = []
    if varis is not None:
        issues = check_evidence(probs, varis, evidence)
        errors = [i for i in issues if i.severity == "error"]
        if errors and validate:
            raise ValueError(
                "this evidence cannot be conditioned on:\n"
                + "\n".join(str(e) for e in errors))

    if seed is not None:
        torch.manual_seed(seed)

    wanted = list(dict.fromkeys(list(query_nodes) + evidence_nodes))
    out = inference.sample_evidence(probs, wanted, n_sample, evidence,
                                    batch_size=batch_size)

    log_w = None
    for node in evidence_nodes:
        ps = out[node].ps.detach().cpu().to(torch.float64)
        log_w = ps if log_w is None else log_w + ps
    if log_w is None:                       # no evidence: the prior
        log_w = torch.zeros((len(evidence), n_sample), dtype=torch.float64)

    ess = effective_sample_size(log_w)

    weights = np.zeros(tuple(log_w.shape))
    for i in range(log_w.shape[0]):
        row = log_w[i]
        if torch.isfinite(row).any():
            norm = (row - torch.logsumexp(row[torch.isfinite(row)], 0)).exp()
            weights[i] = torch.where(torch.isfinite(norm),
                                     norm, torch.zeros_like(norm)).numpy()

    samples = {}
    for node in query_nodes:
        Cs = out[node].Cs.detach().cpu()
        samples[node] = Cs[:, :, 0].numpy()

    degenerate = [i for i, e in enumerate(ess.tolist()) if e == 0.0]
    for i in degenerate:
        issues.append(EvidenceIssue(
            ", ".join(evidence_nodes), "error",
            "no particle is compatible with this evidence",
            f"evidence row {i} gave every particle a log-weight of -inf, so "
            f"the posterior is undefined",
            list(_DETERMINISTIC_ADVICE)))

    return PosteriorResult(samples=samples, log_weights=log_w, weights=weights,
                           ess=ess, n_sample=n_sample, evidence=evidence,
                           issues=issues)
