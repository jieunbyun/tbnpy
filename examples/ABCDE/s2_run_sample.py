import os, sys
from pathlib import Path
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = Path(__file__).parent / "results"
sys.path.append(BASE)

repo_root = os.path.abspath(os.path.join(BASE, "../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tbnpy import inference, adaptiveMH
from s1_define_model import define_variables, define_probs

"""
Overall structure:
s2_run_sample.py
├─ define variables & probs   (already done)
├─ define evidence
├─ forward sampling (initialisation)
├─ adaptive MH run
├─ posterior extraction
└─ plotting
"""

def define_evidence(n_evi=10, seed=123):
    """
    Evidence DataFrame: shape (n_evi, n_evidence_vars)

    OC ~ Normal(0, 0.15)
    """
    rng = np.random.default_rng(seed)

    evidence = pd.DataFrame({
        "OC": rng.normal(loc=0.1, scale=0.7, size=n_evi)
    })

    return evidence

def sample_prior(probs, variables, n_sample=5000):
    """
    Sample prior for all variables using forward sampling.

    Parameters
    ----------
    probs : dict
        BN probability objects
    variables : dict or list
        Variable objects
    n_sample : int
        Number of prior samples

    Returns
    -------
    dict[var_name -> np.ndarray]
        Each array has shape (n_sample,)
    """
    # one dummy evidence row (no conditioning)
    evidence = pd.DataFrame(index=[0])

    # ensure variable list
    if isinstance(variables, dict):
        var_list = list(variables.values())
    else:
        var_list = list(variables)

    query_nodes = [v.name for v in var_list]

    probs_copy = inference.sample_evidence(
        probs=probs,
        query_nodes=query_nodes,
        n_sample=n_sample,
        evidence_df=evidence,
    )

    # container for prior samples
    prior = {}

    for prob in probs_copy.values():
        Cs = prob.Cs  # shape (1, n_sample, dim)
        for j, child_var in enumerate(prob.childs):
            name = child_var.name
            prior[name] = Cs[0, :, j].detach().cpu().numpy()

    return prior

def forward_initialise(probs, latent_vars, evidence, n_chain):
    """
    Use forward sampling to initialise MCMC chains.
    """
    probs_copy = inference.sample_evidence(
        probs=probs,
        query_nodes=[v.name for v in latent_vars],
        n_sample=n_chain,
        evidence_df=evidence,
    )
    return probs_copy

def run_mcmc(probs, varis, evidence, update_blocks, burnin=200, n_chain=5000, n_iter=2000, progress_every=100):
    sampler = adaptiveMH.HybridAdaptiveMH(
        probs=probs,
        variables=list(varis.values()),
        evidence_df=evidence,
        n_chain=n_chain,
        adapt=adaptiveMH.AdaptConfig(
            burnin=burnin,
            gamma=0.6,
            target_accept=0.234,
            alpha=0.5,
        ),
    )

    # --- Initialise from forward samples ---
    probs_copy = forward_initialise(
        probs,
        sampler.latent_vars,
        evidence,
        n_chain,
    )
    sampler.init_state_from_forward_samples(probs_copy)

    # --- Run MCMC ---
    out = sampler.run(
        n_iter=n_iter,
        store_every=10,   # thin
        update_blocks=update_blocks,
        progress_every=progress_every,
    )

    return sampler, out

def extract_posterior(sampler):
    """
    Returns dict[var_name -> 1D np.ndarray]
    (all evidence rows and chains flattened)
    """
    posterior = {}

    for v in sampler.latent_vars:
        x = sampler.state[v.name]  # (n_evi, n_chain)
        posterior[v.name] = x.detach().cpu().numpy().reshape(-1)

    return posterior


import numpy as np
import matplotlib.pyplot as plt

def plot_prior_vs_posterior(prior, posterior, var, bins=60, fname: str = None):
    """
    Plot prior vs posterior for one variable.

    Parameters
    ----------
    prior : dict[str, np.ndarray]
        Prior samples for all variables
    posterior : dict[str, np.ndarray]
        Posterior samples for all variables
    var : Variable
        tbnpy Variable object
    bins : int
        Number of bins for continuous histograms
    fname : str, optional
        File name to save the plot (saved in RESULTS folder). If None, the plot is not saved.
    """
    name = var.name

    if name not in prior:
        raise KeyError(f"Variable '{name}' not found in prior samples.")
    if name not in posterior:
        raise KeyError(f"Variable '{name}' not found in posterior samples.")

    x_prior = prior[name]
    x_post = posterior[name]

    plt.figure(figsize=(5, 4))

    # Discrete variable
    if isinstance(var.values, list):
        K = len(var.values)
        bins_disc = np.arange(K + 1) - 0.5

        plt.hist(
            x_prior,
            bins=bins_disc,
            density=True,
            alpha=0.5,
            label="Prior",
            color="gray",
        )

        plt.hist(
            x_post,
            bins=bins_disc,
            density=True,
            alpha=0.6,
            label="Posterior",
            color="tab:blue",
        )

        plt.xticks(range(K), var.values)
        plt.ylabel("Probability")

    # Continuous variable
    else:
        plt.hist(
            x_prior,
            bins=bins,
            density=True,
            alpha=0.5,
            label="Prior",
            color="gray",
        )

        plt.hist(
            x_post,
            bins=bins,
            density=True,
            alpha=0.6,
            label="Posterior",
            color="tab:blue",
        )

        plt.ylabel("Density")

    plt.xlabel(name)
    plt.legend()
    plt.tight_layout()
    if fname is not None:
        plt.savefig(RESULTS / fname, dpi=300)


if __name__ == "__main__":
    device = "cuda" if os.environ.get("USE_CUDA", "0") == "1" else "cpu"

    varis = define_variables()
    probs = define_probs(varis, device=device)

    n_evi = 5
    evidence = define_evidence(n_evi=n_evi)

    n_chain = 100
    n_iter = 30_000
    burnin = 200

    query_varis = {v: varis[v] for v in ['A', 'B', 'C', 'OC']} # only infer OC's ancestors
    query_probs = {k: v for k, v in probs.items() if any(c.name in query_varis for c in v.childs)}
    update_blocks = ['A', 'B', 'C'] 

    sampler, out = run_mcmc(
        query_probs,
        query_varis,
        evidence,
        update_blocks = update_blocks,
        burnin=burnin,
        n_chain=n_chain,
        n_iter=n_iter,
        progress_every = 100
    )

    prior = sample_prior(probs, varis, n_sample=10_000)
    posterior = extract_posterior(sampler)
    for _, v in query_varis.items():
        if v.name in evidence.columns:
            continue  # skip evidence variables
        plot_prior_vs_posterior(prior, posterior, v, fname=r"plot_" + v.name + f"_{n_evi}_evi_{burnin}_burnin_{n_chain}_chains_{n_iter}_iters" + ".png")

    print("Acceptance rates:")
    for k, v in out["accept_rate"].items():
        print(f"  {k}: {v:.3f}")
