import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import math
import s1_define_model as define_model
import os
import json

class Lnpga:

    def __init__(self, edge_dict, mag_vars, device='cpu'):
        """
        edge_dict: dictionary for this edge, containing distances:
            edge['dist_s1_km'], edge['dist_s2_km'], ...
        mag_vars: dict of magnitude variables {M1: variable.Variable, M2: variable.Variable}
                  each var has .values giving magnitude levels
        """
        self.device = device
        self.edge = edge_dict
        self.mag_vars = mag_vars  # length 2
        self.sigma = 0.57

        # Extract distances to sources automatically
        # e.g., 'dist_s1_km', 'dist_s2_km'
        self.sources = sorted([k for k in edge_dict.keys() if k.startswith("dist_") and k.endswith("_km")])
        # Example: ['dist_s1_km', 'dist_s2_km']

        # Convert to tensor
        self.R = torch.tensor(
            [edge_dict[k] for k in self.sources], dtype=torch.float32, device=device
        )  # shape (n_sources,)

    # ---------------------------------------------------------
    # Mean ln(PGA) model
    # lnPGA = -0.152 + 0.859 M - 1.803 ln(R + 25)
    # ---------------------------------------------------------
    def mean_lnPGA(self, M):
        """
        M: tensor (n_sample,) magnitude
        Returns tensor (n_sample, n_sources)
        """
        R = self.R.unsqueeze(0)  # (1, n_sources)
        M = M.unsqueeze(1)       # (n_sample,1)

        mean = -0.152 + 0.859 * M - 1.803 * torch.log(R + 25.0)
        return mean  # (n_sample, n_sources)

    # ---------------------------------------------------------
    # Compute lnPGA for each source
    # ---------------------------------------------------------
    def sample(self, Cs_par):
        """
        Deterministic lnPGA calculation.
        Cs_par: (n_sample, 3)
            [:,0] = M1 state
            [:,1] = M2 state
            [:,2] = epsilon

        Returns:
            lnPGA: (n_sample, n_sources)
        """
        Cs_par = Cs_par.to(self.device)
        M1_state = Cs_par[:, 0].long()
        M2_state = Cs_par[:, 1].long()
        eps = Cs_par[:, 2]

        # Convert magnitude states to actual numeric values
        M1 = torch.tensor([float(self.mag_vars['m1'].values[i]) for i in M1_state], device=self.device)
        M2 = torch.tensor([float(self.mag_vars['m2'].values[i]) for i in M2_state], device=self.device)

        # Deterministic lnPGA from two magnitudes
        mean1 = self.mean_lnPGA(M1)  # (n, n_sources)
        mean2 = self.mean_lnPGA(M2)

        lnPGA = mean1 + mean2 + self.sigma * eps.unsqueeze(1)
        return lnPGA

    # Log-likelihood
    def log_prob(self, Cs):
        """
        Deterministic log-likelihood:
        log(1) if Cs matches deterministic lnPGA,
        log(0) otherwise.

        Cs: (n_sample, n_sources + 3)
        """
        Cs = Cs.to(self.device)
        n_sample, total_cols = Cs.size()
        n_sources = total_cols - 3

        # Extract observed lnPGA
        lnPGA_obs = Cs[:, :n_sources]

        # Extract parameters
        Cs_par = Cs[:, n_sources:]   # (n_sample, 3)

        # Deterministic prediction
        lnPGA_pred = self.sample(Cs_par)

        # Comparison mask
        match = torch.isclose(lnPGA_obs, lnPGA_pred, atol=1e-6).all(dim=1)

        # log(1) = 0, log(0) = -inf
        logp = torch.where(match,
                           torch.zeros(n_sample, device=self.device),
                           torch.full((n_sample,), float('-inf'), device=self.device))

        return logp

    
if __name__ == "__main__":
    varis = define_model.define_variables()

    BASE = os.path.dirname(os.path.abspath(__file__))
    edges = json.load(open(os.path.join(BASE, 'edges.json')))

    ln_pga1 = Lnpga(edges['e01'], mag_vars={'m1': varis['m1'], 'm2': varis['m2']}, device='cpu')

    Cs_par = torch.tensor([
        [0, 1, 0.5],
        [2, 0, -1.0],
    ], dtype=torch.float32)

    lnPGA_samples = ln_pga1.sample(Cs_par)
    print("lnPGA samples:\n", lnPGA_samples)

    log_probs = ln_pga1.log_prob(lnPGA_samples)