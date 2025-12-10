import torch
import json

import os
BASE = os.path.dirname(os.path.abspath(__file__))

class Pga_residuals:

    def __init__(self, device='cpu'):
        self.device = device

        # Load data
        with open(os.path.join(BASE, 'nodes.json')) as f:
            self.nodes = json.load(f)

        with open(os.path.join(BASE, 'edges.json')) as f:
            self.edges = json.load(f)

        with open(os.path.join(BASE, 'eq_sources.json')) as f:
            self.sources = json.load(f)

        # Compute edge centers
        self.edge_centers = self.compute_edge_centers()

        # Compute correlation matrix among edges
        self.Ccorr = self.compute_spatial_correlation()

    # 1. Compute the center (midpoint) of each edge
    def compute_edge_centers(self):
        """
        Returns tensor of shape (n_edges, 2), each row = (x_center, y_center)
        """
        centers = []

        for eid, e in self.edges.items():
            n1 = self.nodes[e["from"]]
            n2 = self.nodes[e["to"]]

            cx = 0.5 * (n1["x"] + n2["x"])
            cy = 0.5 * (n1["y"] + n2["y"])
            centers.append([cx, cy])

        return torch.tensor(centers, dtype=torch.float32, device=self.device)


    # 2. Jayaram & Baker (2009) spatial correlation function for PGA
    @staticmethod
    def rho_pga(h):
        """
        Spatial correlation of ln(PGA) between two sites separated by h (km).
        """
        a = 40.7   # km
        b = 0.923
        return torch.exp(- (h / a)**b)


    # 3. Compute pairwise correlation matrix between all edge centers
    def compute_spatial_correlation(self):
        """
        Create correlation matrix C where C[i,j] = rho(distance(center_i, center_j))
        """
        centers = self.edge_centers  # (n_edges, 2)
        n = centers.size(0)

        # Pairwise Euclidean distance matrix
        diff = centers.unsqueeze(1) - centers.unsqueeze(0)   # (n,n,2)
        dist = torch.sqrt((diff ** 2).sum(dim=2))            # (n,n)

        # Apply spatial correlation model
        C = self.rho_pga(dist)

        return C
    
    def sample(self, n_sample):
        """
        Draws correlated PGA residual samples.
        
        Returns:
            samples: (n_sample, n_edges) tensor
        """
        C = self.Ccorr.to(self.device)          # correlation matrix (n_edges, n_edges)
        n_edges = C.size(0)

        # Mean vector (zero mean residuals)
        mean = torch.zeros(n_edges, device=self.device)

        # Ensure C is positive definite for Cholesky
        # add a tiny jitter just in case of numerical issues
        jitter = 1e-6 * torch.eye(n_edges, device=self.device)
        C_pd = C + jitter

        # Cholesky factorization
        L = torch.linalg.cholesky(C_pd)      # (n_edges, n_edges)

        # Standard normal samples
        z = torch.randn(n_sample, n_edges, device=self.device)

        # Correlated samples: mean + z * Lᵀ
        samples = z @ L.T

        return samples
    
    def log_prob(self, Cs):
        """
        Computes log-likelihood of samples Cs under the multivariate
        Gaussian defined by correlation matrix self.Ccorr.

        Inputs:
            Cs: (n_sample, n_edges) tensor

        Returns:
            logp: (n_sample,) tensor of log probabilities
        """
        C = self.Ccorr.to(self.device)
        n_edges = C.size(0)

        # Mean vector for residuals (zero-centered)
        mean = torch.zeros(n_edges, device=self.device)

        # Add jitter to ensure covariance matrix is Positive Definite
        jitter = 1e-6 * torch.eye(n_edges, device=self.device)
        C_pd = C + jitter 

        # Build multivariate normal distribution
        mvn = torch.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=C_pd
        )

        # log_prob returns shape (n_sample,)
        logp = mvn.log_prob(Cs.to(self.device))

        return logp

    
if __name__ == "__main__":
    R = Pga_residuals(device='cpu')
    print(R.edge_centers)   # midpoints of edges
    print(R.Ccorr)          # correlation matrix

    samples = R.sample(n_sample=6)
    print(samples)          # 6 samples of correlated PGA residuals

    logp = R.log_prob(samples)
    print(logp)             # log probabilities of the samples