import torch
from torch.distributions import Normal

class C:
    def __init__(self, childs, parents, Cs=[], ps=[], device='cpu'):
        """
        childs: a list of child variables [C]
        parents: list [A, B] where A and B are variable.Variable
        """
        self.childs = childs
        self.parents = parents
        self.Cs = Cs
        self.ps = ps
        self.device = device

        self.A = parents[0]
        self.B = parents[1]
        
        # Convert parent values to tensors
        self.A_values = torch.tensor(self.A.values, dtype=torch.float32, device=device)
        self.B_values = torch.tensor(self.B.values, dtype=torch.float32, device=device)

    def sample(self, Cs_par):
        """
        Cs_par: tensor of shape (N, 2)
                Cs_par[:,0] = indices for A
                Cs_par[:,1] = indices for B

        Returns:
            C_sample:   tensor (N,)
            log_prob:   tensor (N,) where log_prob = log p(C | A, B)
        """

        # Ensure tensor on correct device
        Cs_par = Cs_par.to(self.device).long()

        # Extract mean and std dev using indexing
        a_idx = Cs_par[:, 0]   # shape: (N,)
        b_idx = Cs_par[:, 1]   # shape: (N,)

        means = self.A_values[a_idx]         # μ values
        stds  = self.B_values[b_idx].abs()   # σ values must be positive

        # Normal distribution sampling
        dist = Normal(means, stds)
        # sample C
        Cs = dist.rsample()

        # compute log P(C | A, B)
        logp = dist.log_prob(Cs)

        return Cs, logp  # shape: (N,), (N,)

    def log_prob(self, Cs):
        """
        Cs: shape (N, 3)
            Cs[:,0] = C value
            Cs[:,1] = A index
            Cs[:,2] = B index

        Returns:
            log p(C | A, B), shape (N,)
        """
        Cs = Cs.to(self.device)
        A_idx = Cs[:, 1].long()
        B_idx = Cs[:, 2].long()
        C_val = Cs[:, 0]

        means = self.A_values[A_idx]            # (N,)
        stds  = self.B_values[B_idx].abs()

        dist = Normal(means, stds)
        return dist.log_prob(C_val)             # (N,)