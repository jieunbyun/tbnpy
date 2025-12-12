import torch
from torch.distributions import Normal

class C:
    def __init__(self, childs, parents, sigma=0.6, device="cpu"):
        """
        C | A,B ~ Normal(A + B, sigma^2)

        childs  : [Variable C]
        parents : [Variable A, Variable B]
        sigma   : fixed noise std (float)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        self.sigma = float(sigma)

        # parent variables
        self.A = parents[0]
        self.B = parents[1]

        # value lookup tables
        self.A_values = torch.tensor(
            self.A.values, dtype=torch.float32, device=device
        )
        self.B_values = torch.tensor(
            self.B.values, dtype=torch.float32, device=device
        )

    # ------------------------------------------------------------------
    def sample(self, Cs_pars):
        """
        Cs_pars : (N, 2)
            Cs_pars[:,0] = A index
            Cs_pars[:,1] = B index

        Returns
        -------
        Cs   : (N,) sampled C values
        logp : (N,) log p(C | A,B)
        """
        Cs_pars = Cs_pars.to(self.device).long()

        A_idx = Cs_pars[:, 0]
        B_idx = Cs_pars[:, 1]

        mean = self.A_values[A_idx] + self.B_values[B_idx]
        std = torch.full_like(mean, self.sigma)

        dist = Normal(mean, std)
        Cs = dist.sample()
        logp = dist.log_prob(Cs)

        return Cs, logp

    # ------------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs : (N, 3)
            Cs[:,0] = C value
            Cs[:,1] = A index
            Cs[:,2] = B index

        Returns
        -------
        log p(C | A,B) : (N,)
        """
        Cs = Cs.to(self.device)

        C_val = Cs[:, 0]
        A_idx = Cs[:, 1].long()
        B_idx = Cs[:, 2].long()

        mean = self.A_values[A_idx] + self.B_values[B_idx]
        std = torch.full_like(mean, self.sigma)

        dist = Normal(mean, std)
        return dist.log_prob(C_val)
