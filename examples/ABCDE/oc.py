import torch
from torch.distributions import Normal

from tbnpy.probability import Probability

class OC(Probability):
    def __init__(self, childs, parents, device='cpu'):
        """
        childs: [OC variable]
        parents: [C variable]
        OC = C + Normal(0, 0.5^2)
        """
        super().__init__(childs, parents, device)

        # standard deviation is fixed
        self.std = torch.tensor(0.5, device=device)

    # --------------------------------------------------------------
    def form(self):
        return f"normal(loc={self.parents[0].name}, scale={self.std.item():g})"

    # --------------------------------------------------------------
    def sample(self, Cs_pars):
        """
        Cs_pars: tensor (N, 1)
            Cs_pars[:,0] = C value

        Returns:
            OC samples of shape (N,)
        """
        Cs_pars = Cs_pars.to(self.device)
        C_val = Cs_pars[:, 0]

        dist = Normal(C_val, self.std)
        Cs = dist.rsample()  # (N,)
        ps = dist.log_prob(Cs)  # (N,)

        return Cs, ps  

    # --------------------------------------------------------------
    def log_prob(self, Cs):
        """
        Cs: tensor (N, 2)
            Cs[:,0] = observed OC value 
            Cs[:,1] = C value

        Returns:
            log p(OC | C), tensor shape (N,)
        """
        Cs = Cs.to(self.device)

        OC_val  = Cs[:, 0]
        C_val = Cs[:, 1]

        dist = Normal(C_val, self.std)
        return dist.log_prob(OC_val)   # (N,)
