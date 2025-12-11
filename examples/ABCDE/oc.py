import torch
from torch.distributions import Normal

class OC:
    def __init__(self, childs, parents, device='cpu'):
        """
        childs: [OC variable]
        parents: [C variable]
        OC = C + Normal(0, 1)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

        # standard deviation is fixed to 1.0
        self.std = torch.tensor(1.0, device=device)

    # --------------------------------------------------------------
    def sample(self, OCs_par):
        """
        OCs_par: tensor (N, 1)
            OCs_par[:,0] = C value

        Returns:
            OC samples of shape (N,)
        """
        OCs_par = OCs_par.to(self.device)
        C_val = OCs_par[:, 0]

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
