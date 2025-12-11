import torch

class E:
    def __init__(self, childs, parents, device='cpu'):
        """
        childs: list [E] (unused but consistent with your structure)
        parents: list [C, D]
            C: continuous-valued parent (tensor-like values for samples)
            D: binary parent (0 or 1)
        """
        self.childs = childs
        self.parents = parents
        self.device = device

    # ------------------------------------------------------------------
    def sample(self, Cs_pars):
        """
        Cs_pars: (N, 2)
            Cs_pars[:,0] = C value   (float)
            Cs_pars[:,1] = D index   (0 or 1)
        
        Returns:
            E samples (N,)
        """
        Cs_pars = Cs_pars.to(self.device)

        C_val = Cs_pars[:, 0]
        D_idx = Cs_pars[:, 1].long()

        # E = C if D==0, else 0
        Cs = torch.where(D_idx == 0, C_val, torch.zeros_like(C_val))

        # deterministic function, i.e. P(E | C, D) = 1
        n_sample = Cs_pars.shape[0]
        ps = torch.log(torch.ones(n_sample,)).to(self.device)

        return Cs, ps

    # ------------------------------------------------------------------
    def log_prob(self, Es):
        """
        Es: shape (N, 3)
            Es[:,0] = E value
            Es[:,1] = C value 
            Es[:,2] = D state (0 or 1) 
        
        Returns:
            log p(E | C, D) of shape (N,)
        """

        Es = Es.to(self.device)

        E_val = Es[:, 0]
        C_val = Es[:, 1]
        D_idx = Es[:, 2].long()

        # Deterministic rule: valid_E = C if D==0 else 0
        expected_E = torch.where(D_idx == 0, C_val, torch.zeros_like(C_val))

        # Valid if E_val == expected_E
        is_valid = (E_val == expected_E)

        # log 1 = 0 for valid, log 0 = -inf for invalid
        logp = torch.where(is_valid, torch.zeros_like(E_val), torch.full_like(E_val, -float("inf")))

        return logp
