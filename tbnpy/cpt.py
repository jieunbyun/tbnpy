import numpy as np
import torch

from tbnpy.variable import Variable

class Cpt(object):
    '''Defines the conditional probability Tensor (CPT).
    CPT is based on the same concept as CPM in Ref: Byun et al. (2019). Matrix-based Bayesian Network for
    efficient memory storage and flexible inference. 
    Reliability Engineering & System Safety, 185, 533-545.
    The only different is it's based on tensor operation, using pytorch.
    
    Attributes:
        childs (list): list of instances of Variable objects.
        parents (list): list of instances of Variable objects.
        C (array_like): event matrix.
        p (array_like): probability vector for the events of corresponding rows in C.
        Cs (array_like): event matrix of samples.
        ps (array_like): sampling probability vector for the events of corresponding rows in Cs.

    Notes:
        C and p have the same number of rows.
        Cs and ps have the same number of rows.
    '''

    def __init__(self, childs, parents, C=[], p=[], Cs=[], ps=[], device="cpu"):

        self.device = device

        self.childs = childs
        self.parents = parents
        self.C = C
        self.p = p
        self.Cs = Cs
        self.ps = ps

    # Magic methods
    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if isinstance(other, Cpt):
            return (
                self._childs == other._childs and
                self._parents == other._parents and
                self._C == other._C and
                self._p == other._p and
                self._Cs == other._Cs and
                self._ps == other._ps
            )
        else:
            return False

    def __repr__(self):
        details = [
            f"{self.__class__.__name__}(childs={get_names(self.childs)},",
            f"parents={get_names(self.parents)},",
            f"C={self.C},",
            f"p={self.p},",
        ]

        if self._Cs.size:
            details.append(f"Cs={self._Cs},")
        if self._ps.size:
            details.append(f"ps={self._ps},")

        details.append(")")
        return "\n".join(details)

    # Properties
    @property
    def childs(self):
        return self._childs

    @childs.setter
    def childs(self, value):
        assert isinstance(value, list), 'childs must be a list of Variable'
        assert all([isinstance(x, Variable) for x in value]), 'childs must be a list of Variable'
        self._childs = value

    @property
    def parents(self):
        return self._parents
    
    @parents.setter
    def parents(self, value):
        assert isinstance(value, list), 'parents must be a list of Variable'
        assert all([isinstance(x, Variable) for x in value]), 'parents must be a list of Variable'
        self._parents = value

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        if value is None or (isinstance(value, list) and value == []):
            self._C = torch.empty((0,0), dtype=torch.int64)
            return

        # Convert list/np/tensor → torch.Tensor(int64)
        value = self._to_tensor(value, dtype=torch.int64)

        # shape corrections
        if value.ndim == 1:
            value = value.unsqueeze(1)

        # validate
        if value.numel() > 0:
            assert value.shape[1] == len(self._childs) + len(self._parents), \
                "C must have same number of columns as variables"

        self._C = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if value is None or (isinstance(value, list) and value == []):
            self._p = torch.empty((0,1), dtype=torch.float32)
            return

        value = self._to_tensor(value, dtype=torch.float32)

        # reshape 1D → column
        if value.ndim == 1:
            value = value.unsqueeze(1)

        if self._C.numel() > 0:
            assert value.shape[0] == self._C.shape[0], \
                "p must match number of rows in C"

        self._p = value

    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        if value is None or (isinstance(value, list) and value == []):
            self._Cs = torch.empty((0,0), dtype=torch.int64)
            return

        value = self._to_tensor(value, dtype=torch.int64)

        if value.ndim == 1:
            value = value.unsqueeze(1)

        if value.numel() > 0:
            assert value.shape[1] == len(self._childs) + len(self._parents), \
                "Cs must have same number of columns as variables"

        self._Cs = value

    @property
    def ps(self):
        return self._ps

    @ps.setter
    def ps(self, value):
        if value is None or (isinstance(value, list) and value == []):
            self._ps = torch.empty((0,1), dtype=torch.float32)
            return

        value = self._to_tensor(value, dtype=torch.float32)

        if value.ndim == 1:
            value = value.unsqueeze(1)

        if self._Cs.numel() > 0:
            assert value.shape[0] == self._Cs.shape[0], \
                "ps must match number of rows in Cs"

        self._ps = value

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, value):
        if isinstance(value, torch.device):
            self._device = value
        else:
            self._device = torch.device(value)
    
    def _to_tensor(self, x, dtype=torch.float32):
        """Convert list / numpy array / tensor → torch.Tensor."""
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype)
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=dtype, device=self.device)
        if isinstance(x, list):
            return torch.tensor(x, dtype=dtype, device=self.device)

        supported = (list, np.ndarray, torch.Tensor)
        raise TypeError(
            f"Unsupported data type: {type(x)}. "
            f"Expected one of {supported}."
        )

    def _get_C_binary(self):
        """
        Vectorised conversion from composite C-matrix to binary 3D tensor:
            (n_events × n_variables × max_basic)
        """
        # Step 1: Ensure C is a torch tensor
        if isinstance(self.C, torch.Tensor):
            C_tensor = self.C.to(self.device)
        else:
            C_tensor = torch.tensor(self.C, dtype=torch.long, device=self.device)

        variables = self._childs + self._parents
        n_events, n_variables = C_tensor.shape

        # Step 2: determine max_basic across all variables
        basic_sizes = [len(v.values) for v in variables]
        max_basic = max(basic_sizes)

        # Step 3: allocate output
        Cb = torch.zeros((n_events, n_variables, max_basic),
                        dtype=torch.int8,
                        device=self.device)

        # Step 4: vectorised Bst→Bvec for each variable (loop only over variables)
        for j, var in enumerate(variables):
            # Extract composite state indices for variable j
            Cst = C_tensor[:, j]     # shape (n_events,)

            # Convert to binary (vectorised)
            Cbin = var.get_Cst_to_Cbin(Cst)     # shape (n_events, n_basic)

            # Fill output (pad if variable has fewer than max_basic states)
            n_basic = len(var.values)
            Cb[:, j, :n_basic] = Cbin
        return Cb
    
    def expand_and_check_compatibility(self, C_binary, samples):
        """
        C_binary: (n_event, n_var, max_state)
        samples:  (n_sample, n_parents, max_par_state)
        p:        (n_event,) OR (n_event, 1)
        
        Returns:
            p_exp: (n_sample, n_event) with incompatible event-sample pairs set to zero.

        Notes:
            Cb_exp:   (n_event, n_sample, n_var, max_global_state)
            Sm_exp:   (n_event, n_sample, n_parents, max_global_state)
            mask:     (n_sample, n_event) 1 if compatible, 0 otherwise
        """
        device = C_binary.device
        p = self.p

        n_event, n_var, max_state = C_binary.shape
        n_sample, n_parents, max_par_state = samples.shape
        
        max_global = max(max_state, max_par_state)

        # ---------------------------------------------------------
        # 1. Pad C_binary and samples to same global state dimension
        # ---------------------------------------------------------
        if max_global > max_state:
            pad = max_global - max_state
            C_binary = torch.nn.functional.pad(C_binary, (0, pad))

        if max_global > max_par_state:
            pad = max_global - max_par_state
            samples = torch.nn.functional.pad(samples, (0, pad))

        # Shapes after padding:
        #   C_binary: (n_event,  n_var,     max_global)
        #   samples:  (n_sample, n_parents, max_global)

        # ---------------------------------------------------------
        # 2. Broadcast to (n_event, n_sample, ...)
        # ---------------------------------------------------------
        # C_binary → expand along samples
        Cb_exp = C_binary.unsqueeze(1)               # (n_event, 1, n_var, max_global)
        Cb_exp = Cb_exp.expand(n_event, n_sample, n_var, max_global)

        # samples → expand along events
        Sm_exp = samples.unsqueeze(0)                # (1, n_sample, n_parents, max_global)
        Sm_exp = Sm_exp.expand(n_event, n_sample, n_parents, max_global)

        # ---------------------------------------------------------
        # 3. Compatibility check: multiply parent parts
        # ---------------------------------------------------------
        # Extract parent part from C_binary-expanded
        start = len(self.childs)
        end = start + len(self.parents)
        Cb_parent = Cb_exp[:, :, start:end, :]      # (n_event, n_sample, n_parents, max_global)

        multiplied = Cb_parent * Sm_exp              # same shape

        # For each (event, sample, parent):
        #    if all states = 0 → incompatible parent
        parent_zero_mask = multiplied.sum(dim=-1) == 0   # (n_event, n_sample, n_parents)

        # If ANY parent is incompatible → incompatible event-sample pair
        incompatible_mask = parent_zero_mask.any(dim=-1)   # (n_event, n_sample)

        # compatibility mask: 1 if compatible, 0 if not
        compatibility_mask = (~incompatible_mask).float()   # (n_event, n_sample)

        # ---------------------------------------------------------
        # 4. Expand probabilities and apply mask
        # ---------------------------------------------------------
        if p.dim() == 1:
            p_exp = p.unsqueeze(1).expand(n_event, n_sample)
        else:
            p_exp = p.expand(n_event, n_sample)

        # Set incompatible probabilities to zero
        p_exp = p_exp * compatibility_mask

        return p_exp.T
    
    def sample_from_p_exp(self, p_exp):
        """
        p_exp: (n_samples, n_events) probability matrix (not normalized)

        Returns:
            Cs: (n_samples, n_childs) sampled child composite states
            event_idx: (n_samples,) sampled event indices

        Notes:
            C: (n_events, n_vars) stored in self.C
            n_childs: number of child variables
        """
        device = p_exp.device
        n_samples = p_exp.size(0)
        n_childs = len(self.childs)

        # Ensure C is torch tensor on right device
        C = self.C
        if not torch.is_tensor(C):
            C = torch.tensor(C, dtype=torch.long)
        C = C.to(device)

        # 1. Normalize probabilities across events
        p_norm = p_exp / (p_exp.sum(dim=1, keepdim=True) + 1e-15)

        # 2. Draw uniform random numbers
        u = torch.rand(n_samples, 1, device=device)

        # 3. Compute cumulative distribution function
        cdf = p_norm.cumsum(dim=1)

        # 4. Vectorised sampling using searchsorted
        event_idx = torch.searchsorted(cdf, u, right=False).squeeze(1)

        # 5. Retrieve only child composite states
        Cs = C[event_idx, :n_childs]

        return Cs, event_idx
    
    def sample(self, n_sample=None, Cs_pars=None, batch_size=100_000):
        """
        Samples from this CPT.

        Case 1: No parent nodes
            -> n_sample must be provided
            -> returns (n_sample, n_childs)

        Case 2: With parent nodes
            -> Cs_pars provided as composite parent samples: (n_samples, n_parents)
            -> returns (n_samples, n_childs)

        Uses batching to avoid constructing large tensors at once.
        """
        has_parents = len(self.parents) > 0

        # ===========================================
        # CASE 1 — No parents
        # ===========================================
        if not has_parents:
            assert n_sample is not None, \
                "For CPT without parents, n_sample must be provided."

            # p: (n_events, 1) or (n_events,)
            p = self.p.squeeze()                 # shape: (n_events,)
            p = p / (p.sum() + 1e-12)

            # CDF: (n_events,)
            cdf = p.cumsum(dim=0)

            # Random uniforms: (n_sample,)
            u = torch.rand(n_sample, device=p.device)

            # Sample event indices: (n_sample,)
            event_idx = torch.searchsorted(cdf, u)

            # Retrieve child composite states
            C = self.C.to(p.device)
            n_childs = len(self.childs)

            Cs = C[event_idx, :n_childs]
            ps = p[event_idx]
            return Cs, ps

        # ===========================================
        # CASE 2 — Parents exist
        # Cs_pars: composite parent states (n_samples, n_parents)
        # ===========================================
        assert Cs_pars is not None, \
            "For CPT with parents, Cs_pars must be provided."

        device = self.p.device
        n_samples_total = Cs_pars.size(0)
        n_childs = len(self.childs)

        # Convert parent composite states to binary
        parent_vars = self.parents
        bin_list = []
        for row in Cs_pars:
            parent_bin = []
            for j, v in enumerate(parent_vars):
                parent_bin.append(v.get_Cst_to_Cbin(row[j]).unsqueeze(0))
            parent_bin = torch.cat(parent_bin, dim=0)  # (n_parents, max_basic)
            bin_list.append(parent_bin)

        # shape (n_samples_total, n_parents, max_basic)
        samples_bin = torch.stack(bin_list, dim=0).to(device)

        # Container for output samples
        Cs_out = []
        ps_out = []

        # Batch over samples
        for start in range(0, n_samples_total, batch_size):
            end = min(start + batch_size, n_samples_total)

            # Slice batch
            samples_bin_batch = samples_bin[start:end]   # (batch, n_parents, max_basic)

            # Compatibility check: returns p_exp shape (batch, n_events)
            p_exp = self.expand_and_check_compatibility(
                self._get_C_binary(),    # event binary matrix
                samples_bin_batch        # parent binary batch
            )

            # Sample child states from conditional probability
            Cs_batch, event_idx_batch = self.sample_from_p_exp(p_exp)
            p_norm = p_exp / (p_exp.sum(dim=1, keepdim=True) + 1e-15)
            ps_batch = p_norm[torch.arange(p_norm.size(0)), event_idx_batch]      # (batch, n_childs)

            Cs_out.append(Cs_batch)
            ps_out.append(ps_batch)

        # Stack all batches
        Cs_all = torch.cat(Cs_out, dim=0)
        ps_all = torch.cat(ps_out, dim=0)
        return Cs_all, ps_all
    
    def log_prob(self, Cs, batch_size=100_000):
        """
        Computes log-likelihood of given full composite states under this CPT.

        Parameters
        ----------
        Cs : (n_samples, n_vars)
            Composite states for ALL variables in this CPT.
            The ordering must match self.C (childs first, then parents).

        Returns
        -------
        log_ps : (n_samples,)
            Log-likelihood of each sample.
        """

        device = self.p.device
        Cs = torch.as_tensor(Cs, device=device, dtype=torch.long)

        n_samples_total = Cs.size(0)
        n_childs = len(self.childs)
        n_parents = len(self.parents)

        has_parents = n_parents > 0

        # --------------------------------------------
        # Split Cs into child and parent composite states
        # --------------------------------------------
        Cs_childs = Cs[:, :n_childs]                                # (n_samples, n_childs)
        Cs_pars   = Cs[:, n_childs:n_childs + n_parents]            # (n_samples, n_parents)

        # --------------------------------------------------------
        # CASE 1 — No parents: direct lookup
        # --------------------------------------------------------
        if not has_parents:
            C = torch.as_tensor(self.C, device=device, dtype=torch.long)

            # probabilities
            p = self.p.squeeze()  # (n_events,)
            p = p / (p.sum() + 1e-15)

            # Match child part only
            C_childs = C[:, :n_childs]

            # Compare: (n_samples, n_events, n_childs)
            match = (Cs_childs.unsqueeze(1) == C_childs.unsqueeze(0)).all(dim=-1)

            event_idx = match.float().argmax(dim=1)
            ps = p[event_idx]
            return torch.log(ps + 1e-15)

        # --------------------------------------------------------
        # CASE 2 — Parents exist
        # --------------------------------------------------------
        # Convert parent composite states to C-binary (your existing method)
        bin_list = []
        for row in Cs_pars:
            parent_bin = []
            for j, v in enumerate(self.parents):
                parent_bin.append(v.get_Cst_to_Cbin(row[j]).unsqueeze(0))
            parent_bin = torch.cat(parent_bin, dim=0)  # (n_parents, max_basic)
            bin_list.append(parent_bin)

        samples_bin = torch.stack(bin_list, dim=0).to(device)        # (n_samples, n_parents, max_basic)

        log_ps_out = []

        # --- batch loop ---
        for start in range(0, n_samples_total, batch_size):
            end = min(start + batch_size, n_samples_total)

            Cs_childs_batch = Cs_childs[start:end]
            samples_bin_batch = samples_bin[start:end]

            # 1. Compute compatibility & event probabilities
            p_exp = self.expand_and_check_compatibility(
                self._get_C_binary(),
                samples_bin_batch
            )   # (batch, n_events)

            p_norm = p_exp / (p_exp.sum(dim=1, keepdim=True) + 1e-15)

            # 2. Identify matching event row by child Cs
            C_childs_full = torch.as_tensor(self.C[:, :n_childs], device=device)

            match = (Cs_childs_batch.unsqueeze(1) == C_childs_full.unsqueeze(0)).all(dim=-1)
            event_idx = match.float().argmax(dim=1)

            # 3. Extract likelihood for matching event
            ps = p_norm[torch.arange(end-start), event_idx]
            log_ps_out.append(torch.log(ps + 1e-15))

        return torch.cat(log_ps_out, dim=0)

def get_names(var_list):
    return [x.name for x in var_list]




