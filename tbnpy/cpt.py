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
            NOTE: log probabilities are stored in ps.

    Notes:
        C and p have the same number of rows.
        Cs and ps have the same number of rows.
    '''

    def __init__(self, childs, parents=[], C=[], p=[], Cs=[], ps=[], evidence=[], device="cpu"):

        self.device = device

        self.childs = childs
        self.parents = parents
        self.C = C
        self.p = p
        self.Cs = Cs
        self.ps = ps
        self.evidence = evidence

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
        """
        Acceptable shapes:
            (n, )                               → reshaped to (n,1)
            (n, m)                              → OK
            (n_evi, n, m)                       → OK for evidence-aware sampling

        m must equal len(childs) + len(parents).
        """
        if value is None or (isinstance(value, list) and len(value) == 0):
            self._Cs = torch.empty((0, 0), dtype=torch.int64)
            return

        value = self._to_tensor(value, dtype=torch.int64)

        # ------------------------------
        # Normalize shapes
        # ------------------------------
        if value.ndim == 1:
            # Single variable output → treat as (n,1)
            value = value.unsqueeze(1)

        elif value.ndim == 2:
            # (n, m) → OK
            pass

        elif value.ndim == 3:
            # (n_evi, n_sample, m) → OK
            pass

        else:
            raise ValueError(f"Cs must be 1D, 2D, or 3D; got {value.ndim}D")

        # ------------------------------
        # Validate number of columns
        # ------------------------------
        if value.numel() > 0:
            expected_cols = len(self._childs) + len(self._parents)

            # Cs 2D: (n, m)
            if value.ndim == 2:
                assert value.shape[1] == expected_cols, \
                    f"Cs must have {expected_cols} columns but got {value.shape[1]}."

            # Cs 3D: (n_evi, n_sample, m)
            if value.ndim == 3:
                assert value.shape[2] == expected_cols, \
                    f"Cs must have {expected_cols} columns but got {value.shape[2]}."

        self._Cs = value

    @property
    def ps(self):
        return self._ps

    @ps.setter
    def ps(self, value):
        """
        Acceptable shapes:
            (n,) → reshaped to (n,1)
            (n,1) → OK
            (n_evi, n) → evidence-aware sampling
            (n_evi, n, 1) → reshaped to (n_evi, n)

        Must match Cs batch shape.
        """
        if value is None or (isinstance(value, list) and len(value) == 0):
            self._ps = torch.empty((0, 1), dtype=torch.float32)
            return

        value = self._to_tensor(value, dtype=torch.float32)

        # ------------------------------
        # Normalize shape
        # ------------------------------
        if value.ndim == 1:
            # (n,) → (n,1)
            value = value.unsqueeze(1)

        elif value.ndim == 2:
            # Could be (n,1) or (n_evi, n)
            pass

        elif value.ndim == 3:
            # (n_evi, n, 1) → (n_evi, n)
            if value.shape[-1] == 1:
                value = value.squeeze(-1)
            else:
                raise ValueError("ps 3D must have last dim = 1")

        else:
            raise ValueError(f"ps must be 1D, 2D, or 3D; got {value.ndim}D")

        # ------------------------------
        # Validate with Cs
        # ------------------------------
        if hasattr(self, "_Cs") and self._Cs.numel() > 0:
            Cs = self._Cs

            # Case A: Cs is 2D: (n, m)
            if Cs.ndim == 2:
                n = Cs.shape[0]
                assert value.ndim == 2 and value.shape[0] == n, \
                    f"ps must have first dim = {n}. Got {value.shape}."

            # Case B: Cs is 3D: (n_evi, n, m)
            elif Cs.ndim == 3:
                n_evi, n = Cs.shape[0], Cs.shape[1]
                assert value.ndim == 2, \
                    f"ps must be 2D (n_evi, n). Got {value.shape}."
                assert value.shape == (n_evi, n), \
                    f"ps must match Cs dims {(n_evi, n)} but got {value.shape}."

            else:
                raise ValueError(f"Unexpected Cs shape: {Cs.shape}")

        self._ps = value

    @property
    def evidence(self):
        return self._evidence


    @evidence.setter
    def evidence(self, value):
        """
        evidence: observations of the child variables.

        Expected shape:
            (n_evidence, n_childs)

        Acceptable input types:
            - None
            - empty list []
            - list
            - numpy array
            - torch tensor
        """

        # Handle None or empty list
        if value is None or (isinstance(value, list) and len(value) == 0):
            # store empty evidence matrix
            self._evidence = torch.empty(
                (0, len(self.childs)),
                dtype=torch.int64,
                device=self.device
            )
            return

        # Convert to torch tensor
        value = self._to_tensor(value, dtype=torch.int64)

        n_childs = len(self.childs)
        # Case 1: ONE child variable
        if n_childs == 1:

            # 1D evidence is valid → treat as (N, 1)
            if value.ndim == 1:
                value = value.unsqueeze(1)   # (N,1)

            # otherwise must be (N,1)
            assert value.ndim == 2 and value.shape[1] == 1, (
                f"Evidence for 1 child must be shape (N,) or (N,1), "
                f"but got {value.shape}"
            )

            self._evidence = value
            return

        # Case 2: MULTIPLE child variables
        else:
            # 1D evidence is NOT valid when multiple child variables exist
            assert value.ndim == 2, (
                f"Evidence must be 2D (N, {n_childs}) for a node "
                f"with {n_childs} child variables. Got {value.shape}"
            )

            assert value.shape[1] == n_childs, (
                f"Evidence has {value.shape[1]} columns but expected {n_childs}."
            )

            self._evidence = value
            return

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
            ps = ps.log()
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
        ps_all = ps_all.log()
        return Cs_all, ps_all
    
    def expand_and_check_compatibility_all(self, C_binary, samples_binary):
        """
        Generic compatibility check that treats ALL variables uniformly.
        Unlike expand_and_check_compatibility, this function does not
        distinguish between child and parent variables.
        
        C_binary:       (n_event,  n_vars, max_state)
        samples_binary: (n_sample, n_vars, max_state)

        Returns:
            p_exp: (n_sample, n_event)
                event probabilities with incompatible pairs zeroed.
        """
        device = C_binary.device
        p = self.p.to(device)

        n_event, n_vars, max_state_e = C_binary.shape
        n_sample, n_vars_s, max_state_s = samples_binary.shape

        assert n_vars == n_vars_s, \
            f"C_binary has {n_vars} vars but samples_binary has {n_vars_s}"

        # ---------------------------------------------------------
        # 1. Pad to same max_state if needed
        # ---------------------------------------------------------
        max_global = max(max_state_e, max_state_s)

        if max_global > max_state_e:
            C_binary = torch.nn.functional.pad(C_binary, (0, max_global - max_state_e))

        if max_global > max_state_s:
            samples_binary = torch.nn.functional.pad(samples_binary, (0, max_global - max_state_s))

        # ---------------------------------------------------------
        # 2. Broadcast to compare all events × all samples
        # ---------------------------------------------------------
        # (n_event, n_sample, n_vars, max_state)
        Cb_exp = C_binary.unsqueeze(1).expand(n_event, n_sample, n_vars, max_global)
        Sb_exp = samples_binary.unsqueeze(0).expand(n_event, n_sample, n_vars, max_global)

        # ---------------------------------------------------------
        # 3. Compatibility check across ALL variables
        # ---------------------------------------------------------
        multiplied = Cb_exp * Sb_exp  # same shape

        # For each (event, sample, variable): if all 0 → incompatible
        zero_mask = multiplied.sum(dim=-1) == 0  # (n_event, n_sample, n_vars)

        # Event/sample incompatible if ANY variable is incompatible
        incompatible = zero_mask.any(dim=-1)     # (n_event, n_sample)

        # compatible: 1; incompatible: 0
        compatibility_mask = (~incompatible).float()  # (n_event, n_sample)

        # ---------------------------------------------------------
        # 4. Expand p over samples and apply mask
        # ---------------------------------------------------------
        if p.dim() == 1:
            p_exp = p.unsqueeze(1).expand(n_event, n_sample)
        else:
            p_exp = p.expand(n_event, n_sample)

        # Zero out incompatible ones
        p_exp = p_exp * compatibility_mask

        return p_exp.T  # (n_sample, n_event)
    
    def log_prob(self, Cs, batch_size=100_000):
        """
        Computes log P(sample | parents) for each row of Cs.
        Cs is arranged in the same order as self.variables = childs + parents.
        """
        device = self.p.device
        n_samples_total = Cs.size(0)

        variables = self.childs + self.parents
        max_basic = max(len(v.values) for v in variables)

        # ---------------------------------------------------------
        # Convert full Cs into binary for all variables
        # ---------------------------------------------------------
        bin_list = []
        for row in Cs:
            row_bin = []
            for j, v in enumerate(variables):

                b = v.get_Cst_to_Cbin(row[j])    # shape: (k,) where k varies per variable

                # pad to max_basic
                pad = max_basic - b.numel()
                if pad > 0:
                    b = torch.nn.functional.pad(b, (0, pad))

                row_bin.append(b.unsqueeze(0))    # shape: (1, max_basic)

            row_bin = torch.cat(row_bin, dim=0)   # shape: (n_vars, max_basic)
            bin_list.append(row_bin)

        samples_bin = torch.stack(bin_list, dim=0).to(device)
        # shape: (n_samples, n_vars, max_basic)

        # ---------------------------------------------------------
        # Check compatibility in batches
        # ---------------------------------------------------------

        # C binary
        Cb = self._get_C_binary()      # (n_events, n_vars, max_basic)

        logp_out = []

        for start in range(0, n_samples_total, batch_size):
            end = min(start + batch_size, n_samples_total)

            # batch parents → used in compatibility
            samples_bin_batch = samples_bin[start:end]

            # 1. compatibility using parent info
            p_exp = self.expand_and_check_compatibility_all(
                Cb,
                samples_bin_batch
            )   # (batch, n_events)

            # 2. probability = sum p_exp
            probs = p_exp.sum(dim=1)

            logp_out.append(torch.log(probs + 1e-15))

        return torch.cat(logp_out, dim=0)
    
    def log_prob_evidence(self, Cs_par, batch_size=100_000):
        """
        Compute log P(evidence | parents(samples)) for each parent sample.
        
        Cs_par can be:
            (N_samples, n_parents) or
            (n_evi, N_samples, n_parents)   # evidence exists for parents too

        Returns:
            log_probs: (N_samples,) tensor of log probabilities
        """

        assert hasattr(self, "_evidence") and self._evidence is not None, \
            "Evidence is not set. Use self.evidence = ... first."

        device = Cs_par.device
        evidence = self._evidence.to(device)

        n_evi = evidence.size(0)
        n_childs = len(self._childs)
        n_parents = len(self._parents)

        # Identify case:
        # ---------------------------------------------------------------
        # Case A: Cs_par is (N_samples, n_parents)
        # ---------------------------------------------------------------
        if Cs_par.ndim == 2:
            n_samples = Cs_par.size(0)
            case = "no_par_evi"
        # ---------------------------------------------------------------
        # Case B: Cs_par is (n_evi, N_samples, n_parents)
        # ---------------------------------------------------------------
        elif Cs_par.ndim == 3:
            assert Cs_par.size(0) == n_evi, \
                f"Cs_par first dimension must match n_evidence={n_evi}"
            n_samples = Cs_par.size(1)
            case = "par_evi"
        else:
            raise ValueError("Cs_par must be 2D or 3D.")

        log_out = []

        # Iterate over sample batches
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            bsz = end - start

            # ---------------- CASE A: no evidence for parents ----------------
            if case == "no_par_evi":
                batch_par = Cs_par[start:end]                             # (bsz, n_parents)

                # expand evidence → (n_evi, bsz, n_childs)
                ev_exp = evidence.unsqueeze(1).expand(n_evi, bsz, n_childs)

                # expand parent samples → (n_evi, bsz, n_parents)
                par_exp = batch_par.unsqueeze(0).expand(n_evi, bsz, n_parents)

            # ---------------- CASE B: parent evidence exists -----------------
            else:   # case == "par_evi"
                # parent evidence together with child evidence
                par_exp = Cs_par[:, start:end, :]                        # (n_evi, bsz, n_parents)

                # broadcast child evidence to match bsz
                ev_exp = evidence.unsqueeze(1).expand(n_evi, bsz, n_childs)


            # Concatenate childs + parents
            Cs_full = torch.cat([ev_exp, par_exp], dim=2)                # (n_evi, bsz, n_vars)

            # flatten for log_prob
            Cs_flat = Cs_full.reshape(-1, n_childs + n_parents)

            # compute log_probs for each evidence row/sample
            logp_flat = self.log_prob(Cs_flat)

            # reshape to (n_evi, bsz)
            logp_evi = logp_flat.reshape(n_evi, bsz)

            # sum over evidence → (bsz,)
            logp_batch = logp_evi.sum(dim=0)

            log_out.append(logp_batch)

        return torch.cat(log_out, dim=0)

    def sample_evidence(self, Cs_pars, batch_size=100_000):
        """
        Samples from this CPT when Cs_pars contains parent evidence aligned
        with child evidence.

        Cs_pars shape:
            (n_evi, n_samples, n_parents)

        Returns:
            Cs_out: (n_evi, n_samples, n_childs + n_parents)
            ps_out: (n_evi, n_samples)    # probability of sampled event
        """

        assert Cs_pars.ndim == 3, \
            "Cs_pars must be 3D: (n_evi, n_samples, n_parents)"

        device = self.p.device
        n_evi, n_samples_total, n_parents = Cs_pars.shape
        n_childs = len(self.childs)

        parent_vars = self.parents

        # -----------------------------------------------------------
        # Convert parent's composite states to binary
        # -----------------------------------------------------------
        # We'll produce samples_bin: (n_evi, n_samples, n_parents, max_basic)
        bin_list = []

        for evi_idx in range(n_evi):
            par_bin_rows = []
            for sample_idx in range(n_samples_total):
                parent_bin_one = []
                row = Cs_pars[evi_idx, sample_idx]
                for j, v in enumerate(parent_vars):
                    parent_bin_one.append(v.get_Cst_to_Cbin(row[j]).unsqueeze(0))
                parent_bin_one = torch.cat(parent_bin_one, dim=0)  # (n_parents, max_basic)
                par_bin_rows.append(parent_bin_one)
            par_bin_rows = torch.stack(par_bin_rows, dim=0)  # (n_samples, n_parents, max_basic)
            bin_list.append(par_bin_rows)

        samples_bin = torch.stack(bin_list, dim=0).to(device)
        # samples_bin shape: (n_evi, n_samples, n_parents, max_basic)

        Cs_out_list = []
        ps_out_list = []

        # -----------------------------------------------------------
        # Loop over n_evi (evidence rows)
        # -----------------------------------------------------------
        Cb = self._get_C_binary()   # event definitions (n_events, ..., ...)
        n_events = self.p.numel()

        for evi_idx in range(n_evi):

            samples_bin_evi = samples_bin[evi_idx]  # (n_samples, n_parents, max_basic)

            Cs_evi_batches = []
            ps_evi_batches = []

            for start in range(0, n_samples_total, batch_size):
                end = min(start + batch_size, n_samples_total)

                batch_bin = samples_bin_evi[start:end]   # (batch, n_parents, max_basic)

                # -------------------------------------------------
                # Compute compatibility, returns (batch, n_events)
                # -------------------------------------------------
                p_exp = self.expand_and_check_compatibility(
                    Cb,
                    batch_bin
                )

                # -------------------------------------------------
                # Sample event index per sample
                # -------------------------------------------------
                Cs_batch, event_idx_batch = self.sample_from_p_exp(p_exp)

                # Normalize and select probabilities
                p_norm = p_exp / (p_exp.sum(dim=1, keepdim=True) + 1e-15)
                ps_batch = p_norm[
                    torch.arange(p_norm.size(0)), event_idx_batch
                ]  # (batch,)

                # -------------------------------------------------
                # Build full Cs = [child, parents]
                # -------------------------------------------------
                # 1. child composite states: Cs_batch: (batch, n_childs)
                # 2. parent assignments: Cs_pars[evi_idx, start:end]: (batch, n_parents)
                parents_this_batch = Cs_pars[evi_idx, start:end].to(device)
                Cs_full_batch = torch.cat([Cs_batch, parents_this_batch], dim=1)

                Cs_evi_batches.append(Cs_full_batch)
                ps_evi_batches.append(ps_batch)

            # Merge batches for this evidence row
            Cs_out_list.append(torch.cat(Cs_evi_batches, dim=0))
            ps_out_list.append(torch.cat(ps_evi_batches, dim=0))

        # Final shapes:
        # Cs_out_list: list length n_evi, each (n_samples, n_childs + n_parents)
        # stack → (n_evi, n_samples, n_childs + n_parents)
        Cs_out = torch.stack(Cs_out_list, dim=0)

        # ps_out_list: list length n_evi, each (n_samples,)
        ps_out = torch.stack(ps_out_list, dim=0)
        ps_out = ps_out.log()
        return Cs_out, ps_out

    def sample_evidence(self, Cs_pars, batch_size=100_000):
        """
        Vectorized sampling given aligned parent evidence.

        Cs_pars: (n_evi, n_samples, n_parents)

        Returns:
            Cs_out: (n_evi, n_samples, n_childs + n_parents)
            ps_out: (n_evi, n_samples)
        """

        assert Cs_pars.ndim == 3, \
            "Cs_pars must be (n_evi, n_samples, n_parents)"

        device = self.p.device
        n_evi, n_samples, n_parents = Cs_pars.shape
        n_childs = len(self.childs)

        # -----------------------------------------------------
        # Flatten evidence-parent input:
        #   (n_evi, n_samples, n_parents)
        # → (n_evi * n_samples, n_parents)
        # -----------------------------------------------------
        Cs_pars_flat = Cs_pars.reshape(-1, n_parents).to(device)
        n_total = Cs_pars_flat.size(0)

        # -----------------------------------------------------
        # Convert parent composite states to binary (vectorized)
        # -----------------------------------------------------
        parent_vars = self.parents
        bin_list = []

        for j, v in enumerate(parent_vars):
            # v.get_Cst_to_Cbin takes shape () → (k,)
            # apply over the entire column
            col = Cs_pars_flat[:, j]
            bin_j = torch.stack([v.get_Cst_to_Cbin(val) for val in col], dim=0)
            bin_list.append(bin_j.unsqueeze(1))   # shape (n_total, 1, max_basic)

        # parent_bin: (n_total, n_parents, max_basic)
        parent_bin = torch.cat(bin_list, dim=1).to(device)

        # -----------------------------------------------------
        # Prepare outputs
        # -----------------------------------------------------
        Cs_out_list = []
        ps_out_list = []

        Cb = self._get_C_binary()  # (n_events, n_vars, max_basic)

        # -----------------------------------------------------
        # Batch over flattened samples
        # -----------------------------------------------------
        for start in range(0, n_total, batch_size):
            end = min(start + batch_size, n_total)

            parent_bin_batch = parent_bin[start:end]  # (batch, n_parents, max_basic)

            # 1. compatibility: (batch, n_events)
            p_exp = self.expand_and_check_compatibility(Cb, parent_bin_batch)

            # 2. sample event index per row
            Cs_batch, event_idx_batch = self.sample_from_p_exp(p_exp)

            # 3. probability of selected event
            p_norm = p_exp / (p_exp.sum(dim=1, keepdim=True) + 1e-15)
            ps_batch = p_norm[torch.arange(p_norm.size(0)), event_idx_batch]

            # 4. assemble full Cs = [child | parent]
            parent_vals = Cs_pars_flat[start:end]   # (batch, n_parents)
            Cs_full = torch.cat([Cs_batch, parent_vals], dim=1)

            Cs_out_list.append(Cs_full)
            ps_out_list.append(ps_batch)

        # Merge batches
        Cs_flat_all = torch.cat(Cs_out_list, dim=0)   # (n_total, n_childs + n_parents)
        ps_flat_all = torch.cat(ps_out_list, dim=0)   # (n_total,)

        # -----------------------------------------------------
        # Reshape back to (n_evi, n_samples, ...)
        # -----------------------------------------------------
        Cs_out = Cs_flat_all.reshape(n_evi, n_samples, n_childs + n_parents)
        ps_out = ps_flat_all.reshape(n_evi, n_samples)
        ps_out = ps_out.log()

        return Cs_out, ps_out


def get_names(var_list):
    return [x.name for x in var_list]





