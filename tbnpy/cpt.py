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

def get_names(var_list):
    return [x.name for x in var_list]

