import numpy as np

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
        p (array_like): probability vector related to C.
        Cs (array_like): event matrix of samples.
        q (array_like): sampling probability vector related to Cs.

    Notes:
        C and p have the same number of rows.
        Cs and q have the same number of rows.
    '''

    def __init__(self, variables, no_child, C=[], p=[], Cs=[], q=[], ps=[], sample_idx=[]):

        self.variables = variables
        self.no_child = no_child
        self.C = C
        self.p = p
        self.Cs = Cs
        self.q = q

    # Magic methods
    def __hash__(self):
        return hash(self._name)

    def __eq__(self, other):
        if isinstance(other, Cpm):
            return (
                self._name == other._name and
                self._variables == other._variables and
                self._no_child == other._no_child and
                self._C == other._C and
                self._p == other._p and
                self._Cs == other._Cs and
                self._q == other._q
            )
        else:
            return False

    def __repr__(self):
        details = [
            f"{self.__class__.__name__}(variables={self.get_names()},",
            f"no_child={self.no_child},",
            f"C={self.C},",
            f"p={self.p},",
        ]

        if self._Cs.size:
            details.append(f"Cs={self._Cs},")
        if self._q.size:
            details.append(f"q={self._q},")

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
        if isinstance(value, list):
            value = np.array(value, dtype=np.int32)

        if value.size:
            assert value.dtype in (np.dtype('int64'), np.dtype('int32')), f'Event matrix C must be a numeric matrix: {value}'

            if value.ndim == 1:
                value.shape = (len(value), 1)
            else:
                assert value.shape[1] == len(self._variables), 'C must have the same number of columns as that of variables'

            max_C = np.max(value, axis=0, initial=0)
            max_var = [2**len(x.values)-1 for x in self._variables]
            assert all(max_C <= max_var), f'check C matrix: {max_C} vs {max_var}'

        self._C = value

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size:
            assert len(value) == self._C.shape[0], 'p must have the same length as the number of rows in C'

            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'p must be a numeric vector'

        self._p = value

    @property
    def Cs(self):
        return self._Cs

    @Cs.setter
    def Cs(self, value):
        if isinstance(value, list):
            value = np.array(value, dtype=np.int32)

        if value.size:
            if value.ndim == 1: # event matrix for samples
                value.shape = (len(value), 1)
            else:
                assert value.shape[1] == len(self._variables), 'Cs must have the same number of columns as the number of variables'

            max_Cs = np.max(value, axis=0, initial=0)
            max_var_basic = [len(x.values) for x in self.variables]
            assert all(max_Cs <= max_var_basic), f'check Cs matrix: {max_Cs} vs {max_var_basic}'

        self._Cs = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if isinstance(value, list):
            value = np.array(value)[:, np.newaxis]

        if value.ndim == 1:
            value.shape = (len(value), 1)

        if value.size and self._Cs.size:
            assert len(value) == self._Cs.shape[0], 'q must have the same length as the number of rows in Cs'

        if value.size:
            all(isinstance(y, (float, np.float32, np.float64, int, np.int32, np.int64)) for y in value), 'q must be a numeric vector'

        self._q = value
