import numpy as np
from itertools import chain, combinations
from math import comb

import torch

class Variable:
    """
    A class to manage information about a variable used in matrix-based Bayesian networks.

    Input attributes
    ----------
    name : str
        Name of the variable.
    values : list or str
        If list, considered a discrete-state variable, with states 0, 1, 2, ... .
        (e.g. ['low', 'medium', 'high'] or ['failure', 'survival'])
        If str, considered a continuous variable.
        (e.g. 'PGA (g)' or 'km')

    Notes: How to reduce memory of C matrix using composite states
    -----
    A set of basic states S={A, B, C, ...} can be represented by a composite state 
    f(S) = ∑_{k=1}^{m-1} C(n, k)   # all smaller-sized subsets
        + ( C(n, m) - 1 - ∑_{i=1}^{m} C(n-1 - s_i, m+1 - i) )   # lex rank within size-m group
        where n = len(values), m = |S|, and s_i is the i-th smallest element in S.

    Example:
    with values = ['low', 'medium', 'high'], the composite state for S={1, 2} (i.e. {medium, high}) is:
    f(S) = ∑_{k=1}^{1} C(3, k)
        + ( C(3, 2) - 1 - ∑_{i=1}^{2} C(3-1 - s_i, 2+1 - i) )
         = C(3, 1)
        + ( C(3, 2) - 1 - (C(3-1-1, 2) + C(3-1-2, 1)) )
         = 3 + (3 - 1 - (C(1, 2) + C(0, 1))) = 5 
    where C(a, b) = 0 if a < b.

    Tips
    -----------
    When applicable, use an ordering where lower indices represent worse outcomes, as some modules assume this ordering.
    For example: `['failure', 'survival']` since `0 < 1`.
    """

    def __init__(self, name: str, values: list=[]):
        '''Initialise the Variable object.

        Args:
            name (str): name of the variable.
            values (list or np.ndarray): description of states.
            B_flag (str): flag to determine how B is generated.
        '''
        assert isinstance(name, str), 'name should be a string'
        assert isinstance(values, (list, np.ndarray)), \
            'values must be a list or np.ndarray'

        self._name = name
        self._values = values
        
    # Magic methods
    def __hash__(self):
        return hash(self._name)

    def __lt__(self, other):
        return self._name < other.name

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self._name == other._name and self._values == other._values
        else:
            return False

    def __repr__(self):
        return (
            f"Variable(name={repr(self._name)},\n"
            f"  values={repr(self._values)},\n"
        )

    # Property for 'name'
    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        assert isinstance(new_name, str), 'name must be a string'
        self._name = new_name

    # Property for 'values'
    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, new_values):
        assert isinstance(new_values, (list, np.ndarray, str)), 'values must be a list, np.ndarray, or str'
        self._values = new_values


    def get_state(self, state_set):
        '''Finds the state index of a given set of basic states.

        The sets are ordered as follows (cf. gen_B):
        [{0}, {1}, ..., {n-1}, {0, 1}, {0, 2}, ..., {n-2, n-1},
        {0, 1, 2}, ..., {0, 1, ..., n-1}]



        Args:
            state_set (set): set of basic states.

        Returns:
            state (int): state index in B matrix of the given set.
        '''
        assert isinstance(state_set, set), 'set must be a set'

        # Find the index by calculation
        # The number of elements in the target set
        num_elements = len(state_set)
        # Number of basic states
        n = len(self.values)

        # Initialize the state
        state = 0
        # Add the number of sets with fewer elements
        for k in range(1, num_elements):
            state += len(list(combinations(range(n), k)))
        # Find where the target set is in the group
        # with 'num_elements' elements
        combinations_list = list(combinations(range(n), num_elements))

        # Convert target_set to a sorted tuple
        # to match the combinations
        target_tuple = tuple(sorted(state_set))
        # Find the index within the group
        idx_in_group = combinations_list.index(target_tuple)

        # Add the position within the group to the state
        state += idx_in_group

        return state

    def get_set(self, state):
        '''Finds the set of basic states represented by a given state index.

        The sets are ordered as follows (cf. gen_B):
        [{0}, {1}, ..., {n-1}, {0, 1}, {0, 2}, ..., {n-2, n-1},
        {0, 1, 2}, ..., {0, 1, ..., n-1}]

        Args:
            state (int): state index.

        Returns:
            set (set): set of basic states.
        '''
        assert np.issubdtype(type(state), np.integer), 'state must be an integer'

        # the number of states
        n = len(self.values)
        # Initialize the state tracker
        current_state = 0

        # Iterate through the group sizes
        # (1-element sets, 2-element sets, etc.)
        for k in range(1, n+1):
            # Count the number of sets of size k
            comb_count = len(list(combinations(range(n), k)))

            # Check if the index falls within this group
            if current_state + comb_count > state:
                # If it falls within this group,
                # calculate the exact set
                combinations_list = list(combinations(range(n), k))
                set_tuple = combinations_list[state - current_state]
                return set(set_tuple)

            # Otherwise, move to the next group
            current_state += comb_count

        # If the index is out of bounds, raise an error
        raise IndexError(f"The given state index must be not greater than {2**n-1}")

    def get_state_from_vector(self, vector):
        '''Finds the state index for a given binary vector.

        Args:
            vector (list or np.ndarray): binary vector.
            1 if the basic state is involved, 0 otherwise.

        Returns:
            state (int): state index.
            -1 if the vector is all zeros.
        '''
        assert isinstance(vector, (list, np.ndarray)), \
            'vector must be a list or np.ndarray'
        assert len(vector) == len(self.values), \
            'vector must have the same length as values'

        # Count the number of 1's in the vector
        num_ones = sum(vector)

        # Return -1 if the vector is all zeros
        if num_ones == 0:
            return -1

        # Number of basic states
        n = len(vector)

        # Initialize the state
        state = 0
        # Add the number of vectors with fewer 1's
        for k in range(1, num_ones):
            state += len(list(combinations(range(n), k)))

        # Find where this vector is in the group with 'num_ones' ones
        one_positions = [i for i, val in enumerate(vector) if val == 1]
        # Find the position of this specific combination in the group
        combs = list(combinations(range(n), num_ones))
        idx_in_group = combs.index(tuple(one_positions))

        # Add the position within the group to the state
        state += idx_in_group

        return state

    def build_state_to_binary_lookup(self):
        """
        Build a lookup table mapping composite state index f(S) → binary vector,
        using the exact combinatorial ranking formula given in the Variable documentation.

        Returns:
            lookup (torch.Tensor): shape (n_composite_states, n_basic)
            ordered_states: a list of sets (optional debug)
        """
        n = len(self.values)
        max_states = 2**n  # including empty set

        # We exclude empty set (state index = -1)
        all_sets = []
        for integer in range(1, max_states):  # skip 0 = empty
            # Convert integer bitmap to set of basic indices
            S = {i for i in range(n) if (integer >> i) & 1}
            all_sets.append(S)

        # Step 1: compute composite index f(S) for each S
        def f(S):
            m = len(S)
            s_sorted = sorted(S)

            # First term: sum over smaller subsets
            first_term = sum(comb(n, k) for k in range(1, m))

            # Second term: lexicographic rank inside size-m group
            subtract_sum = sum(comb(n-1 - s_i, m+1 - (i+1)) 
                            for i, s_i in enumerate(s_sorted))
            second_term = comb(n, m) - 1 - subtract_sum

            return first_term + second_term

        # Compute states and allocate lookup table
        composite_states = [f(S) for S in all_sets]
        n_composite = max(composite_states) + 1

        lookup = torch.zeros((n_composite, n), dtype=torch.int8)

        # Fill lookup: for state index f(S), mark binary vector
        for S, state in zip(all_sets, composite_states):
            for b in S:
                lookup[state, b] = 1

        return lookup

    def get_Cst_to_Cbin(self, Cst):
        """
        Vectorised: convert composite state indices → binary matrix.

        Args:
            Cst: (n_events,) int64 tensor
            var: Variable object (for n_basic)

        Returns:
            Cbin: (n_events, n_basic) binary tensor
        """
        Cst = Cst.to(torch.long)

        lookup = self.build_state_to_binary_lookup().to(Cst.device)
        # Vectorised lookup
        Cbin = lookup[Cst]

        return Cbin
    
    def build_bitmask_to_state_lookup(self):
        """
        Build lookup table mapping bitmask id -> composite state f(S).
        No loops over events. Only loops over all subsets (2^n subsets).
        """

        n_basic = len(self.values)
        max_subsets = 2 ** n_basic
        lookup = torch.full((max_subsets,), -1, dtype=torch.long)

        for bitmask in range(max_subsets):

            # Skip empty set (no composite index)
            if bitmask == 0:
                continue

            # Find S = {i | bitmask has bit i}
            S = [i for i in range(n_basic) if (bitmask >> i) & 1]
            m = len(S)

            # ---- First term: sum_{k=1}^{m-1} C(n_basic, k)
            term1 = sum(comb(n_basic, k) for k in range(1, m))

            # ---- Second term: C(n_basic,m) - 1 - sum C(...)
            subtract_sum = 0
            for i, s_i in enumerate(S):
                subtract_sum += comb(n_basic - 1 - s_i, m + 1 - (i + 1))

            term2 = comb(n_basic, m) - 1 - subtract_sum

            lookup[bitmask] = term1 + term2

        return lookup

    def get_Cbin_to_Cst(self, Cbin):
        """
        Fully vectorised: no loops over events.
        Cbin: (n_events, n_basic)
        Returns Cst: (n_events,)
        """

        device = Cbin.device
        n_basic = Cbin.shape[1]
        # ---- Step 1: build lookup table once
        lookup = self.build_bitmask_to_state_lookup().to(device)

        # ---- Step 2: encode each row into integer bitmask (vectorised)
        powers = (2 ** torch.arange(n_basic, device=device)).reshape(1, -1)
        ids = (Cbin * powers).sum(dim=1)     # shape: (n_events,)

        # ---- Step 3: lookup all states (vectorised)
        Cst = lookup[ids]

        return Cst
