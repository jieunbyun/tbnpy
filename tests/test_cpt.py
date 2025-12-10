import pytest

from tbnpy import cpt, variable
import numpy as np

import torch

@pytest.fixture
def dict_cpt1():
    # P(A1|A2,A3)
    ''' Use instance of Variables in the variables'''
    A1 = variable.Variable(**{'name': 'A1',
                              'values': ['s', 'f']})
    A2 = variable.Variable(**{'name': 'A2',
                              'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3',
                              'values': ['s', 'f']})

    return {'childs': [A1],
            'parents': [A2, A3],
            'C': np.array([[1, 1, 2], [1, 0, 1], [0, 0, 0]]),
            'p': [1, 1, 1]}

@pytest.fixture
def dict_cpt2():
    ''' Use instance of Variables in the variables'''
    # P(A1|A2,A3)
    A1 = variable.Variable(**{'name': 'A1',
                              'values': ['low','mid', 'high']})
    A2 = variable.Variable(**{'name': 'A2',
                              'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3',
                              'values': ['s', 'f']})

    return {'childs': [A1],
            'parents': [A2, A3],
            'C': np.array([[0, 1, 2], [1, 1, 2], [1, 0, 1], [2, 0, 1], [2, 0, 0]]),
            'p': [0.9, 0.1, 0.5, 0.5, 1.0]}

@pytest.fixture
def dict_cpt3():
    # P(A2)
    A2 = variable.Variable(**{'name': 'A2',
                              'values': ['s', 'f']})
    return {'childs': [A2],
            'parents': [],
            'C': np.array([[1], [0]]),
            'p': [0.7, 0.3]}

@pytest.fixture
def dict_cpt4():
    # P(A3)
    A3 = variable.Variable(**{'name': 'A3',
                              'values': ['s', 'f']})
    return {'childs': [A3],
            'parents': [],
            'C': np.array([[1], [0]]),
            'p': [0.9, 0.1]}

@pytest.fixture
def dict_cpt5():
    # P(A2, A3)
    A2 = variable.Variable(**{'name': 'A2',
                              'values': ['s', 'f']})
    A3 = variable.Variable(**{'name': 'A3',
                              'values': ['s', 'f']})
    return {'childs': [A2, A3],
            'parents': [],
            'C': np.array([[2, 0], [0, 1], [1, 1]]),
            'p': [0.5, 0.3, 0.2]}

def test_init1(dict_cpt1):

    a = cpt.Cpt(**dict_cpt1)
    assert isinstance(a, cpt.Cpt)

def test_init2(dict_cpt1):

    a = cpt.Cpt(**dict_cpt1)
    assert isinstance(a, cpt.Cpt)
    assert a.childs==dict_cpt1['childs']
    assert a.parents == dict_cpt1['parents']
    np.testing.assert_array_equal(a.C, dict_cpt1['C'])

    assert isinstance(a.C, torch.Tensor)
    assert isinstance(a.p, torch.Tensor)

def test_init3(dict_cpt1):
    # using list for P
    v = dict_cpt1
    a = cpt.Cpt(childs=v['childs'], parents=v['parents'], C=np.array([[0, 2, 1], [1, 2, 1]]), p=[0.1, 0.9])
    assert isinstance(a, cpt.Cpt)

def test_init4(dict_cpt1):
    # no p
    v = dict_cpt1
    a = cpt.Cpt(childs=[v['childs'][0]], parents=v['parents'], C=np.array([[0, 2, 1], [1, 2, 1]]))
    assert isinstance(a, cpt.Cpt)


def test_init5():
    # variables must be a list of Variable
    with pytest.raises(AssertionError):
        a = cpt.Cpt(childs=['1'], parents=[], C=np.array([1]), p=[0.9])

def test_get_C_binary1(dict_cpt1):
    a = cpt.Cpt(**dict_cpt1)
    Cb = a._get_C_binary()

    expected = torch.tensor([
        [[0,1],[0,1],[1,1]], # event 1
        [[0,1],[1,0],[0,1]], # event 2
        [[1,0],[1,0],[1,0]], # event 3
    ], dtype=torch.int8) # shape (3 events, 3 variables, max_basic=2)

    n_events = dict_cpt1['C'].shape[0]
    n_variables = len(dict_cpt1['childs']) + len(dict_cpt1['parents'])
    max_basic = max(len(v.values) for v in dict_cpt1['childs'] + dict_cpt1['parents'])

    assert isinstance(Cb, torch.Tensor)
    assert Cb.shape == (n_events, n_variables, max_basic)
    assert torch.equal(Cb, expected)

def test_get_C_binary2(dict_cpt2):
    a = cpt.Cpt(**dict_cpt2)
    Cb = a._get_C_binary()

    expected = torch.tensor([
        [[1,0,0],[0,1,0],[1,1,0]], # event 1
        [[0,1,0],[0,1,0],[1,1,0]], # event 2
        [[0,1,0],[1,0,0],[0,1,0]], # event 3
        [[0,0,1],[1,0,0],[0,1,0]], # event 4
        [[0,0,1],[1,0,0],[1,0,0]], # event 5
    ], dtype=torch.int8) # shape (3 events, 3 variables, max_basic=3)

    n_events = dict_cpt2['C'].shape[0]
    n_variables = len(dict_cpt2['childs']) + len(dict_cpt2['parents'])
    max_basic = max(len(v.values) for v in dict_cpt2['childs'] + dict_cpt2['parents'])

    assert isinstance(Cb, torch.Tensor)
    assert Cb.shape == (n_events, n_variables, max_basic)
    assert torch.equal(Cb, expected)

def test_expand_and_check_compatibility1(dict_cpt1):

    # Setup CPT
    T = cpt.Cpt(**dict_cpt1)
    A1 = T.childs[0]
    A2 = T.parents[0]
    A3 = T.parents[1]

    # Expand C to binary
    C_binary = T._get_C_binary()

    # Composite-state samples for parents (A2,A3)
    raw_samples = torch.tensor([
        [1,1],
        [0,1],
        [1,0],
        [0,0]
    ], dtype=torch.long)

    # Convert composite → binary for each parent
    sample_bin_list = []
    for row in raw_samples:
        bin_A2 = A2.get_Cst_to_Cbin(row[0]).unsqueeze(0)  # shape (1,2)
        bin_A3 = A3.get_Cst_to_Cbin(row[1]).unsqueeze(0)  # shape (1,2)

        parent_stack = torch.cat([bin_A2, bin_A3], dim=0)  # (2,2)
        sample_bin_list.append(parent_stack)

    # Stack into shape (n_sample, n_parents, max_state)
    samples_bin = torch.stack(sample_bin_list, dim=0)  # (5,2,2)

    # Run compatibility expansion
    p_exp = T.expand_and_check_compatibility(
        C_binary,
        samples_bin
    )

    # Expected filtered probabilities
    expected = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # Assertions
    assert p_exp.shape == expected.shape
    assert torch.allclose(p_exp, expected), f"\nExpected:\n{expected}\nGot:\n{p_exp}"

def test_expand_and_check_compatibility2(dict_cpt2):

    # Setup CPT
    T = cpt.Cpt(**dict_cpt2)
    A1 = T.childs[0]
    A2 = T.parents[0]
    A3 = T.parents[1]

    # Expand C to binary
    C_binary = T._get_C_binary()

    # Composite-state samples for parents (A2,A3)
    raw_samples = torch.tensor([
        [1,1],
        [0,1],
        [1,0],
        [0,0]
    ], dtype=torch.long)

    # Convert composite → binary for each parent
    sample_bin_list = []
    for row in raw_samples:
        bin_A2 = A2.get_Cst_to_Cbin(row[0]).unsqueeze(0)  # shape (1,2)
        bin_A3 = A3.get_Cst_to_Cbin(row[1]).unsqueeze(0)  # shape (1,2)

        parent_stack = torch.cat([bin_A2, bin_A3], dim=0)  # (2,2)
        sample_bin_list.append(parent_stack)

    # Stack into shape (n_sample, n_parents, max_state)
    samples_bin = torch.stack(sample_bin_list, dim=0)  # (5,2,2)

    # Run compatibility expansion
    p_exp = T.expand_and_check_compatibility(
        C_binary,
        samples_bin
    )

    # Expected filtered probabilities
    expected = torch.tensor([
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.0],
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # Assertions
    assert p_exp.shape == expected.shape
    assert torch.allclose(p_exp, expected), f"\nExpected:\n{expected}\nGot:\n{p_exp}"

def test_expand_and_check_compatibility_all1(dict_cpt1):

    # Setup CPT
    T = cpt.Cpt(**dict_cpt1)
    A1 = T.childs[0]
    A2 = T.parents[0]
    A3 = T.parents[1]

    # Expand C to binary
    C_binary = T._get_C_binary()

    # Composite-state samples for parents (A2,A3)
    raw_samples = torch.tensor([
        [1,1,1],
        [1,0,1],
        [0,1,0],
        [0,0,0]
    ], dtype=torch.long)

    # Convert composite → binary for each parent
    sample_bin_list = []
    for row in raw_samples:
        bin_A1 = A1.get_Cst_to_Cbin(row[0]).unsqueeze(0)  # shape (1,2)
        bin_A2 = A2.get_Cst_to_Cbin(row[1]).unsqueeze(0)  # shape (1,2)
        bin_A3 = A3.get_Cst_to_Cbin(row[2]).unsqueeze(0)  # shape (1,2)

        parent_stack = torch.cat([bin_A1, bin_A2, bin_A3], dim=0)  # (2,2)
        sample_bin_list.append(parent_stack)

    # Stack into shape (n_sample, n_variables, max_state)
    samples_bin = torch.stack(sample_bin_list, dim=0)  # (5,2,2)

    # Run compatibility expansion
    p_exp = T.expand_and_check_compatibility_all(
        C_binary,
        samples_bin
    )

    # Expected filtered probabilities
    expected = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # Assertions
    assert p_exp.shape == expected.shape
    assert torch.allclose(p_exp, expected), f"\nExpected:\n{expected}\nGot:\n{p_exp}"

def test_expand_and_check_compatibility_all2(dict_cpt2):

    # Setup CPT
    T = cpt.Cpt(**dict_cpt2)
    A1 = T.childs[0]
    A2 = T.parents[0]
    A3 = T.parents[1]

    # Expand C to binary
    C_binary = T._get_C_binary()

    # Composite-state samples for parents (A2,A3)
    raw_samples = torch.tensor([
        [1,1,1],
        [1,0,1],
        [2,1,0],
        [2,0,0]
    ], dtype=torch.long)

    # Convert composite → binary for each parent
    sample_bin_list = []
    for row in raw_samples:
        bin_A1 = A1.get_Cst_to_Cbin(row[0]).unsqueeze(0)  # shape (1,3)

        b = A2.get_Cst_to_Cbin(row[1]) # shape (2,)
        b = torch.nn.functional.pad(b, (0,1)) # pad to (3,)
        bin_A2 = b.unsqueeze(0)  # shape (1,3)

        b = A3.get_Cst_to_Cbin(row[2]) # shape (2,)
        b = torch.nn.functional.pad(b, (0,1)) # pad to (3,)
        bin_A3 = b.unsqueeze(0)  # shape (1,3)

        parent_stack = torch.cat([bin_A1, bin_A2, bin_A3], dim=0)  # (3,2)
        sample_bin_list.append(parent_stack)

    # Stack into shape (n_sample, n_variables, max_state)
    samples_bin = torch.stack(sample_bin_list, dim=0)  # (5,3,2)

    # Run compatibility expansion
    p_exp = T.expand_and_check_compatibility_all(
        C_binary,
        samples_bin
    )

    # Expected filtered probabilities
    expected = torch.tensor([
        [0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # Assertions
    assert p_exp.shape == expected.shape
    assert torch.allclose(p_exp, expected), f"\nExpected:\n{expected}\nGot:\n{p_exp}"

def test_sample_from_p_exp1(dict_cpt1):
    # Setup CPT
    T = cpt.Cpt(**dict_cpt1)

    # Deterministic probability matrix
    p_exp = torch.tensor([
        [1.0, 0.0, 0.0],   # → event 0
        [0.0, 1.0, 0.0],   # → event 1
        [1.0, 0.0, 0.0],   # → event 0
        [0.0, 0.0, 1.0]    # → event 2
    ], dtype=torch.float32)

    # Expected sampled event indices (per row)
    expected_idx = torch.tensor([0, 1, 0, 2], dtype=torch.long)

    # Expected child results (first column of C)
    expected_Cs = torch.tensor([[1], [1], [1], [0]], dtype=torch.long)

    # Repeat multiple times to ensure deterministic behavior
    for _ in range(10):
        Cs, event_idx = T.sample_from_p_exp(p_exp)

        # Shape checks
        assert Cs.shape == (4, 1)
        assert event_idx.shape == (4,)

        # Deterministic index match
        assert torch.equal(event_idx, expected_idx), \
            f"\nExpected idx:\n{expected_idx}\nGot:\n{event_idx}"

        # Deterministic Cs match
        assert torch.equal(Cs, expected_Cs), \
            f"\nExpected Cs:\n{expected_Cs}\nGot:\n{Cs}"


def test_sample_from_p_exp2(dict_cpt2):
    # Setup CPT
    T = cpt.Cpt(**dict_cpt2)
    
    # Probability matrix (n_samples = 4, n_events = 5)
    p_exp = torch.tensor([
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5, 0.0],
        [0.9, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0]
    ], dtype=torch.float32)

    # Allowed outcomes for each row
    allowed_child_states = [
        {0, 1},   # row 0 can map to event 0 or 1
        {1, 2},   # row 1 can map to event 2 or 3
        {0, 1},   # row 2 can map to event 0 or 1
        {2},      # row 3 maps to event 4 (deterministic)
    ]

    # Event → child mapping from dict_cpt2
    C = torch.tensor(dict_cpt2["C"], dtype=torch.long)
    n_childs = 1  # A1 only

    # Repeat multiple times to ensure robustness
    for _ in range(10):

        Cs, event_idx = T.sample_from_p_exp(p_exp)

        # Shape check
        assert Cs.shape == (4, 1)
        assert event_idx.shape == (4,)

        Cs_list = Cs.squeeze(1).tolist()
        idx_list = event_idx.tolist()

        for i in range(4):
            child_val = Cs_list[i]
            idx_val = idx_list[i]

            # Check child value is allowed
            assert child_val in allowed_child_states[i], \
                f"Row {i}: got child {child_val}, allowed {allowed_child_states[i]}"

            # Check consistency: child came from the correct row of C
            assert child_val == C[idx_val, 0].item(), \
                f"Row {i}: Cs={child_val}, but C[event_idx]={C[idx_val,0].item()}"

def test_sample1(dict_cpt3):
    # No parents case
    T = cpt.Cpt(**dict_cpt3)

    n_sample = 10

    for _ in range(10):   # repeat multiple times to ensure robustness

        Cs, ps = T.sample(n_sample=n_sample)   # returns (Cs, idx, ps)

        # Shape checks
        assert Cs.shape == (n_sample, 1)
        assert ps.shape == (n_sample,)

        # Composite child states must be 0 or 1
        unique_vals = torch.unique(Cs)
        assert all(int(x) in {0, 1} for x in unique_vals), \
            f"Invalid sampled state detected: {unique_vals}"

        # Check probabilities returned match p[event_idx]
        event_idx = (Cs == 0).long().squeeze(1)   # vectorized
        expected_ps = T.p[event_idx]
        assert torch.allclose(ps, expected_ps.squeeze()), \
            f"ps mismatch:\nexpected {expected_ps}\n got {ps}"

        # Basic sanity: over 10 samples, we expect *at least one* 1 and one 0
        # (rare failures allowed in theory but practically never in 10×10 draws)
        num_ones = (Cs == 1).sum().item()
        num_zeros = (Cs == 0).sum().item()
        assert num_ones > 0, "Sampling produced all zeros — suspicious"
        assert num_zeros > 0, "Sampling produced all ones — suspicious"


def test_sample2(dict_cpt3):
    T = cpt.Cpt(**dict_cpt3)

    n_sample = 1_000_000

    # Fix seed for reproducibility
    torch.manual_seed(123)

    Cs, _ = T.sample(n_sample=n_sample)    # shape (1e6, 1)

    # Count frequencies
    counts = torch.bincount(Cs.squeeze())   # [count_of_0s, count_of_1s]

    freq_0 = counts[0].item() / n_sample
    freq_1 = counts[1].item() / n_sample

    print("Observed frequencies:")
    print("  P(child = 0) =", freq_0)
    print("  P(child = 1) =", freq_1)

    # Assertions allow small variation
    assert abs(freq_0 - 0.3) < 0.01, f"P(0) deviates too much: {freq_0}"
    assert abs(freq_1 - 0.7) < 0.01, f"P(1) deviates too much: {freq_1}"

def test_sample3(dict_cpt5):
    T = cpt.Cpt(**dict_cpt5)

    print("\nCPT Event Matrix (C):")
    print(T.C)

    n_sample = 10
    Cs, ps = T.sample(n_sample=n_sample)

    print("\nSampled child states (Cs):")
    print(Cs)

    assert Cs.shape == (n_sample, 2)
    assert ps.shape == (n_sample,)

    # Allowed CPT rows
    allowed = {
        (2, 0),   # event 0
        (0, 1),   # event 1
        (1, 1),   # event 2
    }

    # Build reverse mapping: Cs tuple → event index
    reverse_map = {
        (2, 0): 0,
        (0, 1): 1,
        (1, 1): 2,
    }

    # Convert Cs to list of tuples
    Cs_list = [tuple(map(int, row)) for row in Cs.tolist()]

    # Extract probabilities for later comparison
    p = T.p.squeeze()   # tensor([0.85, 0.10, 0.05])

    for i, row in enumerate(Cs_list):

        # Check row is valid
        assert row in allowed, f"Invalid sampled row: {row}"

        # Infer event index
        event_idx = reverse_map[row]

        # Expected probability
        expected_p = p[event_idx]

        # Compare against ps[i]
        assert abs(ps[i].item() - expected_p.item()) < 1e-6, \
            f"Row {i}: probability mismatch: expected {expected_p}, got {ps[i]}"


def test_sample4(dict_cpt2):

    # Parent composite samples (A2, A3)
    Cs_pars = torch.tensor([
        [1,1],   # row 0
        [0,1],   # row 1
        [1,0],   # row 2
        [0,0]    # row 3
    ], dtype=torch.long)

    T = cpt.Cpt(**dict_cpt2)

    # Expected probability lookup *per sample*
    # Based only on the sampled CHILD state, not event index
    expected_prob = [
        {0: 0.9, 1: 0.1},     # row 0: child=0 →0.9, child=1 →0.1
        {1: 0.5, 2: 0.5},     # row 1: child=1 →0.5, child=2 →0.5
        {0: 0.9, 1: 0.1},     # row 2: same as row 0
        {2: 1.0},             # row 3: child=2 →1.0 only
    ]

    for _ in range(10):

        Cs, ps = T.sample(Cs_pars=Cs_pars)

        assert Cs.shape == (4, 1)
        assert ps.shape == (4,)

        Cs_list = Cs.squeeze(1).tolist()
        ps_list = ps.tolist()

        for i, child in enumerate(Cs_list):
            # Look up expected probability for that child state
            assert child in expected_prob[i], \
                f"Row {i}: unexpected child {child}"

            exp_p = expected_prob[i][child]

            assert abs(ps_list[i] - exp_p) < 1e-6, \
                f"Row {i}: ps={ps_list[i]}, expected {exp_p}"


def test_sample5(dict_cpt1):
    # Parent composite samples (A2, A3)
    Cs_pars = torch.tensor([
        [1,1],   # only event 0 matches → child 1
        [0,1],   # only event 1 matches → child 1
        [1,0],   # event 0 or 1 → both child 1
        [0,0],   # only event 2 matches → child 0
    ], dtype=torch.long)

    # Setup CPT
    T = cpt.Cpt(**dict_cpt1)

    # Expected child outcomes (deterministic)
    expected_child = [1, 1, 1, 0]

    # Expected probabilities (always 1.0 for dict_cpt1)
    expected_ps = [1.0, 1.0, 1.0, 1.0]

    for _ in range(10):   # multiple repetitions for robustness

        Cs, ps = T.sample(Cs_pars=Cs_pars)

        # Shape checks
        assert Cs.shape == (4, 1)
        assert ps.shape == (4,)

        # Convert to Python lists
        Cs_list = Cs.squeeze(1).tolist()
        ps_list = ps.tolist()

        # Check each sample
        for i in range(4):

            # Check child value
            assert Cs_list[i] == expected_child[i], \
                f"Row {i}: got child {Cs_list[i]}, expected {expected_child[i]}"

            # Check returned probability
            assert abs(ps_list[i] - expected_ps[i]) < 1e-12, \
                f"Row {i}: ps {ps_list[i]}, expected {expected_ps[i]}"
            
def test_log_prob1(dict_cpt1):

    # Build CPT
    mycpt = cpt.Cpt(**dict_cpt1)

    # Cs = [A1, A2, A3] in composite form (child first, then parents)
    Cs = torch.tensor([
        [0, 1, 1],   # row 0: child=0, parents = (1,1)
        [1, 0, 1],   # row 1
        [1, 1, 0],   # row 2
        [1, 0, 0],   # row 3 (not a CPT event; probability should be 0)
    ], dtype=torch.long)

    expected_probs = torch.tensor([
        0.0,
        1.0,
        1.0,
        0.0,
    ])

    expected_logp = torch.log(expected_probs + 1e-15)

    # Run log_prob
    out = mycpt.log_prob(Cs)

    # Compare
    assert torch.allclose(out, expected_logp, atol=1e-6), \
        f"log_prob output incorrect.\nGot: {out}\nExpected: {expected_logp}"


def test_log_prob2(dict_cpt2):
    """
    Tests that log_prob(Cs) returns correct log probabilities
    for the event rows in the CPT.
    """

    # Prepare CPT
    mycpt = cpt.Cpt(**dict_cpt2)

    # Test samples (same as rows)
    Cs = torch.tensor([
        [0, 1, 1],   # event 0
        [1, 0, 1],   # event 1
        [1, 1, 0],   # event 2
        [2, 0, 0],   # event 3
    ], dtype=torch.long)

    # Expected log probabilities
    expected_logp = torch.log(torch.tensor([0.9, 0.5, 0.1, 1.0]))

    # Run log_prob
    out = mycpt.log_prob(Cs)

    # Compare
    assert torch.allclose(out, expected_logp, atol=1e-6), \
        f"log_prob output incorrect.\nGot: {out}\nExpected: {expected_logp}"

def test_log_prob3(dict_cpt5):

    # Build CPT
    mycpt = cpt.Cpt(**dict_cpt5)

    Cs = torch.tensor([
        [1, 0],   # event 0
        [0, 1],   # event 1
        [1, 1],   # event 2
        [0, 0],   # not an event; prob should be 0
    ], dtype=torch.long)

    expected_probs = torch.tensor([
        0.5,
        0.3,
        0.2,
        0.5,
    ])

    expected_logp = torch.log(expected_probs + 1e-15)

    # Run log_prob
    out = mycpt.log_prob(Cs)

    # Compare
    assert torch.allclose(out, expected_logp, atol=1e-6), \
        f"log_prob output incorrect.\nGot: {out}\nExpected: {expected_logp}"
