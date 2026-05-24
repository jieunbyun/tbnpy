import numpy as np
import os
import pytest
from tbnpy import cpt, variable
from examples.ABCDE import c, e, s1_define_model
from tbnpy import inference
from .test_cpt import dict_cpt2
import pandas as pd
import torch

device = ('cuda' if os.environ.get('USE_CUDA', '0') == '1' else 'cpu')

def test_find_ancestor_order1():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    # Run test for ancestors of C
    result = inference.get_ancestor_order(probs, query_nodes=['C'])

    # A and B must come before C
    pos = {node: i for i, node in enumerate(result)}
    assert pos['A'] < pos['C']
    assert pos['B'] < pos['C']

def test_find_ancestor_order2():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    # Run test for ancestors of C
    result = inference.get_ancestor_order(probs, query_nodes=['C', 'D'])

    assert set(result) == {'A', 'B', 'C', 'D'}

    pos = {node: i for i, node in enumerate(result)}

    assert pos['A'] < pos['C']
    assert pos['B'] < pos['C']

def test_find_ancestor_order3():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    # Run test for ancestors of A
    result = inference.get_ancestor_order(probs, query_nodes=['A'])

    expected = ['A']

    assert result == expected, f"Expected {expected}, got {result}"

def test_find_ancestor_order4():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    # Run test for ancestors of OC
    result = inference.get_ancestor_order(probs, query_nodes=['OC'])

    assert set(result) == {'A', 'B', 'C', 'OC'}

    pos = {node: i for i, node in enumerate(result)}
    assert pos['A'] < pos['C']
    assert pos['B'] < pos['C']
    assert pos['C'] < pos['OC']


# ---------------------------------------------------------------------------
# Multi-child node support for get_ancestor_order
# ---------------------------------------------------------------------------
#
# The tests below use small mock objects that expose only the attributes
# get_ancestor_order reads (`.name`, `.childs`, `.parents`). This isolates
# the topological-ordering logic from any concrete probability backend.
# ---------------------------------------------------------------------------

class _MockVar:
    def __init__(self, name):
        self.name = name


class _MockProb:
    def __init__(self, childs, parents):
        self.childs = childs
        self.parents = parents


def test_find_ancestor_order_multi_child_basic():
    # N1 owns two child variables [X, Y]; N2 has X as a parent.
    X = _MockVar('X')
    Y = _MockVar('Y')
    P = _MockVar('P')

    probs = {
        'N1': _MockProb(childs=[X, Y], parents=[]),
        'N2': _MockProb(childs=[P],    parents=[X]),
    }

    result = inference.get_ancestor_order(probs, query_nodes=['N2'])

    assert set(result) == {'N1', 'N2'}
    pos = {node: i for i, node in enumerate(result)}
    assert pos['N1'] < pos['N2']


def test_find_ancestor_order_multi_child_diamond():
    # N1 owns [X, Y]; N2 lists BOTH X and Y as parents.
    # Edge N1 -> N2 must be counted exactly once, so indegree[N2] == 1
    # and Kahn's algorithm produces a complete ordering.
    X = _MockVar('X')
    Y = _MockVar('Y')
    Z = _MockVar('Z')

    probs = {
        'N1': _MockProb(childs=[X, Y], parents=[]),
        'N2': _MockProb(childs=[Z],    parents=[X, Y]),
    }

    result = inference.get_ancestor_order(probs, query_nodes=['N2'])

    assert result == ['N1', 'N2']


def test_find_ancestor_order_multi_child_chain():
    # N1 -> {N2, N3} via X, Y; then N2 and N3 feed into N4 via P, Q.
    #
    #         N1
    #        /  \
    #       X    Y
    #       |    |
    #       N2   N3
    #       |    |
    #       P    Q
    #        \  /
    #         N4 -> R
    X = _MockVar('X')
    Y = _MockVar('Y')
    P = _MockVar('P')
    Q = _MockVar('Q')
    R = _MockVar('R')

    probs = {
        'N1': _MockProb(childs=[X, Y], parents=[]),
        'N2': _MockProb(childs=[P],    parents=[X]),
        'N3': _MockProb(childs=[Q],    parents=[Y]),
        'N4': _MockProb(childs=[R],    parents=[P, Q]),
    }

    result = inference.get_ancestor_order(probs, query_nodes=['N4'])

    assert set(result) == {'N1', 'N2', 'N3', 'N4'}
    pos = {node: i for i, node in enumerate(result)}
    assert pos['N1'] < pos['N2']
    assert pos['N1'] < pos['N3']
    assert pos['N2'] < pos['N4']
    assert pos['N3'] < pos['N4']


def test_find_ancestor_order_duplicate_child_variable_rejected():
    # A variable that appears as a child of two different probability objects
    # is ambiguous and must be rejected.
    X = _MockVar('X')

    probs = {
        'N1': _MockProb(childs=[X], parents=[]),
        'N2': _MockProb(childs=[X], parents=[]),
    }

    with pytest.raises(AssertionError, match="appears as child of more than one"):
        inference.get_ancestor_order(probs, query_nodes=['N1'])


def test_sample1():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    sampled_probs = inference.sample(probs, query_nodes=['E'], n_sample=3)

    # Only query nodes are returned
    assert sampled_probs.keys() == {'E'}
    assert hasattr(sampled_probs['E'], 'Cs'), "Node E missing sampled Cs"
    assert hasattr(sampled_probs['E'], 'ps'), "Node E missing sampled ps"

def test_sample2():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    sampled_probs = inference.sample(probs, query_nodes=['OC'], n_sample=3)

    # Only query nodes are returned
    assert sampled_probs.keys() == {'OC'}
    assert hasattr(sampled_probs['OC'], 'Cs'), "Node OC missing sampled Cs"
    assert hasattr(sampled_probs['OC'], 'ps'), "Node OC missing sampled ps"

def test_sample_batch_size():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    n_sample = 7
    sampled_probs = inference.sample(
        probs,
        query_nodes=['OC'],
        n_sample=n_sample,
        batch_size=2,
    )

    # Only query node is returned; verify sample count
    assert sampled_probs.keys() == {'OC'}
    assert sampled_probs['OC'].Cs.shape[0] == n_sample

def test_sample_evidence1(dict_cpt2):

    # --------------------------------------------------------
    # Build BN: A2 → A1 ← A3
    # --------------------------------------------------------
    mycpt = {}

    mycpt['A1'] = cpt.Cpt(**dict_cpt2)

    mycpt['A2'] = cpt.Cpt(
        childs=[mycpt['A1'].parents[0]],
        C=np.array([[0], [1]]),
        p=np.array([0.3, 0.7]),
        device=device
    )

    mycpt['A3'] = cpt.Cpt(
        childs=[mycpt['A1'].parents[1]],
        C=np.array([[0], [1]]),
        p=np.array([0.6, 0.4]),
        device=device
    )

    # --------------------------------------------------------
    # Evidence: A2 = 1 for all 3 rows
    # --------------------------------------------------------
    evidence_df = pd.DataFrame({'A2': [1, 1, 1]})

    n_sample = 2
    n_evi = len(evidence_df)

    sampled_probs = inference.sample_evidence(
        probs=mycpt,
        query_nodes=['A1'],
        n_sample=n_sample,
        evidence_df=evidence_df
    )

    # Only query node is returned
    assert sampled_probs.keys() == {'A1'}

    # --------------------------------------------------------
    # A1 — CHILD OF A2 AND A3 (query node)
    # --------------------------------------------------------
    A1 = sampled_probs['A1']

    assert A1.Cs.dim() == 3
    assert A1.Cs.shape == (n_evi, n_sample, 3)  # (A1, A2, A3)

    # Probability map from dict_cpt2
    expected_map = {
        (0, 1, 0): 0.9,
        (1, 1, 0): 0.1,
        (0, 1, 1): 0.9,
        (1, 1, 1): 0.1,
    }

    for i in range(n_evi):
        for j in range(n_sample):
            triple = tuple(int(x) for x in A1.Cs[i, j, :].tolist())
            assert triple in expected_map, f"Unexpected A1 triple {triple}"

            expected_logp = torch.log(torch.tensor(expected_map[triple]))

            assert abs(A1.ps[i, j].item() - expected_logp.item()) < 1e-6

def test_sample_evidence_batch_size(dict_cpt2):

    mycpt = {}

    mycpt['A1'] = cpt.Cpt(**dict_cpt2)

    mycpt['A2'] = cpt.Cpt(
        childs=[mycpt['A1'].parents[0]],
        C=np.array([[0], [1]]),
        p=np.array([0.3, 0.7]),
        device=device
    )

    mycpt['A3'] = cpt.Cpt(
        childs=[mycpt['A1'].parents[1]],
        C=np.array([[0], [1]]),
        p=np.array([0.6, 0.4]),
        device=device
    )

    evidence_df = pd.DataFrame({'A2': [1, 1, 1]})
    n_sample = 5
    n_evi = len(evidence_df)

    sampled_probs = inference.sample_evidence(
        probs=mycpt,
        query_nodes=['A1'],
        n_sample=n_sample,
        evidence_df=evidence_df,
        batch_size=2,
    )

    # Only query node is returned
    assert sampled_probs.keys() == {'A1'}
    assert sampled_probs['A1'].Cs.shape == (n_evi, n_sample, 3)

