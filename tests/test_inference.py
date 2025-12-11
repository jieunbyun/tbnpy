import numpy as np
import os
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

def test_sample1():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    sampled_probs = inference.sample(probs, query_nodes=['E'], n_sample=3)

    # Check that sampled Cs and ps exist for each ancestor
    for node in ['A', 'B', 'C', 'D', 'E']:
        assert hasattr(sampled_probs[node], 'Cs'), f"Node {node} missing sampled Cs"
        assert hasattr(sampled_probs[node], 'ps'), f"Node {node} missing sampled ps"

def test_sample2():
    varis = s1_define_model.define_variables()
    probs = s1_define_model.define_probs(varis, device=device)

    sampled_probs = inference.sample(probs, query_nodes=['OC'], n_sample=3)

    assert sampled_probs.keys() == {'A', 'B', 'C', 'OC'}

    # Check that sampled Cs and ps exist for each ancestor
    for node in ['A', 'B', 'C', 'OC']:
        assert hasattr(sampled_probs[node], 'Cs'), f"Node {node} missing sampled Cs"
        assert hasattr(sampled_probs[node], 'ps'), f"Node {node} missing sampled ps"

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

    # --------------------------------------------------------
    # A2 — OBSERVED VARIABLE
    # --------------------------------------------------------
    A2 = sampled_probs['A2']

    assert A2.Cs.shape == (n_evi, n_sample, 1)

    # All entries must be 1
    assert torch.all(A2.Cs[:, :, 0] == 1)

    # log probability of A2 = 1 in its CPT: log(0.7)
    expected_logp_A2 = torch.log(torch.tensor(0.7))

    assert torch.allclose(
        A2.ps,
        torch.full((n_evi, n_sample), expected_logp_A2),
        atol=1e-6
    )


    # --------------------------------------------------------
    # A3 — ROOT VARIABLE
    # --------------------------------------------------------
    A3 = sampled_probs['A3']

    assert A3.Cs.shape == (n_evi, n_sample, 1)

    vals = A3.Cs[:, :, 0]
    assert torch.all((vals == 0) | (vals == 1))

    for i in range(n_evi):
        for j in range(n_sample):
            v = int(vals[i, j].item())
            p = 0.6 if v == 0 else 0.4
            expected_logp = torch.log(torch.tensor(p))

            assert abs(A3.ps[i, j].item() - expected_logp.item()) < 1e-6


    # --------------------------------------------------------
    # A1 — CHILD OF A2 AND A3
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

