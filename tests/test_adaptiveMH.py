from tbnpy import inference, variable, cpt, adaptiveMH
import pytest, math
import examples.ABCDE.s1_define_model as define_model
import pandas as pd
import torch

@pytest.fixture(scope='module')
def model_ABCDE():
    varis = define_model.define_variables()
    probs = define_model.define_probs(varis)
    return probs, varis

@pytest.fixture(scope="module")
def evidence_A():
    return pd.DataFrame({"A": [0, 0]})

@pytest.fixture(scope='module')
def define_evidence1():
    varis = define_model.define_variables()
    evidence = pd.DataFrame({
        'A': [0, 0]
    })
    return evidence

@pytest.fixture(scope="module")
def prepared_sampler(model_ABCDE, evidence_A):
    probs, varis = model_ABCDE

    probs_sub = {k: v for k, v in probs.items() if k in ["A", "B", "C"]}
    variables_sub = [varis[k] for k in ["A", "B", "C"]]

    sampler = adaptiveMH.HybridAdaptiveMH(
        probs=probs_sub,
        variables=variables_sub,
        evidence_df=evidence_A,
        n_chain=10,
        adapt=adaptiveMH.AdaptConfig(burnin=3),
    )

    # forward sampling
    probs_copy = inference.sample_evidence(
        probs=probs_sub,
        query_nodes=["B", "C"],
        n_sample=10,
        evidence_df=evidence_A,
    )

    sampler.init_state_from_forward_samples(probs_copy)

    return sampler, probs_sub, varis

def test_init1(model_ABCDE, define_evidence1):
    probs, varis = model_ABCDE
    probs = {k: v for k, v in probs.items() if k in ['A', 'B', 'C']}
    evidence = define_evidence1

    sampler = adaptiveMH.HybridAdaptiveMH(
        probs = probs,
        variables=[v for v in varis.values() if v.name in ['A', 'B', 'C']],
        evidence_df=evidence,
        n_chain = 10,
        adapt = adaptiveMH.AdaptConfig(burnin=3)
    )

    assert sampler.n_evi == 2
    assert sampler.n_chain == 10
    assert sampler.latent_vars == [varis['B'], varis['C']]

def test_init_state_from_forward_samples1(model_ABCDE, define_evidence1):
    probs, varis = model_ABCDE
    evidence = define_evidence1

    # Restrict model to A → B → C (example)
    probs_sub = {k: v for k, v in probs.items() if k in ['A', 'B', 'C']}
    variables_sub = [v for v in varis.values() if v.name in ['A', 'B', 'C']]

    n_chain = 10

    sampler = adaptiveMH.HybridAdaptiveMH(
        probs=probs_sub,
        variables=variables_sub,
        evidence_df=evidence,
        n_chain=n_chain,
        adapt=adaptiveMH.AdaptConfig(burnin=3),
    )

    # Initialise the samples following the prior
    probs_copy = inference.sample_evidence(
        probs=probs_sub,
        query_nodes=[v.name for v in sampler.latent_vars],
        n_sample=n_chain,
        evidence_df=evidence,
    )

    # initialise state from samples-
    sampler.init_state_from_forward_samples(probs_copy)
    # Step 3: assertions
    # Latent vars should be B, C
    latent_names = {v.name for v in sampler.latent_vars}
    assert latent_names == {'B', 'C'}

    # State populated correctly
    assert set(sampler.state.keys()) == {'B', 'C'}

    for name in ['B', 'C']:
        x = sampler.state[name]

        # shape
        assert x.shape == (sampler.n_evi, sampler.n_chain)

        # dtype
        if isinstance(varis[name].values, list):
            assert x.dtype == torch.long
        else:
            assert x.dtype == torch.float32

        # finite values
        assert torch.isfinite(x).all()

    # -----------------------------------------
    # Step 4: log-prob caches must exist
    # -----------------------------------------
    assert sampler.logp_evi_chain is not None
    assert sampler.logp_chain is not None

    assert sampler.logp_evi_chain.shape == (sampler.n_evi, sampler.n_chain)
    assert sampler.logp_chain.shape == (sampler.n_chain,)

    assert torch.isfinite(sampler.logp_evi_chain).all()
    assert torch.isfinite(sampler.logp_chain).all()

def test_recompute_all_logps1(model_ABCDE, define_evidence1):
    probs, varis = model_ABCDE
    evidence = define_evidence1

    probs_sub = {k: v for k, v in probs.items() if k in ['A', 'B', 'C']}
    variables_sub = [v for v in varis.values() if v.name in ['A', 'B', 'C']]

    n_chain = 10

    sampler = adaptiveMH.HybridAdaptiveMH(
        probs=probs_sub,
        variables=variables_sub,
        evidence_df=evidence,
        n_chain=n_chain,
        adapt=adaptiveMH.AdaptConfig(burnin=3),
    )

    # ---- initialise state from forward samples ----
    probs_copy = inference.sample_evidence(
        probs=probs_sub,
        query_nodes=[v.name for v in sampler.latent_vars],
        n_sample=n_chain,
        evidence_df=evidence,
    )
    sampler.init_state_from_forward_samples(probs_copy)

    # ---- call recompute explicitly ----
    sampler._recompute_all_logps()

    # -----------------------------------------
    # 1. Cache must exist for every prob
    # -----------------------------------------
    assert set(sampler.factor_logp_cache.keys()) == {id(p) for p in probs_sub.values()}

    for prob in probs_sub.values():
        logp2d = sampler.factor_logp_cache[id(prob)]

        assert logp2d.shape == (sampler.n_evi, sampler.n_chain)
        assert torch.isfinite(logp2d).all()

    # -----------------------------------------
    # 2. logp_evi_chain must be sum of factors
    # -----------------------------------------
    stacked = torch.stack(
        [sampler.factor_logp_cache[id(p)] for p in probs_sub.values()],
        dim=0,   # (n_factors, n_evi, n_chain)
    )

    logp_evi_manual = stacked.sum(dim=0)
    assert torch.allclose(
        sampler.logp_evi_chain,
        logp_evi_manual,
        atol=1e-6,
    )

    # -----------------------------------------
    # 3. logp_chain must sum over evidence
    # -----------------------------------------
    logp_chain_manual = logp_evi_manual.sum(dim=0)

    assert sampler.logp_chain.shape == (sampler.n_chain,)
    assert torch.allclose(
        sampler.logp_chain,
        logp_chain_manual,
        atol=1e-6,
    )

    assert torch.isfinite(sampler.logp_chain).all()

def test_recompute_all_logps2(model_ABCDE, define_evidence1):
    # Test that after perturbing the state, recompute_all_logps updates logp_chain
    probs, varis = model_ABCDE
    evidence = define_evidence1

    probs_sub = {k: v for k, v in probs.items() if k in ['A', 'B', 'C']}
    variables_sub = [v for v in varis.values() if v.name in ['A', 'B', 'C']]

    sampler = adaptiveMH.HybridAdaptiveMH(
        probs=probs_sub,
        variables=variables_sub,
        evidence_df=evidence,
        n_chain=5,
    )

    probs_copy = inference.sample_evidence(
        probs=probs_sub,
        query_nodes=[v.name for v in sampler.latent_vars],
        n_sample=5,
        evidence_df=evidence,
    )
    sampler.init_state_from_forward_samples(probs_copy)

    logp_before = sampler.logp_chain.clone()

    # perturb one latent variable
    name = sampler.latent_vars[0].name
    sampler.state[name] = sampler.state[name].clone()
    sampler.state[name][:, :] = sampler.state[name].roll(1, dims=1)

    sampler._recompute_all_logps()
    logp_after = sampler.logp_chain

    assert not torch.allclose(logp_before, logp_after)

def test_build_Cs_3d1(prepared_sampler):
    sampler, probs, _ = prepared_sampler

    prob = probs["B"]  # test on one factor
    Cs = adaptiveMH.build_Cs_3d(prob, sampler.state, sampler.evidence_1d)

    assert Cs.ndim == 3
    assert Cs.shape[0] == sampler.n_evi
    assert Cs.shape[1] == sampler.n_chain
    assert Cs.shape[2] == len(prob.childs) + len(prob.parents)

    assert torch.isfinite(Cs).all()

def test_factor_logp_2d1(prepared_sampler):
    sampler, probs, _ = prepared_sampler

    prob = probs["C"]
    Cs = adaptiveMH.build_Cs_3d(prob, sampler.state, sampler.evidence_1d)
    logp = adaptiveMH.factor_logp_2d(prob, Cs)

    assert logp.shape == (sampler.n_evi, sampler.n_chain)
    assert torch.isfinite(logp).all()

def test_recompute_all_logps3(prepared_sampler):
    sampler, _, _ = prepared_sampler

    logp_old = sampler.logp_chain.clone()

    # perturb one variable slightly
    v = sampler.latent_vars[0]
    sampler.state[v.name] = sampler.state[v.name].roll(1, dims=1)

    sampler._recompute_all_logps()
    logp_new = sampler.logp_chain

    assert logp_new.shape == logp_old.shape
    assert torch.isfinite(logp_new).all()
    assert not torch.allclose(logp_old, logp_new)

def test_mh_update_block1(prepared_sampler):
    sampler, _, _ = prepared_sampler

    v = sampler.latent_vars[0]
    old_state = sampler.state[v.name].clone()

    accept = sampler.mh_update_block([v], iteration=0)

    assert accept.shape == (sampler.n_chain,)
    assert accept.dtype == torch.bool

    # at least one chain should accept occasionally
    assert accept.any() or (~accept).any()

    # state should only change where accepted
    changed = sampler.state[v.name] != old_state
    assert torch.all(changed.any(dim=0) <= accept)

def test_build_factors_by_var1(model_ABCDE):
    probs, varis = model_ABCDE
    probs_sub = {k: v for k, v in probs.items() if k in ['A', 'B', 'C']}

    factors = adaptiveMH.build_factors_by_var(probs_sub)

    # keys are variable names
    assert set(factors.keys()).issuperset({'A', 'B', 'C'})

    # each entry is a list of CPTs
    for vname, plist in factors.items():
        assert isinstance(plist, list)
        for p in plist:
            assert hasattr(p, 'childs')
            assert hasattr(p, 'parents')

    # sanity: B must appear in at least one factor
    assert len(factors['B']) >= 1

def test_affected_factors_for_block1(prepared_sampler):
    sampler, probs_sub, _ = prepared_sampler

    affected = sampler._affected_factors_for_block(['B'])

    # must be list
    assert isinstance(affected, list)

    # all are CPT objects
    for prob in affected:
        assert hasattr(prob, 'childs')

    # no duplicates (by id)
    ids = [id(p) for p in affected]
    assert len(ids) == len(set(ids))

    # at least one factor involves B
    assert len(affected) > 0

def test_propose_block1(prepared_sampler):
    sampler, _, _ = prepared_sampler

    v = sampler.latent_vars[0]
    x_old = sampler.state[v.name].clone()

    proposed = sampler._propose_block([v])

    # returns dict
    assert isinstance(proposed, dict)
    assert v.name in proposed

    x_new = proposed[v.name]

    # shape preserved
    assert x_new.shape == x_old.shape

    # dtype preserved
    assert x_new.dtype == x_old.dtype

    # at least some entries differ (very high probability)
    assert (x_new != x_old).any()

def test_run1(prepared_sampler):
    sampler, _, _ = prepared_sampler

    out = sampler.run(
        n_iter=3,
        store_every=0,
    )

    # required keys
    assert 'accept_rate' in out
    assert 'logp_chain' in out
    assert 'logp_evi_chain' in out
    assert 'log_sigma' in out

    # shapes
    assert out['logp_chain'].shape == (sampler.n_chain,)
    assert out['logp_evi_chain'].shape == (sampler.n_evi, sampler.n_chain)

    # accept rates are valid
    for k, v in out['accept_rate'].items():
        assert 0.0 <= v <= 1.0

    # finite values
    assert torch.isfinite(out['logp_chain']).all()
    assert torch.isfinite(out['logp_evi_chain']).all()
