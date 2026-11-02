import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tbnpy import inference
from tbnpy.compile import CompileError, compile_file, compile_model
from tbnpy.spec import ModelSpec, SpecError, load_spec, save_spec

ABCDE = Path(__file__).resolve().parents[1] / "examples" / "ABCDE" / "abcde.tbn.json"


def minimal(**overrides):
    """A two-node model: discrete root X -> continuous Y."""
    data = {
        "name": "minimal",
        "variables": [
            {"name": "X", "kind": "discrete", "states": [0, 1]},
            {"name": "Y", "kind": "continuous"},
        ],
        "nodes": [
            {"name": "X", "type": "discrete_cpt", "parents": [],
             "probabilities": [0.25, 0.75]},
            {"name": "Y", "type": "distribution", "parents": ["X"],
             "family": "normal", "parameters": {"loc": "X", "scale": "1.0"}},
        ],
    }
    data.update(overrides)
    return ModelSpec.from_dict(data)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def test_minimal_model_validates():
    assert minimal().validate() is not None


def test_cycle_is_reported_with_the_nodes_involved():
    spec = minimal(nodes=[
        {"name": "X", "type": "equation", "parents": ["Y"], "equation": "Y"},
        {"name": "Y", "type": "equation", "parents": ["X"], "equation": "X"},
    ], variables=[{"name": "X", "kind": "continuous"},
                  {"name": "Y", "kind": "continuous"}])
    with pytest.raises(SpecError, match="cycle"):
        spec.validate()


def test_variable_with_no_node_is_rejected():
    spec = minimal()
    spec.variables.append(type(spec.variables[0])(name="Z", kind="continuous"))
    with pytest.raises(SpecError, match="not produced by any node"):
        spec.validate()


def test_two_nodes_cannot_own_the_same_variable():
    data = minimal().to_dict()
    data["nodes"].append({"name": "Y2", "child": "Y", "type": "distribution",
                          "parents": [], "family": "normal",
                          "parameters": {"loc": "0", "scale": "1"}})
    with pytest.raises(SpecError, match="child of both"):
        ModelSpec.from_dict(data).validate()


def test_unknown_parent_is_rejected():
    data = minimal().to_dict()
    data["nodes"][1]["parents"] = ["nope"]
    with pytest.raises(SpecError, match="not a declared variable"):
        ModelSpec.from_dict(data).validate()


def test_cpt_rows_must_sum_to_one():
    data = minimal().to_dict()
    data["nodes"][0]["probabilities"] = [0.25, 0.70]
    with pytest.raises(SpecError, match="sums to"):
        ModelSpec.from_dict(data).validate()


def test_cpt_row_count_must_match_the_parents():
    data = minimal().to_dict()
    data["variables"].append({"name": "Z", "kind": "discrete", "states": [0, 1, 2]})
    data["nodes"].append({"name": "Z", "type": "discrete_cpt", "parents": ["X"],
                          "probabilities": [[0.2, 0.3, 0.5]]})   # needs 2 rows
    with pytest.raises(SpecError, match="expected 2 CPT row"):
        ModelSpec.from_dict(data).validate()


def test_discrete_cpt_rejects_a_continuous_parent():
    data = minimal().to_dict()
    data["variables"].append({"name": "Z", "kind": "discrete", "states": [0, 1]})
    data["nodes"].append({"name": "Z", "type": "discrete_cpt", "parents": ["Y"],
                          "probabilities": [[0.5, 0.5], [0.5, 0.5]]})
    with pytest.raises(SpecError, match="discretise it first"):
        ModelSpec.from_dict(data).validate()


def test_unknown_node_type_lists_the_valid_ones():
    data = minimal().to_dict()
    data["nodes"][0]["type"] = "magic"
    with pytest.raises(SpecError, match="expected one of"):
        ModelSpec.from_dict(data)


def test_unexpected_field_is_rejected_rather_than_ignored():
    data = minimal().to_dict()
    data["nodes"][0]["equation"] = "X"       # wrong field for a discrete_cpt
    with pytest.raises(SpecError, match="unexpected field"):
        ModelSpec.from_dict(data)


def test_future_spec_version_is_refused():
    data = minimal().to_dict()
    data["spec_version"] = 99
    with pytest.raises(SpecError, match="version 99"):
        ModelSpec.from_dict(data)


def test_topological_order_puts_parents_first():
    order = minimal().topological_order()
    assert order.index("X") < order.index("Y")


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------
def test_json_round_trip_preserves_the_model(tmp_path):
    spec = load_spec(ABCDE)
    path = save_spec(spec, tmp_path / "rt.tbn.json")
    again = load_spec(path)
    assert again.to_dict() == spec.to_dict()


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def test_compiled_abcde_matches_the_hand_written_model():
    """The declarative spec must reproduce s1_define_model.py exactly."""
    from examples.ABCDE import s1_define_model as hand

    _, _, probs = compile_file(ABCDE)
    hv = hand.define_variables()
    hp = hand.define_probs(hv)

    rows = torch.tensor([[0.5, 1.0, 2.0], [-1.0, 0.0, 0.0], [2.4, 0.0, 1.0]])
    assert torch.allclose(probs["C"].log_prob(rows), hp["C"].log_prob(rows))

    pars = torch.tensor([[1.0, 2.0]] * 4000)
    torch.manual_seed(0)
    mine, _ = probs["C"].sample(Cs_pars=pars)
    torch.manual_seed(0)
    theirs, _ = hp["C"].sample(Cs_pars=pars)
    assert torch.allclose(mine, theirs)

    gate = torch.tensor([[1.7, 1.7, 0.0], [0.0, 1.7, 1.0], [0.0, 1.7, 0.0]])
    assert probs["E"].log_prob(gate).tolist() == hp["E"].log_prob(gate).tolist()


def test_compiled_model_runs_through_inference():
    _, _, probs = compile_file(ABCDE)
    out = inference.sample(probs, ["E", "A"], n_sample=500)
    assert out["E"].Cs.shape[0] == 500

    ev = inference.sample_evidence(probs, ["A", "C"], 200,
                                   pd.DataFrame({"OC": [-0.3, 0.4]}))
    assert ev["C"].Cs.shape == (2, 200, 3)
    assert torch.isfinite(ev["C"].ps).all()


def test_equation_node_is_deterministic_and_rejects_inconsistent_values():
    _, _, probs = compile_file(ABCDE)
    Cs, logp = probs["E"].sample(Cs_pars=torch.tensor([[1.7, 0.0], [1.7, 1.0]]))
    assert Cs.tolist() == pytest.approx([1.7, 0.0])
    assert logp.tolist() == [0.0, 0.0]
    bad = probs["E"].log_prob(torch.tensor([[9.0, 1.7, 0.0]]))
    assert bad.item() == -float("inf")


def test_equation_tolerance_widens_the_accepted_band():
    data = {
        "name": "tol",
        "variables": [{"name": "P", "kind": "continuous"},
                      {"name": "Q", "kind": "continuous"}],
        "nodes": [
            {"name": "P", "type": "distribution", "parents": [],
             "family": "uniform", "parameters": {"low": "0", "high": "1"}},
            {"name": "Q", "type": "equation", "parents": ["P"],
             "equation": "P * 2", "tolerance": 0.05},
        ],
    }
    _, probs = compile_model(ModelSpec.from_dict(data))
    near = probs["Q"].log_prob(torch.tensor([[1.02, 0.5]]))
    far = probs["Q"].log_prob(torch.tensor([[1.20, 0.5]]))
    assert near.item() == 0.0 and far.item() == -float("inf")


def test_discretise_bins_a_continuous_parent():
    data = {
        "name": "disc",
        "variables": [
            {"name": "V", "kind": "continuous"},
            {"name": "B", "kind": "discrete", "states": ["low", "mid", "high"]},
        ],
        "nodes": [
            {"name": "V", "type": "distribution", "parents": [],
             "family": "uniform", "parameters": {"low": "0", "high": "3"}},
            {"name": "B", "type": "discretise", "parents": ["V"],
             "intervals": [[0, 1], [1, 2], [2, 3]]},
        ],
    }
    _, probs = compile_model(ModelSpec.from_dict(data))
    Cs, _ = probs["B"].sample(Cs_pars=torch.tensor([[0.5], [1.5], [2.5]]))
    assert Cs.tolist() == [0, 1, 2]
    consistent = probs["B"].log_prob(torch.tensor([[1.0, 1.5]]))
    wrong = probs["B"].log_prob(torch.tensor([[0.0, 1.5]]))
    assert consistent.item() == 0.0 and wrong.item() == -float("inf")


def test_overlapping_intervals_are_rejected():
    with pytest.raises(SpecError, match="overlaps"):
        ModelSpec.from_dict({
            "name": "bad",
            "variables": [{"name": "V", "kind": "continuous"},
                          {"name": "B", "kind": "discrete", "states": ["a", "b"]}],
            "nodes": [
                {"name": "V", "type": "distribution", "parents": [],
                 "family": "uniform", "parameters": {"low": "0", "high": "3"}},
                {"name": "B", "type": "discretise", "parents": ["V"],
                 "intervals": [[0, 2], [1, 3]]},
            ],
        })


def test_unknown_family_lists_the_available_ones():
    data = minimal().to_dict()
    data["nodes"][1]["family"] = "gaussianish"
    with pytest.raises(CompileError, match="unknown family"):
        compile_model(ModelSpec.from_dict(data))


def test_family_must_match_the_child_kind():
    data = minimal().to_dict()
    data["nodes"][1]["family"] = "categorical"
    data["nodes"][1]["parameters"] = {"probs": ["0.5", "0.5"]}
    with pytest.raises(CompileError, match="discrete child"):
        compile_model(ModelSpec.from_dict(data))


def test_missing_distribution_parameter_is_reported():
    data = minimal().to_dict()
    data["nodes"][1]["parameters"] = {"loc": "X"}
    with pytest.raises(CompileError, match="missing"):
        compile_model(ModelSpec.from_dict(data))


def test_log_prob_reports_a_wrong_column_count():
    _, probs = compile_model(minimal())
    with pytest.raises(CompileError, match="columns"):
        probs["Y"].log_prob(torch.tensor([[1.0]]))


def test_metadata_survives_compilation_for_bespoke_algorithms():
    data = minimal().to_dict()
    data["nodes"][1]["metadata"] = {"solver_hint": "block-update with X"}
    _, probs = compile_model(ModelSpec.from_dict(data))
    assert probs["Y"].spec_metadata["solver_hint"] == "block-update with X"


def test_analysis_hints_are_optional_and_do_not_affect_compilation():
    with_hints = compile_model(load_spec(ABCDE))[1]
    spec = load_spec(ABCDE)
    spec.analysis = None
    without = compile_model(spec)[1]
    assert sorted(with_hints) == sorted(without)


# ---------------------------------------------------------------------------
# Distribution families
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("family,params,lo,hi", [
    ("uniform", {"low": "2.0", "high": "5.0"}, 2.0, 5.0),
    ("loguniform", {"low": "10.0", "high": "1000.0"}, 10.0, 1000.0),
])
def test_bounded_families_sample_inside_their_support(family, params, lo, hi):
    data = {
        "name": "fam",
        "variables": [{"name": "V", "kind": "continuous"}],
        "nodes": [{"name": "V", "type": "distribution", "parents": [],
                   "family": family, "parameters": params}],
    }
    _, probs = compile_model(ModelSpec.from_dict(data))
    torch.manual_seed(0)
    Cs, logp = probs["V"].sample(n_sample=5000)
    assert Cs.min() >= lo and Cs.max() <= hi
    assert torch.isfinite(logp).all()


def test_root_sample_requires_n_sample():
    # A generated root node; discrete_cpt roots go through tbnpy.cpt.Cpt,
    # which raises its own assertion for the same mistake.
    data = {
        "name": "root",
        "variables": [{"name": "V", "kind": "continuous"}],
        "nodes": [{"name": "V", "type": "distribution", "parents": [],
                   "family": "normal", "parameters": {"loc": "0", "scale": "1"}}],
    }
    _, probs = compile_model(ModelSpec.from_dict(data))
    with pytest.raises(CompileError, match="n_sample"):
        probs["V"].sample()


def test_child_sample_requires_parent_columns():
    _, probs = compile_model(minimal())
    with pytest.raises(CompileError, match="Cs_pars"):
        probs["Y"].sample(n_sample=10)
