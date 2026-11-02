from pathlib import Path

import numpy as np
import pytest
import torch

import tbnpy
from tbnpy import cpt, inference
from tbnpy.compile import compile_file
from tbnpy.probability import (ModelError, Probability, check_model, describe,
                               is_probability, statement)
from tbnpy.variable import Variable

ABCDE = Path(__file__).resolve().parents[1] / "examples" / "ABCDE" / "abcde.tbn.json"


# ---------------------------------------------------------------------------
# Small models to break on purpose
# ---------------------------------------------------------------------------
class Shift(Probability):
    """Y | X ~ Normal(X, 1), or whatever the overrides make it."""

    def __init__(self, childs, parents, sample=None, log_prob=None, device="cpu"):
        super().__init__(childs, parents, device)
        self._sample = sample
        self._log_prob = log_prob

    def form(self):
        return "normal(loc=X, scale=1)"

    def sample(self, Cs_pars=None, n_sample=None):
        if self._sample is not None:
            return self._sample(Cs_pars)
        dist = torch.distributions.Normal(Cs_pars[:, 0].float(), 1.0)
        Cs = dist.sample()
        return Cs, dist.log_prob(Cs)

    def log_prob(self, Cs):
        if self._log_prob is not None:
            return self._log_prob(Cs)
        return torch.distributions.Normal(Cs[:, 1].float(), 1.0).log_prob(Cs[:, 0])


def two_nodes(**overrides):
    """A discrete root X -> continuous Y, with Y's object made by ``Shift``."""
    X = Variable("X", [0, 1])
    Y = Variable("Y", (-torch.inf, torch.inf))
    probs = {
        "X": cpt.Cpt(childs=[X], C=np.array([[0], [1]]), p=np.array([0.5, 0.5])),
        "Y": Shift([Y], [X], **overrides),
    }
    return {"X": X, "Y": Y}, probs


class Bare:
    """Follows the protocol without inheriting anything."""

    def __init__(self, childs, parents):
        self.childs, self.parents, self.device = childs, parents, "cpu"

    def sample(self, Cs_pars=None, n_sample=None):
        n = n_sample if Cs_pars is None else Cs_pars.shape[0]
        return torch.zeros(n), torch.zeros(n)

    def log_prob(self, Cs):
        return torch.zeros(Cs.shape[0])


# ---------------------------------------------------------------------------
# Readings
# ---------------------------------------------------------------------------
def test_variable_repr_is_one_evaluable_line():
    assert repr(Variable("A", [-0.3, 0.3])) == "Variable(name='A', values=[-0.3, 0.3])"
    assert repr(Variable("C", (0.0, 1.0))) == "Variable(name='C', values=(0.0, 1.0))"


def test_repr_names_the_distribution_and_its_form():
    _, probs = two_nodes()
    assert repr(probs["Y"]) == "Shift(P(Y | X): normal(loc=X, scale=1))"
    assert repr(probs["X"]) == "Cpt(P(X): table of 2 rows)"


def test_a_dict_of_objects_prints_as_a_list_of_distributions():
    _, _, probs = compile_file(ABCDE)
    text = repr(probs)
    assert "DistributionSpecNode(P(C | A, B): normal(loc=A + B, scale=0.6))" in text
    assert "EquationSpecNode(P(E | C, D): E = If(D == 0, C, 0))" in text
    assert "object at 0x" not in text


def test_str_gives_each_variable_its_domain():
    _, probs = two_nodes()
    text = str(probs["Y"])
    assert text.splitlines()[0] == "P(Y | X)  [Shift]"
    assert "form    normal(loc=X, scale=1)" in text
    assert "child   Y  continuous (-inf, inf)" in text
    assert "parent  X  discrete, 2 states: 0, 1" in text


def test_cpt_str_shows_its_table_with_labels():
    A = Variable("A", ["fail", "survive"])
    B = Variable("B", ["low", "mid", "high"])
    # B's composite state 3 is the set {low, mid}.
    table = cpt.Cpt(childs=[A], parents=[B], C=[[0, 3], [1, 3], [1, 2]], p=[0.2, 0.8, 1.0])
    lines = str(table).splitlines()
    assert lines[0] == "P(A | B)  [Cpt]"
    assert lines[-4].split() == ["A", "B", "p"]
    assert lines[-3].split() == ["fail", "{low,", "mid}", "0.2"]
    assert lines[-1].split() == ["survive", "high", "1"]


def test_describe_reads_any_protocol_object():
    X = Variable("X", [0, 1])
    bare = Bare([X], [])
    assert is_probability(bare)
    assert describe(bare).splitlines() == ["P(X)  [Bare]",
                                           "  child   X  discrete, 2 states: 0, 1"]


def test_long_lists_of_names_are_elided():
    childs = [Variable(f"V{i}", (0.0, 1.0)) for i in range(10)]
    assert statement(Bare(childs, [])) == "P(V0, V1, V2, ..., V9 (10))"


def test_describe_shortens_a_long_list_of_parents():
    X = Variable("X", [0, 1])
    parents = [Variable(f"P{i}", [0, 1]) for i in range(129)]
    lines = describe(Bare([X], parents)).splitlines()
    assert [line.split()[1] for line in lines[2:10]] == [f"P{i}" for i in range(8)]
    assert lines[10] == "  parent  ... 120 more (129 parents in all)"
    assert lines[11].split()[1] == "P128"
    assert len(lines) == 12


def test_a_failing_form_does_not_break_the_reading():
    class Broken(Shift):
        def form(self):
            raise RuntimeError("no form")

    _, probs = two_nodes()
    broken = Broken(probs["Y"].childs, probs["Y"].parents)
    assert repr(broken) == "Broken(P(Y | X))"


def test_the_package_exports_the_new_names():
    assert tbnpy.Probability is Probability
    assert tbnpy.check_model is check_model
    assert tbnpy.__version__ == "0.3.0"


# ---------------------------------------------------------------------------
# check_model: sound models
# ---------------------------------------------------------------------------
def test_compiled_and_hand_written_abcde_pass():
    from examples.ABCDE import s1_define_model as hand

    assert check_model(compile_file(ABCDE)[2]).ok
    assert check_model(hand.define_probs(hand.define_variables())).ok


def test_a_check_leaves_seeded_runs_unchanged():
    _, _, probs = compile_file(ABCDE)
    torch.manual_seed(5)
    before = inference.sample(probs, ["E"], n_sample=50)["E"].Cs
    torch.manual_seed(5)
    check_model(probs, n_sample=20)
    after = inference.sample(probs, ["E"], n_sample=50)["E"].Cs
    assert torch.equal(before, after)


def test_str_of_a_clean_check_says_so():
    _, probs = two_nodes()
    text = str(check_model(probs))
    assert text == ("Model check: 2 probability objects, 2 variables, "
                    "8 sampled rows: no problems found")


# ---------------------------------------------------------------------------
# check_model: what it catches
# ---------------------------------------------------------------------------
def issues_of(probs, severity="error"):
    check = check_model(probs, raise_on_error=False)
    return [(i.node, i.issue) for i in check.issues if i.severity == severity]


def test_errors_raise_by_default_and_carry_the_check():
    _, probs = two_nodes(sample=lambda pars: (torch.zeros(len(pars), 2), torch.zeros(len(pars))))
    with pytest.raises(ModelError, match="Cs has the wrong shape") as exc:
        check_model(probs)
    assert not exc.value.check.ok


def test_a_missing_method_is_reported():
    _, probs = two_nodes()
    probs["Z"] = object()
    assert ("Z", "not a probability object") in issues_of(probs)


def test_a_variable_owned_twice_is_reported():
    varis, probs = two_nodes()
    probs["Y2"] = Bare([varis["Y"]], [])
    assert ("Y2", "'Y' is a child of two objects") in issues_of(probs)


def test_a_parent_nobody_produces_is_reported():
    varis, probs = two_nodes()
    del probs["X"]
    assert ("Y", "parent 'X' is no object's child") in issues_of(probs)


def test_a_cycle_is_reported():
    P = Variable("P", (0.0, 1.0))
    Q = Variable("Q", (0.0, 1.0))
    probs = {"P": Bare([P], [Q]), "Q": Bare([Q], [P])}
    assert ("(model)", "the graph has a cycle") in issues_of(probs)


def test_a_parent_with_another_domain_is_a_warning():
    varis, probs = two_nodes()
    probs["Y"].parents = [Variable("X", ["a", "b", "c"])]
    assert ("Y", "parent 'X' has a different domain from its owner's") in issues_of(probs, "warning")


def test_wrong_logp_shape_is_reported():
    _, probs = two_nodes(sample=lambda pars: (torch.zeros(len(pars)), torch.zeros(len(pars), 1)))
    assert ("Y", "logp has the wrong shape") in issues_of(probs)


def test_a_discrete_value_must_be_a_state_index():
    X = Variable("X", [0, 1])
    probs = {"X": Bare([X], [])}
    probs["X"].sample = lambda n_sample=None, Cs_pars=None: (torch.full((n_sample,), 0.5),
                                                              torch.zeros(n_sample))
    assert ("X", "sampled 'X' is not a state index") in issues_of(probs)
    probs["X"].sample = lambda n_sample=None, Cs_pars=None: (torch.full((n_sample,), 7.0),
                                                              torch.zeros(n_sample))
    assert ("X", "sampled 'X' out of range") in issues_of(probs)


def test_a_continuous_value_must_lie_in_its_support():
    U = Variable("U", (0.0, 1.0))
    probs = {"U": Bare([U], [])}
    probs["U"].sample = lambda n_sample=None, Cs_pars=None: (torch.full((n_sample,), 2.0),
                                                              torch.zeros(n_sample))
    assert ("U", "sampled 'U' outside its support") in issues_of(probs)


def test_log_prob_must_agree_with_sample():
    _, probs = two_nodes(log_prob=lambda Cs: torch.zeros(Cs.shape[0]))
    assert ("Y", "log_prob disagrees with sample's logp") in issues_of(probs)


def test_an_unimplemented_log_prob_is_only_a_warning():
    def missing(Cs):
        raise NotImplementedError

    _, probs = two_nodes(log_prob=missing)
    check = check_model(probs)
    assert check.ok
    assert [i.issue for i in check.warnings] == ["log_prob is not implemented"]


def test_a_raising_object_blocks_its_descendants():
    varis, probs = two_nodes(sample=lambda pars: 1 / 0)
    Z = Variable("Z", (-torch.inf, torch.inf))
    probs["Z"] = Shift([Z], [varis["Y"]])
    check = check_model(probs, raise_on_error=False)
    assert [(i.node, i.issue) for i in check.errors] == [("Y", "sample() raised")]
    assert [(i.node, i.issue) for i in check.warnings] == [("Z", "not sampled")]
