import math

import pytest
import torch

from tbnpy.expr import (CATEGORICAL, CONTINUOUS, NUMERIC_STATE, Expression,
                        ExpressionError, NameSpec)


def cont():
    return NameSpec(CONTINUOUS)


def t(*vals):
    return torch.tensor(list(vals), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Operator handling
# ---------------------------------------------------------------------------
def test_caret_binds_tighter_than_multiplication():
    """GeNIe's ``^`` is exponentiation, and must not inherit Python's XOR
    precedence -- otherwise ``a * H^2`` silently becomes ``(a * H) ** 2``."""
    e = Expression("0.325 * K_a * Gamma * H^2",
                   {"K_a": cont(), "Gamma": cont(), "H": cont()})
    got = e({"K_a": t(0.3), "Gamma": t(20.0), "H": t(10.0)}).item()
    assert got == pytest.approx(0.325 * 0.3 * 20.0 * 100.0)


def test_caret_and_double_star_agree():
    scope = {"x": cont()}
    a = Expression("2 * x^3", scope)({"x": t(2.0)})
    b = Expression("2 * x**3", scope)({"x": t(2.0)})
    assert torch.allclose(a, b) and a.item() == pytest.approx(16.0)


def test_unary_minus_and_precedence():
    e = Expression("-x^2", {"x": cont()})
    assert e({"x": t(3.0)}).item() == pytest.approx(-9.0)


def test_arithmetic_is_vectorised_over_the_batch():
    e = Expression("a * b + 1", {"a": cont(), "b": cont()})
    out = e({"a": t(1.0, 2.0, 3.0), "b": t(10.0, 10.0, 10.0)})
    assert out.tolist() == [11.0, 21.0, 31.0]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------
def test_max_clamps_like_the_seepage_model():
    e = Expression("Max(h_0 - h_i, 0)", {"h_0": cont(), "h_i": cont()})
    assert e({"h_0": t(5.0, 1.0), "h_i": t(2.0, 4.0)}).tolist() == [3.0, 0.0]


def test_trig_and_pi_reproduce_rankine_coefficient():
    e = Expression("Tan((45 - phi/2) * Pi()/180)^2", {"phi": cont()})
    got = e({"phi": torch.tensor([30.0], dtype=torch.float64)}).item()
    assert got == pytest.approx(math.tan(30 * math.pi / 180) ** 2)


def test_dtype_and_device_follow_the_inputs():
    e = Expression("x * 2", {"x": cont()})
    assert e({"x": torch.tensor([1.0], dtype=torch.float64)}).dtype is torch.float64
    assert e({"x": torch.tensor([1.0], dtype=torch.float32)}).dtype is torch.float32


def test_if_selects_elementwise():
    e = Expression("If(x > 0, x, 0)", {"x": cont()})
    assert e({"x": t(2.0, -2.0)}).tolist() == [2.0, 0.0]


# ---------------------------------------------------------------------------
# Discrete variables
# ---------------------------------------------------------------------------
def test_numeric_discrete_resolves_to_its_label_value():
    e = Expression("A + B", {"A": NameSpec(NUMERIC_STATE, 2),
                             "B": NameSpec(NUMERIC_STATE, 3)})
    assert e({"A": t(-0.3), "B": t(0.2)}).item() == pytest.approx(-0.1)


def test_choose_selects_on_the_state_index():
    scope = {"S": NameSpec(CATEGORICAL, 2), "a": cont(), "b": cont()}
    e = Expression("Choose(S, a, b)", scope)
    out = e({"a": t(10.0, 10.0), "b": t(99.0, 99.0)},
            {"S": torch.tensor([0, 1])})
    assert out.tolist() == [10.0, 99.0]
    assert e.index_names == frozenset({"S"})


def test_choose_options_may_be_expressions():
    scope = {"S": NameSpec(CATEGORICAL, 2), "I_w": cont()}
    e = Expression("Choose(S, 0.2*I_w, 0.3*I_w)", scope)
    out = e({"I_w": t(10.0, 10.0)}, {"S": torch.tensor([0, 1])})
    assert out.tolist() == pytest.approx([2.0, 3.0])


def test_choose_requires_one_option_per_state():
    scope = {"S": NameSpec(CATEGORICAL, 3), "a": cont(), "b": cont()}
    with pytest.raises(ExpressionError, match="3 states"):
        Expression("Choose(S, a, b)", scope)


def test_choose_rejects_a_continuous_selector():
    with pytest.raises(ExpressionError, match="continuous"):
        Expression("Choose(x, a, b)", {"x": cont(), "a": cont(), "b": cont()})


def test_categorical_variable_cannot_be_used_arithmetically():
    with pytest.raises(ExpressionError, match="non-numeric states"):
        Expression("S * 2", {"S": NameSpec(CATEGORICAL, 2)})


# ---------------------------------------------------------------------------
# Safety -- an expression is data from a model file, not trusted code
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source", [
    "__import__('os').system('echo hi')",
    "open('secrets.txt')",
    "x.__class__",
    "[i for i in range(3)]",
    "(lambda: 1)()",
    "{'a': 1}",
])
def test_dangerous_constructs_are_rejected(source):
    with pytest.raises(ExpressionError):
        Expression(source, {"x": cont()})


def test_unknown_name_names_what_is_in_scope():
    with pytest.raises(ExpressionError, match="Gamma"):
        Expression("Gamma * H", {"Gamma": cont()})


def test_unknown_function_is_rejected():
    with pytest.raises(ExpressionError, match="unknown function"):
        Expression("Frobnicate(1)", {})


def test_wrong_arity_is_rejected():
    with pytest.raises(ExpressionError, match="1 argument"):
        Expression("Exp(1, 2)", {})


def test_syntax_error_mentions_the_node():
    with pytest.raises(ExpressionError, match="node 'q'"):
        Expression("1 +", {}, label="node 'q'")


def test_reported_names_cover_only_what_is_referenced():
    e = Expression("a + 1", {"a": cont(), "b": cont()})
    assert e.names == frozenset({"a"})
