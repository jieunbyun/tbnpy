"""
The ABCDE model again, written as data rather than as classes.

abcde.tbn.json describes the same six variables and six nodes as
s1_define_model.py: C and OC are `distribution` nodes, E is an `equation` node,
and A, B and D are `discrete_cpt` nodes. tbnpy.compile turns it into the same
(varis, probs) pair, so everything downstream (inference, adaptiveMH,
diagnostics) runs on it unchanged.
"""

import os, sys
BASE = os.path.dirname(os.path.abspath(__file__))

repo_root = os.path.abspath(os.path.join(BASE, "../.."))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import torch

import tbnpy
from tbnpy import inference
from tbnpy.compile import compile_file

SPEC = os.path.join(BASE, "abcde.tbn.json")


def define_model(device="cpu"):
    """Return (spec, varis, probs) compiled from abcde.tbn.json."""
    return compile_file(SPEC, device=device)


if __name__ == "__main__":
    spec, varis, probs = define_model()

    # Each compiled node prints as the distribution it defines, with its form
    # taken from the spec: a family and its parameters, an equation, or a table.
    print(probs)
    print(probs["E"])

    # The compiled model keeps tbnpy's protocol, like the hand-written one.
    print(tbnpy.check_model(probs))

    # And it gives the same numbers: compare with s1_define_model.py.
    from s1_define_model import define_probs, define_variables
    hand = define_probs(define_variables())
    rows = torch.tensor([[0.5, 1.0, 2.0], [-1.0, 0.0, 0.0]])    # [C | A, B]
    print("log p(C | A, B), compiled:    ", probs["C"].log_prob(rows))
    print("log p(C | A, B), hand-written:", hand["C"].log_prob(rows))

    out = inference.sample(probs, ["E"], n_sample=10_000)
    print("P(E == 0) ~", (out["E"].Cs == 0).float().mean().item(), "(D = 1 with probability 0.6)")
