Repository structure
====================

This repository is organised into the following main components.

Core library (tbnpy/)
---------------------
The ``tbnpy`` directory contains the core implementation of the
tensor-based Bayesian network (TBN) framework.
For scalable Bayesian network inference, TBN performs tensorised operations
that generate many Monte Carlo samples simultaneously.
It is designed to be compatible with matrix-based Bayesian networks
through the ``mbnpy`` module, especially for discrete variables.

- ``variable.py``: Variable definitions
- ``cpt.py``: Conditional probability tensors
- ``inference.py``: Inference routines
- ``adaptiveMH.py``: Adaptive Metropolis–Hastings sampler

Examples (examples/)
--------------------
Runnable, self-contained experiments demonstrating full workflows.

- ``ABCDE/``: Minimal mixed-variable Bayesian network
- ... (additional example directories)

Tests (tests/)
--------------
Unit tests verifying correctness of each module using pytest.
