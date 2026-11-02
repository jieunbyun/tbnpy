Repository structure
====================

This repository is organised into the following main components.

Core library (tbnpy/)
---------------------
The ``tbnpy`` directory contains the core implementation of the
tensor-based Bayesian network (TBN) framework.
It is designed to be compatible with matrix-based Bayesian networks
through the ``mbnpy`` module, especially for discrete variables.

Defining a model:

- ``variable.py``: Variable definitions
- ``probability.py``: The probability-object protocol, the optional ``Probability`` base class, and ``check_model``
- ``cpt.py``: Conditional probability tensors
- ``spec.py``: A model written as data (JSON), validated
- ``compile.py``: Compiles a spec into variables and probability objects
- ``expr.py``: The expression language of specs

Running and checking it:

- ``inference.py``: Inference routines
- ``adaptiveMH.py``: Adaptive Metropolis–Hastings sampler
- ``diagnostics.py``: Prior sense checks, evidence validation and weighted posteriors

For scalable Bayesian network inference, TBN performs tensorised operations
that generate many Monte Carlo samples simultaneously.

Examples (examples/)
--------------------
Runnable, self-contained experiments demonstrating full workflows.

- ``ABCDE/``: Minimal mixed-variable Bayesian network, as classes and as a spec (``abcde.tbn.json``)
- ``EMA_aftershock/``: Mainshock-aftershock analysis of the Eastern Massachusetts highway network
- ... (additional example directories)

Tests (tests/)
--------------
``python -m pytest tests`` runs the test suite.
