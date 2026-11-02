# Overview

TBNpy is a Python toolkit for **tensor-based Bayesian network (TBN) modelling**, 
designed to handle high-dimensional probabilistic models within the Bayesian network framework.

The library enables scalable probabilistic reasoning for complex infrastructure
and engineering systems, where conventional Bayesian network implementations
become computationally limiting.

## TBNPy: Python toolkit for tensor-based Bayesian network (TBN)

Bayesian network is a powerful framework for probabilistic reasoning under uncertainty.
However, traditional implementations often struggle with:

- Large numbers of variables and states
- Flexible or user-defined inference tasks

TBNpy addresses these challenges by:

- Accelerated Monte Carlo sample generation through tensor-based operations
- Support for customised variables and probability distributions

In this way, TBNpy aims to bring **model-based information**
(through Bayesian network structures and probability distributions)
and **numerical efficiency**
(through Monte Carlo sampling and tensor operations)
together within a unified framework.

## What TBNpy is for

TBNpy is particularly suited to:

- System-level risk assessment of large-scale infrastructure networks
- Dynamic risk analysis in engineering systems
- Research and prototyping of advanced Bayesian inference algorithms

## Installation

```bash
pip install tbnpy
python -c "import tbnpy; print(tbnpy.__version__)"
```

This documentation describes tbnpy 0.3.0; [the changelog](../changelog.md) lists what
changed in each version. To work on tbnpy itself, install it in editable mode from the
repository root with `pip install -e ".[dev]"`, and run the tests with
`python -m pytest tests`.

## A first look

```python
import tbnpy
from tbnpy.compile import compile_file

spec, varis, probs = compile_file("examples/ABCDE/abcde.tbn.json")

probs                            # every node, one line each
print(probs["C"])                # one node at length, with each variable's domain
print(tbnpy.check_model(probs))  # every object keeps tbnpy's protocol
```

## TBNpy workflows

A typical TBNpy workflow consists of the following steps:

1. **BN graph structure**  
   Define the Bayesian network structure using nodes and directed edges.

2. **Variable definition**  
   Specify each variable’s name and (optionally) its state space.

3. **Probability distribution definition**  
   Define probability distributions conditional on parent variables.
   - For discrete and tractable variables, use conditional probability tensors (CPTs).
   - For continuous or intractable variables, use custom probability distribution classes.
     - Custom distributions must accept tensor inputs, and all operations must be implemented using PyTorch tensor operations.
     - Inheriting `tbnpy.Probability` is optional; it makes each object print as the distribution it defines.
   - Alternatively, write the whole model as data (a JSON spec) and compile it with `tbnpy.compile`.

4. **Checking**  
   Type a variable's or probability object's name to see what it is, and call
   `tbnpy.check_model(probs)` to confirm that every object keeps tbnpy's protocol.

5. **Inference**  
   Perform probabilistic inference using Monte Carlo sampling–based methods.
