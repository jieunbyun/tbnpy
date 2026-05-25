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

4. **Inference**  
   Perform probabilistic inference using Monte Carlo sampling–based methods.
