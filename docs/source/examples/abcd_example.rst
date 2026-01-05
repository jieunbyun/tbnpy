
ABCDE example: mixed discrete–continuous Bayesian network
=========================================================

This example demonstrates how **TBN** performs inference on a Bayesian
network that combines **discrete and continuous variables**, using
tensorised Monte Carlo sampling and adaptive MCMC.

The example is intentionally minimal, but it exercises the full
workflow: model definition, evidence conditioning, scalable inference,
and posterior analysis.

Model structure
---------------

The Bayesian network consists of six variables:

- **A**: discrete, two states
- **B**: discrete, three states
- **C**: continuous, conditioned on *(A, B)*
- **OC**: continuous observation of *C*
- **D**: binary
- **E**: continuous, deterministic function of *(C, D)*

The dependency structure is:

- A, B → C
- C → OC
- C, D → E

Variables **A**, **B**, and **C** are treated as latent, while **OC**
is observed.

Step 1: defining variables and probability models
-------------------------------------------------

The file ``s1_define_model.py`` defines:

1. Variable domains (discrete vs. continuous)
2. Conditional probability objects for each node

In particular:

- **A** and **B** are defined using categorical CPTs
- **C | A, B** follows a Gaussian model
- **OC | C** is a noisy observation model
- **E | C, D** is deterministic

This separation between variables and probability objects allows
TBN to mix arbitrary discrete, continuous, and deterministic relations
within a single network.

The full model definition is shown below.

.. literalinclude:: ../../../examples/ABCDE/s1_define_model.py
   :language: python
   :linenos:

Step 2: evidence and scalable inference
---------------------------------------

The file ``s2_run_sample.py`` performs inference conditioned on evidence
for **OC**.

Key steps are:

1. Evidence definition  
   Multiple observations of **OC** are generated and stored in a
   tabular format, allowing batched conditioning.

2. Forward sampling initialisation  
   Initial MCMC chains are generated using forward sampling from the
   prior, improving stability and convergence.

3. Adaptive MCMC  
   An adaptive Metropolis–Hastings sampler is used to infer the
   posterior of *(A, B, C)* given the evidence.

TBN evaluates many Monte Carlo samples simultaneously by reformulating
inference computations as tensor operations. This enables efficient
scaling across chains, evidence rows, and iterations.

.. literalinclude:: ../../../examples/ABCDE/s2_run_sample.py
   :language: python

Prior vs posterior distributions
--------------------------------

The prior and posterior distributions for variables **A**, **B**, and
**C** illustrate how evidence propagates through both discrete and
continuous parts of the network.

- **A**: posterior probability shifts relative to the prior
- **B**: moderate posterior update through indirect influence
- **C**: posterior density becomes significantly narrower

This demonstrates how noisy observations constrain latent variables.

What this example illustrates
------------------------------

This example highlights several key features of TBN:

- Unified treatment of discrete and continuous variables
- Tensorised evaluation of large numbers of Monte Carlo samples
- Efficient adaptive MCMC for scalable Bayesian network inference
- Clear separation between model specification and inference

Although small, the ABCDE example mirrors the same inference pipeline
used for larger infrastructure and system-level models.

Next steps
----------

- Extend the example by conditioning on **E**
- Increase the number of evidence rows to stress-test scalability
- Enable GPU acceleration by setting ``USE_CUDA=1``
