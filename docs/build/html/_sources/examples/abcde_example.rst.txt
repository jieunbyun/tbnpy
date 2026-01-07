
ABCDE example
==============

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

The BN graph is:

.. image:: ../_static/ABCDE_bngraph.png
   :alt: ABCDE example's BN graph
   :align: center
   :width: 300px

**OC** is shaded as an observed variable.

Task
----

Given multiple noisy observations of **OC**, the task is to infer the
posterior distributions of **A**, **B**, and **C**,
i.e. :math:`p(A, B, C \mid OC=\text{evidence})`.

Step 0: Define custom variables and probability distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**A**, **B**, and **D** are discrete, tractable (i.e. small state-space) variables,
so they use built-in CPT classes.

**C**, **OC**, and **E** are continuous (or non-tabular) variables, requiring custom
probability models:

- **C | A, B**: Gaussian distribution with mean and variance depending on the
  discrete parents *(A, B)*:

  .. literalinclude:: ../../../examples/ABCDE/c.py
     :language: python
     :linenos:

- **OC | C**: Noisy observation model with Gaussian noise:

  .. literalinclude:: ../../../examples/ABCDE/oc.py
     :language: python
     :linenos:

- **E | C, D**: Deterministic relation defined by a function:

  .. literalinclude:: ../../../examples/ABCDE/e.py
     :language: python
     :linenos:

Important notes
^^^^^^^^^^^^^^^

A custom probability model class should include (at minimum) the following methods:

1. ``__init__`` defines core properties such as ``childs``, ``parents``, and ``device``
   (``cpu`` or ``gpu``). Additional properties may be added as needed.

2. ``sample`` generates samples of child variables conditioned on given samples of
   the parent nodes.

   - Inputs: ``self`` and ``Cs_pars``, where ``Cs_pars`` is a tensor of shape
     ``(num_samples, num_parents)`` storing realisations of the parent variables.
   - Returns:
     - **(recommended)** a tensor of child samples of shape ``(num_samples, num_childs)``
     - a tensor of log-probabilities of shape ``(num_samples,)`` for the generated samples

3. ``log_prob`` computes the log-probability of given samples.

   - Inputs: ``self`` and ``Cs``, where ``Cs`` is a tensor of shape
     ``(num_samples, num_childs + num_parents)`` storing realisations of both
     child (first) and parent (second) variables.
   - Returns: a tensor of shape ``(num_samples,)`` storing log-probabilities.

All methods should be compatible with both CPU and GPU tensors and should use
PyTorch tensor operations for efficiency.


Step 1: defining variables and probability models
-------------------------------------------------

The file ``s1_define_model.py`` defines:

1. Variables (discrete vs. continuous)
2. Conditional probability objects for each node

In particular:

- **P(A)** and **P(B)** are defined using categorical CPTs
- **P(C | A, B)** follows a Gaussian model
- **P(OC | C)** is a noisy observation model
- **P(E | C, D)** is deterministic

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
   posterior of *(A, B, C)* given the evidence, i.e. **P(A, B, C | OC=oc)**.

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
- **C**: posterior density becomes narrower

This demonstrates how noisy observations constrain latent variables.