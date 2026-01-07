tbnpy.adaptiveMH
================

Overview
--------

This module implements an **adaptive, tensorised Metropolis–Hastings (MH) sampler**
for **hybrid Bayesian networks** containing both discrete and continuous variables.

Key design features:

- Supports **many evidence rows** and **many parallel chains** simultaneously.
- Uses **tensorised log-probability evaluation** for efficiency.
- Handles **hybrid variables**:
  - discrete variables via symmetric adaptive categorical proposals,
  - continuous variables via Gaussian random-walk proposals.
- Adapts proposal parameters during burn-in using **Robbins–Monro** updates.

The sampler is factor-based: each probability object (typically
:class:`~tbnpy.cpt.Cpt`) is treated as a *factor* contributing to the joint log-probability.


Glossary
--------

- **n_evi**: number of evidence rows.
- **n_chain**: number of parallel MCMC chains.
- **state**: dictionary mapping ``var_name -> tensor (n_evi, n_chain)``.
- **evidence_1d**: dictionary mapping ``var_name -> tensor (n_evi,)``.
- **factor**: a probability object defining a local conditional distribution.

Utility functions
-----------------

Variable type helpers
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: is_discrete(var) -> bool

   Return ``True`` if ``var`` represents a discrete variable
   (i.e. ``var.values`` is a list).

.. py:function:: is_continuous(var) -> bool

   Return ``True`` if ``var`` represents a continuous variable
   (i.e. ``var.values`` is a tuple).

.. py:function:: num_categories(var) -> int

   Return the number of discrete categories of ``var``.
   Raises an assertion error if ``var`` is not discrete.


Factor construction utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: build_Cs_3d(prob, state: dict, evidence_1d: dict) -> torch.Tensor

   Build a 3D tensor of composite states for a probability object.

   Parameters
   ----------
   prob
      Probability object with attributes ``childs`` and ``parents``.
   state
      Dictionary mapping latent variable names to tensors of shape ``(n_evi, n_chain)``.
   evidence_1d
      Dictionary mapping observed variable names to tensors of shape ``(n_evi,)``.

   Returns
   -------
   torch.Tensor
      Tensor of shape ``(n_evi, n_chain, n_childs + n_parents)`` in column order::

        [child_0, ..., child_{n_childs-1}, parent_0, ..., parent_{n_parents-1}]

   Notes
   -----
   Evidence variables are broadcast across chains.
   Latent variables are taken from ``state``.


.. py:function:: factor_logp_2d(prob, Cs_3d: torch.Tensor) -> torch.Tensor

   Evaluate a factor log-probability on a 3D tensor.

   Parameters
   ----------
   prob
      Probability object exposing ``log_prob``.
   Cs_3d
      Tensor of shape ``(n_evi, n_chain, dim)``.

   Returns
   -------
   torch.Tensor
      Log-probabilities of shape ``(n_evi, n_chain)``.


.. py:function:: build_factors_by_var(probs: dict) -> dict

   Build an adjacency mapping from variable name to dependent factors.

   Parameters
   ----------
   probs
      Mapping ``{node_name: probability_object}``.

   Returns
   -------
   dict
      Mapping ``{var_name: list[probability_object]}`` where each factor
      depends on the variable either as a child or a parent.


Proposal kernels
----------------

Discrete proposals
~~~~~~~~~~~~~~~~~~

.. py:function:: propose_discrete_adaptive(x: torch.Tensor, logits: torch.Tensor, alpha: float) -> torch.Tensor

   Symmetric adaptive proposal for a discrete variable.

   Parameters
   ----------
   x
      Current state, shape ``(n_evi, n_chain)``, integer in ``[0, K-1]``.
   logits
      Global adaptive logits of shape ``(K,)``.
   alpha
      Mixing weight between local and global proposals (``0 < alpha < 1``).

   Returns
   -------
   torch.Tensor
      Proposed state, same shape as ``x``.

   Notes
   -----
   The proposal is a mixture of:

   - **local move**: uniform jump to any other category,
   - **global move**: draw from a learned categorical distribution.

   The mixture remains symmetric, preserving MH correctness.


Continuous proposals
~~~~~~~~~~~~~~~~~~~~

.. py:function:: propose_continuous_rw_gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor

   Gaussian random-walk proposal.

   Parameters
   ----------
   x
      Current state tensor of shape ``(n_evi, n_chain)``.
   sigma
      Standard deviation of the proposal.

   Returns
   -------
   torch.Tensor
      Proposed state, same shape as ``x``.


Adaptation configuration
------------------------

.. py:class:: AdaptConfig

   Configuration parameters for adaptive MH.

   Attributes
   ----------
   burnin : int
      Number of iterations during which adaptation is active.
   gamma : float
      Robbins–Monro exponent (typically in ``(0.5, 1]``).
   target_accept : float
      Target acceptance rate for continuous proposals.
   min_log_sigma, max_log_sigma : float
      Bounds for continuous proposal scales.
   alpha : float
      Mixing weight for discrete proposals.


HybridAdaptiveMH
----------------

.. py:class:: HybridAdaptiveMH

   Adaptive Metropolis–Hastings sampler for hybrid Bayesian networks.

   The sampler maintains a latent state:

   - ``state[var_name]`` has shape ``(n_evi, n_chain)``
   - evidence variables are fixed and broadcast across chains

   Each MH step updates **one variable (or block)** at a time.

Initialisation
~~~~~~~~~~~~~~

.. py:method:: HybridAdaptiveMH.init_state_from_forward_samples(probs_copy: dict)

   Initialise latent state from forward-sampling output
   (e.g. produced by :mod:`tbnpy.inference`).

   Parameters
   ----------
   probs_copy
      Dictionary of probability objects where each object has:

      - ``prob.Cs`` of shape ``(n_evi, n_chain, dim)``
      - ``prob.ps`` of shape ``(n_evi, n_chain)``


.. py:method:: HybridAdaptiveMH.init_state_random(seed: int | None = None)

   Initialise latent state randomly.

   Discrete variables are drawn uniformly; continuous variables
   are drawn from a standard normal distribution.


Core MCMC update
~~~~~~~~~~~~~~~~

.. py:method:: HybridAdaptiveMH.mh_update_block(vars: list, iteration: int) -> torch.Tensor

   Perform one MH update for a block of variables.

   Parameters
   ----------
   vars
      List of variables to update jointly.
   iteration
      Current iteration index (used for adaptation).

   Returns
   -------
   torch.Tensor
      Boolean tensor of shape ``(n_chain,)`` indicating accepted chains.


Running the sampler
~~~~~~~~~~~~~~~~~~~

.. py:method:: HybridAdaptiveMH.run(n_iter: int, update_blocks: list[list[str]] | None = None, store_every: int = 0, progress_every: int = 100) -> dict

   Run the adaptive MH sampler.

   Parameters
   ----------
   n_iter
      Number of MCMC iterations.
   update_blocks
      Optional variable blocks (by name) to update jointly.
   store_every
      Thinning interval for storing full latent states (can be memory intensive).
   progress_every
      Print progress every given number of iterations.

   Returns
   -------
   dict
      Dictionary containing:

      - ``accept_rate`` : per-block acceptance rates
      - ``logp_chain`` : final log-probability per chain
      - ``logp_evi_chain`` : final log-probability per evidence×chain
      - ``log_sigma`` : learned proposal scales for continuous variables
      - ``states_thinned`` (optional) : stored latent states
