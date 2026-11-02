tbnpy.diagnostics
=================

Checks of a model's *numbers*, where :func:`tbnpy.check_model` checks its *protocol*:

- :func:`~tbnpy.diagnostics.run_sense_check` forward-samples the prior and reports
  marginals, bound violations, rank correlations, and how each discrete variable moves
  the others, which catches unit errors, sign flips, dead variables and NaN paths;
- :func:`~tbnpy.diagnostics.check_evidence` refuses, with an explanation, conditioning
  that cannot work, such as an exact observation of a deterministic continuous node;
- :func:`~tbnpy.diagnostics.posterior` wraps likelihood weighting
  (:func:`tbnpy.inference.sample_evidence`) and returns summaries together with the
  effective sample size that qualifies them.

.. code-block:: python

   from tbnpy import diagnostics

   report = diagnostics.run_sense_check(probs, varis, n_sample=20_000)
   print(report.flags)

   r = diagnostics.posterior(probs, ["C"], {"OC": -0.3}, n_sample=50_000, varis=varis)
   print(r.summary(varis), r.ess)

API
---

.. automodule:: tbnpy.diagnostics
   :members: run_sense_check, SenseCheckReport, forward_sample_all, plot_report, check_evidence, EvidenceIssue, posterior, PosteriorResult, effective_sample_size, weighted_quantile
