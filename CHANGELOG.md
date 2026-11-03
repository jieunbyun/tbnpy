# Changelog

## 0.3.4

### Added

- A continuous variable's `support` may be open on one side: `[0, null]` is [0, ∞),
  `[null, 0]` is (−∞, 0]. JSON cannot write infinity, so a positive quantity such as a
  ground-motion intensity could not be declared in a spec before.

## 0.3.3

### Added

- Custom spec nodes can use prestored data. A spec is JSON, so data too large or too
  binary to write into a node's `kwargs` (rule sets, tables, grids, model weights) is
  named there by file, relative to the spec's folder. `compile_file` passes that folder
  to every custom class whose constructor has a `directory` parameter; `compile_model`
  takes it as `directory=...` for a spec built in memory. Classes without the parameter
  are built as before.

## 0.3.2

### Removed

- The spec's optional `analysis` section (`AnalysisSpec`: intended query and observed
  nodes, notes, options). A Bayesian network answers any query within its scope, so a
  model description does not name its queries in advance. A file that still has the
  section loads as before; the section is read past and not written back.
  `examples/ABCDE/abcde.tbn.json` no longer has one.

## 0.3.1

### Added

- A `custom` spec node may own several child variables: list them in `childs` instead of
  naming one `child`. A field drawn jointly over many sites, such as a ground-motion
  field, is then one node of a spec, as it is one probability object. Specs written by
  this version say `"spec_version": 2`; version 1 files load unchanged.

### Fixed

- A seeded run now gives the same samples in every process. `inference.get_ancestor_order`
  ordered nodes by iterating a Python `set`, and string hashing differs between
  processes, so the order in which nodes drew their random numbers, and hence the
  samples, changed from run to run. Among nodes ready at the same time, the order now
  follows `probs`. Seeded results therefore differ from 0.3.0's, once, and are then
  reproducible.
- `inference.sample` and `inference.sample_evidence` return their query nodes in sampling
  order, not a set's.

## 0.3.0

### Added

- `tbnpy.probability`, which states the probability-object protocol in one place and adds:
  - `Probability`, an optional base class. An object that inherits it prints as the
    distribution it defines: `C(P(C | A, B): normal(loc=A + B, scale=0.6))`. `print`
    gives each variable's domain too. Subclasses say what their distribution is in
    `form()`, and can add lines in `form_details()`.
  - `describe(obj)`, the same reading for any probability object, inherited or not.
  - `check_model(probs)`, which samples a few rows through a whole model and reports
    every object that breaks the protocol: wrong shapes, values outside a variable's
    domain, a non-finite `logp`, or a `log_prob` that disagrees with `sample`. It
    restores the random number generator's state, so it never changes a seeded run.
  - `tbnpy.Probability`, `tbnpy.describe` and `tbnpy.check_model` at the top level.
- `examples/ABCDE/abcde.tbn.json`, the ABCDE model as a spec, and
  `s1_define_model_from_spec.py`, which compiles it and compares it with the
  hand-written model.
- API pages for `tbnpy.probability`, `tbnpy.spec`, `tbnpy.compile`, `tbnpy.expr` and
  `tbnpy.diagnostics`, and this changelog.
- A release workflow that tests, builds and publishes from a version tag.

### Changed

- `Cpt` and every node compiled by `tbnpy.compile` inherit `Probability`. `repr(cpt)` is
  now one line; `print(cpt)` shows its table with state labels.
- `repr(Variable)` is one line, `Variable(name='A', values=[-0.3, 0.3])`. Before, it
  spanned two lines and left its parenthesis open.
- The probability classes of both examples, ABCDE and EMA aftershock, inherit
  `Probability`, and each example's `s1_define_model.py` / `s01_define_model.py` prints
  and checks the model it defines.
- `describe` shortens a list of more than ten childs or parents.
- The version is written once, in `tbnpy/__init__.py`. `pyproject.toml` and the docs
  read it from there.

### Fixed

- `spec.py`, `compile.py`, `expr.py` and `diagnostics.py`, with their tests, are now in
  the repository. 0.2.0 shipped them, but they had not been committed.
- `HybridAdaptiveMH` failed with an `AttributeError` when `variables` was a dictionary,
  although its signature allows one.
- EMA aftershock example, found by `check_model`: the aftershock damage `Dt` returned a
  log density of `+inf` for most active aftershock rows, because `erfc` underflowed in
  single precision when the truncation bound lay far in the tail; it now uses
  `torch.special.log_ndtr`. A sampled `D_t` could also round to just below
  `D_{t-1}`, which its own `log_prob` rejects.
- `tests/test_cpt.py` compared `Cpt.sample`'s log probabilities with probabilities.
- `requirements.txt` had a stray comma before `ndtools-duco`.
- Docs: `inference.sample`'s `on_batch` and `accumulate_query`, and
  `HybridAdaptiveMH`'s constructor, are documented; `Cpt.ps` is always a log probability.

### Compatibility

Nothing is removed or renamed. A probability object that does not inherit
`Probability` works as before, and inference, sampling and log probabilities give the
same numbers. Only the text of `repr(Variable)` and `repr(Cpt)` changes. In the EMA
example, `Dt`'s log densities change where they were infinite, and its samples change
only where they had rounded below `D_{t-1}`.

## 0.2.0

- Declarative model specs (`tbnpy.spec`, `tbnpy.compile`, `tbnpy.expr`) and model
  diagnostics (`tbnpy.diagnostics`).
- Batched forward sampling whose memory depends on the batch size, not the sample size.

## 0.1.0

- First release: `Variable`, `Cpt`, forward sampling and likelihood weighting
  (`tbnpy.inference`), and the adaptive Metropolis-Hastings sampler (`tbnpy.adaptiveMH`).
