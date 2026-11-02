# Changelog

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
- The ABCDE classes `C`, `OC` and `E` inherit `Probability`, and
  `s1_define_model.py` prints and checks the model it defines.
- The version is written once, in `tbnpy/__init__.py`. `pyproject.toml` and the docs
  read it from there.

### Fixed

- `spec.py`, `compile.py`, `expr.py` and `diagnostics.py`, with their tests, are now in
  the repository. 0.2.0 shipped them, but they had not been committed.
- `tests/test_cpt.py` compared `Cpt.sample`'s log probabilities with probabilities.
- `requirements.txt` had a stray comma before `ndtools-duco`.

### Compatibility

Nothing is removed or renamed. A probability object that does not inherit
`Probability` works as before, and inference, sampling and log probabilities give the
same numbers. Only the text of `repr(Variable)` and `repr(Cpt)` changes.

## 0.2.0

- Declarative model specs (`tbnpy.spec`, `tbnpy.compile`, `tbnpy.expr`) and model
  diagnostics (`tbnpy.diagnostics`).
- Batched forward sampling whose memory depends on the batch size, not the sample size.

## 0.1.0

- First release: `Variable`, `Cpt`, forward sampling and likelihood weighting
  (`tbnpy.inference`), and the adaptive Metropolis-Hastings sampler (`tbnpy.adaptiveMH`).
