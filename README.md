# tbnpy

**tbnpy** is a Python toolkit for **tensor-based Bayesian networks (TBNs)**, designed for scalable probabilistic inference in systems with large, structured state spaces.  
It is particularly suited to applications where classical Bayesian network implementations struggle due to combinatorial growth in system states.

The package provides:
- A tensor-based formulation of Bayesian networks
- Scalable exact and approximate inference algorithms
- Support for hybrid discrete–continuous models
- Reusable system-level rules and probabilistic components
- Examples illustrating end-to-end modelling and inference workflows

## Documentation

📘 **Full documentation is available here:**  
👉 https://jieunbyun.github.io/tbnpy/

The online documentation includes:
- Conceptual overview of tensor-based Bayesian networks
- Installation and getting-started guides
- API reference for all core modules
- Worked examples (including the ABCDE example)
- Repository structure and design rationale

## Repository structure

```
tbnpy/
├── tbnpy/              # Core library
├── examples/           # Worked examples and case studies
├── docs/               # Sphinx documentation source
├── .github/            # CI/CD workflows (GitHub Actions)
└── README.md
```


## Installation

The `ndtools` utilities can be installed either as a **released package** from PyPI
or in **editable (developer) mode** from the source repository.

### Using pip (recommended for users)

Install the latest released version from PyPI:

```bash
pip install tbnpy
```

This installs a stable version of the tools suitable for general use.

Verify the installation:

```bash
python -c "import tbnpy; print(tbnpy.__version__)"
```

---

### Using pip (editable / developer install)

If you are developing `tbnpy` or working directly with the source code,
install in editable mode from the repository root:

```bash
# Activate your environment first (if using conda)
conda activate <your-env>

# Install in editable mode
pip install -e .
```

## Development status

tbnpy is under active development.  
The API may evolve as new modelling patterns, inference strategies, and large-scale case studies are incorporated.

## Citation

## Citation

A dedicated tbnpy methodology paper is currently under preparation.  
In the meantime, please cite the following related work:

Byun, J.-E., & Song, J. (2021). *Generalized matrix-based Bayesian network for multi-state systems*. Reliability Engineering & System Safety, 211, 107468.


## License

This project is released under an MIT license.  
See the repository for license details.

---

For detailed usage, examples, and API documentation, please refer to:  
🔗 https://jieunbyun.github.io/tbnpy/
