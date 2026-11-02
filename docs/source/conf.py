# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add project path
import os
import re
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'tbnpy'
copyright = '2026, DUCO research group'
author = 'DUCO research group'

# The version is written once, in tbnpy/__init__.py. Read it as text, so that
# building the docs does not need tbnpy's dependencies installed.
with open(os.path.join(os.path.dirname(__file__), '../../tbnpy/__init__.py'), encoding='utf-8') as f:
    release = re.search(r'__version__\s*=\s*"([^"]+)"', f.read()).group(1)
version = release

# The docs build installs Sphinx only (docs/requirements.txt), so autodoc
# imports tbnpy against stand-ins for its heavy dependencies.
autodoc_mock_imports = ['torch', 'numpy', 'pandas', 'matplotlib', 'ndtools']
autodoc_member_order = 'bysource'

html_baseurl = "https://jieunbyun.github.io/tbnpy/"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",           # to use Markdown
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
