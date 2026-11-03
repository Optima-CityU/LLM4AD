# SPDX-License-Identifier: BSD-2-Clause
"""Versioned, import-free documentation build configuration."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
project = 'LLM4AD'
author = 'LLM4AD Developers'
copyright = '2026, Optima Group'
release = runpy.run_path(str(ROOT / 'llm4ad/_version.py'))['__version__']
version = release
extensions = ['myst_parser']
html_theme = 'furo'
exclude_patterns = ['_build', 'api', 'README.md']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
