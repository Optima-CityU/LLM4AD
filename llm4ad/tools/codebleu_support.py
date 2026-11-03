# SPDX-License-Identifier: BSD-2-Clause
"""Check that CodeBLEU can parse Python with the installed tree-sitter.

The PyPI release codebleu 0.7.0 requires tree-sitter<0.23, which cannot load
tree-sitter-python 0.23 grammars. The compatible 0.7.1 build is only available
from GitHub, and PyPI packages cannot declare direct-URL dependencies, so it is
installed separately.
"""
from __future__ import annotations

import functools

CODEBLEU_URL = 'https://github.com/k4black/codebleu/archive/b0edb622f6a52fe9d1edc407be5061d3e1462a7f.zip'
CODEBLEU_INSTALL = (f'python -m pip install "codebleu @ {CODEBLEU_URL}" '
                    '"tree-sitter==0.23.2" "tree-sitter-python==0.23.0"')

_PROBE = 'def f(a):\n    return a + 1\n'


@functools.lru_cache(maxsize=None)
def require_codebleu(method: str) -> None:
    """Raise ImportError with install instructions unless CodeBLEU syntax matching works."""
    try:
        from codebleu.syntax_match import calc_syntax_match
        score = calc_syntax_match([_PROBE], _PROBE, 'python')
    except Exception as error:
        raise ImportError(
            f'{method} needs CodeBLEU 0.7.1 with tree-sitter 0.23 for code similarity '
            f'({type(error).__name__}: {error}). Install it with:\n    {CODEBLEU_INSTALL}',
            name='codebleu') from error
    if score != 1.0:
        raise ImportError(
            f'{method} found an incompatible CodeBLEU installation (identical code scored {score}). '
            f'Install it with:\n    {CODEBLEU_INSTALL}', name='codebleu')
