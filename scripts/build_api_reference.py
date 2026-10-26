# SPDX-License-Identifier: BSD-2-Clause
"""Generate searchable API signatures/docstrings without importing optional backends."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs/api.rst'


def literal(value):
    return '.. code-block:: text\n\n' + '\n'.join('    ' + line.rstrip() if line.strip() else '' for line in value.splitlines()) + '\n\n'


def reference():
    sections = ['API reference\n=============\n\n',
                'Generated from the checked-out source. Optional packages are not imported.\n'
                'All top-level public classes/functions and public methods are indexed.\n'
                'Signatures are complete; missing docstrings remain visible as documentation gaps.\n\n']
    for path in sorted((ROOT / 'llm4ad').rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith('_')]
        if not definitions:
            continue
        module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
        sections += [module + '\n' + '-' * len(module) + '\n\n']
        for node in definitions:
            title = node.name
            sections += [title + '\n' + '~' * len(title) + '\n\n']
            if isinstance(node, ast.ClassDef):
                sections += [literal('class ' + node.name + '(' + ', '.join(ast.unparse(b) for b in node.bases) + ')')]
                fields = [ast.unparse(n) for n in node.body if isinstance(n, ast.AnnAssign)]
                if fields:
                    sections += [literal('\n'.join(fields))]
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not n.name.startswith('_') or n.name == '__init__')]
            else:
                methods = [node]
            sections += [literal(ast.get_docstring(node) or 'Documentation gap: no docstring is present in this source definition.')]
            for method in methods:
                signature = f'{node.name + "." if isinstance(node, ast.ClassDef) else ""}{method.name}({ast.unparse(method.args)})'
                if method.returns:
                    signature += ' -> ' + ast.unparse(method.returns)
                sections += [literal(signature)]
                if method is not node:
                    sections += [literal(ast.get_docstring(method) or 'Documentation gap: no method docstring is present.')]
    return ''.join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    content = reference()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding='utf-8') != content:
            raise SystemExit('API reference is outdated; run scripts/build_api_reference.py')
    else:
        OUTPUT.parent.mkdir(exist_ok=True)
        OUTPUT.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
