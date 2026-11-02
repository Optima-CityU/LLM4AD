# SPDX-License-Identifier: BSD-2-Clause
"""CodeBLEU dependency setup for MEoH/PartEvo code-similarity ranking."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from llm4ad.base import TextFunctionProgramConverter as Converter
from llm4ad.tools.codebleu_support import CODEBLEU_INSTALL, CODEBLEU_URL, require_codebleu

ROOT = Path(__file__).resolve().parents[1]


def extras_require():
    tree = ast.parse((ROOT / 'setup.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and n.targets[0].id == 'EXTRAS_REQUIRE')
    return ast.literal_eval(node.value)


def fake_codebleu(calc_syntax_match):
    package, module = ModuleType('codebleu'), ModuleType('codebleu.syntax_match')
    module.calc_syntax_match = calc_syntax_match
    package.syntax_match = module
    return {'codebleu': package, 'codebleu.syntax_match': module}


def codebleu_works():
    try:
        require_codebleu('test')
        return importlib.util.find_spec('pymoo') is not None
    except ImportError:
        return False
    finally:
        require_codebleu.cache_clear()


class DependencySetupTests(unittest.TestCase):
    def test_extras_are_pypi_safe_and_pin_compatible_tree_sitter(self):
        extras = extras_require()
        for name, packages in extras.items():
            for package in packages:
                with self.subTest(extra=name, package=package):
                    self.assertNotIn('://', package, 'PyPI rejects direct-URL dependencies')
        for name in ['meoh', 'partevo']:
            with self.subTest(extra=name):
                # PyPI's codebleu 0.7.0 requires tree-sitter<0.23 and cannot parse these grammars.
                self.assertNotIn('codebleu', [p.split('=')[0].split('<')[0].split('>')[0] for p in extras[name]])
                self.assertIn('tree-sitter==0.23.2', extras[name])
                self.assertIn('tree-sitter-python==0.23.0', extras[name])

    def test_install_command_pins_the_tested_versions(self):
        self.assertTrue(CODEBLEU_URL.endswith('.zip'))
        self.assertIn(f'"codebleu @ {CODEBLEU_URL}"', CODEBLEU_INSTALL)
        for pin in ['tree-sitter==0.23.2', 'tree-sitter-python==0.23.0']:
            self.assertIn(pin, CODEBLEU_INSTALL)

    def test_readme_shows_the_same_install_command(self):
        self.assertIn(CODEBLEU_INSTALL, (ROOT / 'README.md').read_text(encoding='utf-8'))

    def test_ci_installs_the_same_codebleu_build(self):
        workflow = (ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8')
        self.assertIn('from llm4ad.tools.codebleu_support import CODEBLEU_URL', workflow)


class RequireCodebleuTests(unittest.TestCase):
    def setUp(self):
        require_codebleu.cache_clear()
        self.addCleanup(require_codebleu.cache_clear)

    def test_missing_codebleu_explains_installation(self):
        with patch.dict(sys.modules, {'codebleu': None, 'codebleu.syntax_match': None}):
            with self.assertRaises(ImportError) as context:
                require_codebleu('MEoH')
        self.assertEqual(context.exception.name, 'codebleu')
        self.assertIn(CODEBLEU_INSTALL, str(context.exception))

    def test_tree_sitter_mismatch_fails_fast(self):
        def incompatible(*args):
            raise TypeError('an integer is required')  # codebleu 0.7.0 with tree-sitter-python 0.23
        with patch.dict(sys.modules, fake_codebleu(incompatible)):
            with self.assertRaisesRegex(ImportError, 'MEoH needs CodeBLEU 0.7.1') as context:
                require_codebleu('MEoH')
        self.assertIn('TypeError: an integer is required', str(context.exception))
        self.assertIn(CODEBLEU_INSTALL, str(context.exception))

    def test_wrong_similarity_is_rejected(self):
        with patch.dict(sys.modules, fake_codebleu(lambda *args: 0.0)):
            with self.assertRaisesRegex(ImportError, 'identical code scored 0.0'):
                require_codebleu('PartEvo')

    def test_compatible_codebleu_is_checked_once(self):
        calls = []
        with patch.dict(sys.modules, fake_codebleu(lambda *args: calls.append(args) or 1.0)):
            require_codebleu('MEoH')
            require_codebleu('MEoH')
        self.assertEqual(len(calls), 1)

    def test_meoh_and_partevo_check_before_searching(self):
        for module_name, method in [('llm4ad.method.meoh.meoh', 'MEoH'), ('llm4ad.method.partevo.partevo', 'PartEvo')]:
            with self.subTest(method=method):
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError as error:
                    self.skipTest(f'Optional dependency {error.name} is unavailable')
                failure = ImportError('incompatible CodeBLEU', name='codebleu')
                with patch.object(module, 'require_codebleu', side_effect=failure) as check:
                    with self.assertRaises(ImportError):
                        task = SimpleNamespace(template_program='def f():\n    return 0\n', task_description='')
                        getattr(module, method)(llm=SimpleNamespace(), evaluation=task)
                check.assert_called_once_with(method)


@unittest.skipUnless(codebleu_works(), 'Compatible CodeBLEU/tree-sitter and pymoo are not installed')
class InstalledCodebleuTests(unittest.TestCase):
    """Runs in the optional CI job, which installs the GitHub CodeBLEU build."""

    def test_syntax_match_distinguishes_program_structure(self):
        from codebleu.syntax_match import calc_syntax_match
        simple = 'def f(a):\n    return a\n'
        loop = 'def f(a):\n    for i in range(a):\n        a += i\n    return a\n'
        self.assertEqual(calc_syntax_match([loop], loop, 'python'), 1.0)
        self.assertLess(calc_syntax_match([loop], simple, 'python'), 1.0)

    def test_meoh_population_advances_a_generation(self):
        # An incompatible CodeBLEU used to fail inside survival, which swallowed the error
        # and left MEoH on its initial population.
        from llm4ad.method.meoh.population import Population
        population = Population(pop_size=2)
        bodies = ['return a', 'return a + 1', 'return a * 2', 'for i in range(3):\n        a += i\n    return a']
        for index, body in enumerate(bodies):
            code = f'def f(a):\n    {body}\n'
            function = Converter.text_to_function(code)
            function.entire_code = code
            function.score = [float(index), float(index)]  # dominated pairs are ranked by CodeBLEU
            population.register_function(function)
        # Generation 0 survives after its first entry (pop_size // 4 == 0); later generations after two.
        self.assertEqual(population.generation, 2)
        self.assertEqual(len(population), 2)
        self.assertIn(population.selection().score, [f.score for f in population.population])


if __name__ == '__main__':
    unittest.main()
