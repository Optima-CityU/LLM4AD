# SPDX-License-Identifier: BSD-2-Clause
"""Core parsing, transformation and real process-boundary tests."""
from __future__ import annotations

import multiprocessing
import time
import unittest

from llm4ad.base import Evaluation, SecureEvaluator, SampleTrimmer, ModifyCode
from llm4ad.base import TextFunctionProgramConverter as Converter


class ScalarEvaluation(Evaluation):
    def __init__(self, **kwargs):
        super().__init__('def f():\n    return 7\n', **kwargs)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func() + kwargs.get('offset', 0)


class ParsingTests(unittest.TestCase):
    def test_multiline_signature_is_not_a_function_body(self):
        sample = 'Description\n```python\ndef f(\n    x: int,\n    scale=(1, 2),\n) -> int:\n    return x + scale[0]\n```'
        body = SampleTrimmer.trim_preface_of_function(sample)
        program = SampleTrimmer.sample_to_program(body, 'def f(x, scale=(1, 2)):\n    return x\n')
        self.assertEqual(program.exec()[0](5), 6)

    def test_one_line_generated_function(self):
        body = SampleTrimmer.trim_preface_of_function('def f(x): return x * 3')
        program = SampleTrimmer.sample_to_program(body, 'def f(x):\n    return x\n')
        self.assertEqual(program.exec()[0](4), 12)

    def test_empty_response_and_plain_completion(self):
        self.assertEqual(SampleTrimmer.auto_trim(''), '')
        self.assertEqual(SampleTrimmer.auto_trim('    return 3'), '    return 3')
        self.assertIsNone(SampleTrimmer.sample_to_program('', 'def f():\n    return 0\n'))

    def test_converter_round_trip_and_function_replacement(self):
        template = 'import math\n\ndef f(x):\n    """Example."""\n    return math.sqrt(x)\n'
        program = Converter.text_to_program(template)
        self.assertEqual(program.exec()[0](9), 3)
        replacement = Converter.function_to_program('def f(x):\n    return x * 2\n', program)
        self.assertEqual(replacement.exec()[0](9), 18)
        self.assertEqual(program.exec()[0](9), 3)

    def test_converter_rejects_invalid_or_ambiguous_functions(self):
        self.assertIsNone(Converter.text_to_function('def invalid('))
        with self.assertRaises(ValueError):
            Converter.text_to_function('def a():\n    pass\ndef b():\n    pass\n')

    def test_seed_import_preserves_future_and_numpy_alias(self):
        source = '"""Example."""\nfrom __future__ import annotations\nimport numpy\n\ndef f():\n    return np.random.random()\n'
        source = ModifyCode.add_numpy_random_seed_to_func(source, 'f', 19)
        namespace = {}
        exec(source, namespace)
        self.assertEqual(namespace['f'](), namespace['f']())

    def test_protected_division_is_executable(self):
        source = ModifyCode.replace_div_with_protected_div('def f(x):\n    return 1 / x\n', delta=0.5)
        namespace = {}
        exec(source, namespace)
        self.assertEqual(namespace['f'](0), 2)


class ProcessEvaluationTests(unittest.TestCase):
    def test_direct_success_and_exception(self):
        evaluator = SecureEvaluator(ScalarEvaluation(safe_evaluate=False))
        result, duration = evaluator.evaluate_program_record_time('def f():\n    return 7\n', offset=2)
        self.assertEqual(result, 9)
        self.assertGreaterEqual(duration, 0)
        self.assertIsNone(evaluator.evaluate_program('def f():\n    raise ValueError("invalid")\n'))

    def test_spawn_and_available_fork(self):
        before = multiprocessing.get_start_method(allow_none=True)
        for fork in ([False, True] if 'fork' in multiprocessing.get_all_start_methods() else [False]):
            with self.subTest(fork=fork):
                evaluator = SecureEvaluator(ScalarEvaluation(fork_proc=fork, timeout_seconds=10))
                self.assertEqual(evaluator.evaluate_program('def f():\n    return 7\n', offset=1), 8)
        self.assertEqual(multiprocessing.get_start_method(allow_none=True), before)

    def test_timeout_terminates_worker(self):
        evaluator = SecureEvaluator(ScalarEvaluation(fork_proc=False, timeout_seconds=0.25))
        before = {p.pid for p in multiprocessing.active_children()}
        self.assertIsNone(evaluator.evaluate_program('def f():\n    while True:\n        pass\n'))
        self.assertEqual({p.pid for p in multiprocessing.active_children()}, before)

    def test_dead_worker_without_deadline_does_not_hang(self):
        evaluator = SecureEvaluator(ScalarEvaluation(fork_proc=False, timeout_seconds=None))
        started = time.monotonic()
        self.assertIsNone(evaluator.evaluate_program('def f():\n    raise SystemExit(0)\n'))
        self.assertLess(time.monotonic() - started, 10)

    def test_random_seed_without_original_numpy_import(self):
        evaluator = SecureEvaluator(ScalarEvaluation(safe_evaluate=False, random_seed=19))
        source = 'def f():\n    return np.random.random()\n'
        self.assertEqual(evaluator.evaluate_program(source), evaluator.evaluate_program(source))

    def test_invalid_context_setting(self):
        with self.assertRaises(ValueError):
            SecureEvaluator(ScalarEvaluation(fork_proc='invalid'))


if __name__ == '__main__':
    unittest.main()
