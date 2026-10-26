# SPDX-License-Identifier: BSD-2-Clause
"""Public search contracts with bounded fake backends and real small evaluations."""
from __future__ import annotations

import contextlib
import importlib
import inspect
import io
import tempfile
from pathlib import Path
from threading import Lock
import unittest

from llm4ad.base import Evaluation, LLM
from llm4ad.base.errors import ReplayExhaustedError, ReplayMismatchError
from llm4ad.tools.llm.replay import RecordingLLM, ReplayLLM
from llm4ad.tools.reproducibility import seed_all

METHODS = {'randsample': 'RandSample', 'hillclimb': 'HillClimb', 'eoh': 'EoH',
           'reevo': 'ReEvo', 'funsearch': 'FunSearch', 'lhns': 'LHNS',
           'mcts_ahd': 'MCTS_AHD', 'mles': 'MLES', 'meoh': 'MEoH',
           'moead': 'MOEAD', 'nsga2': 'NSGA2', 'partevo': 'PartEvo'}


class TinyEvaluation(Evaluation):
    def __init__(self, kind='scalar', fail_candidates=False, fail_template=False):
        super().__init__('def f():\n    return 0\n', task_description='Maximize f().', safe_evaluate=False)
        self.kind = kind
        self.fail_candidates = fail_candidates
        self.fail_template = fail_template

    def evaluate_program(self, program_str, callable_func, **kwargs):
        value = callable_func()
        if self.fail_template or (self.fail_candidates and value != 0):
            return None
        if self.kind == 'vector':
            return [float(value), -float(value)]
        if self.kind == 'images':
            return {'score': float(value), 'image': [], 'observation': {}}
        return float(value)


class CountingLLM(LLM):
    def __init__(self, *, malformed=False, stop_after=30):
        super().__init__()
        self.calls = 0
        self.closes = 0
        self.malformed = malformed
        self.stop_after = stop_after
        self.lock = Lock()

    def draw_sample(self, prompt, *args, **kwargs):
        with self.lock:
            self.calls += 1
            order = self.calls
        # Keep regressions bounded even if the method loses its budget counter.
        if order > self.stop_after:
            raise KeyboardInterrupt('Fixture safety limit reached')
        if self.malformed:
            return 'This response contains no function.'
        return '{Constant candidate}\n<concept>Constant candidate</concept>\ndef f():\n    return ' + str(order) + '\n'

    def close(self):
        self.closes += 1


def construct(test, name, llm, *, budget=3, workers=1, task=None, **overrides):
    try:
        module = importlib.import_module(f'llm4ad.method.{name}.{name}')
    except ModuleNotFoundError as error:
        if error.name.split('.')[0] in {'scipy', 'pymoo', 'codebleu', 'torch', 'transformers', 'sklearn', 'seaborn'}:
            test.skipTest(f'Optional dependency {error.name} is unavailable')
        raise
    cls = getattr(module, METHODS[name])
    parameters = inspect.signature(cls).parameters
    kind = 'vector' if name in {'meoh', 'moead', 'nsga2'} else 'images' if name == 'mles' else 'scalar'
    kwargs = {'llm': llm, 'evaluation': task or TinyEvaluation(kind), 'profiler': None,
              'max_sample_nums': budget, 'num_samplers': workers, 'num_evaluators': 2}
    for key, value in {'pop_size': 2, 'selection_num': 2, 'max_generations': None, 'init_size': 2,
                       'samples_per_prompt': 4, 'initial_sample_nums_max': 4, 'partition_number': 1,
                       'feature_used': ('objective',)}.items():
        if key in parameters:
            kwargs[key] = value
    kwargs.update(overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        return cls(**kwargs)


class SearchContracts(unittest.TestCase):
    def run_quietly(self, method):
        seed_all(17)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                method.run()
            except KeyboardInterrupt as error:
                self.fail(str(error))

    def assert_closed(self, method, llm):
        self.assertEqual(llm.closes, 1)
        self.assertTrue(method._evaluation_executor._shutdown)
        state = method._search_state
        self.assertTrue(all(not thread.is_alive() for thread in state.threads))

    def check_malformed_budget(self, name):
        llm = CountingLLM(malformed=True, stop_after=30)
        method = construct(self, name, llm)
        self.run_quietly(method)
        self.assertEqual(llm.calls, 3)
        self.assertEqual(method._tot_sample_nums, 3)
        self.assert_closed(method, llm)

    def check_valid_budget(self, name):
        llm = CountingLLM(stop_after=30)
        method = construct(self, name, llm, budget=5)
        self.run_quietly(method)
        # ReEvo also queries the LLM for reflections, outside its candidate budget.
        self.assertGreaterEqual(llm.calls, 5)
        if name not in {'reevo', 'partevo'}:
            self.assertEqual(llm.calls, 10 if name == 'mcts_ahd' else 5)
        self.assertEqual(method._tot_sample_nums, 5)
        self.assert_closed(method, llm)

    def check_replay_failure(self, name):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'responses.jsonl'
            with RecordingLLM(CountingLLM(), path):
                pass
            replay = ReplayLLM(path)
            method = construct(self, name, replay, budget=3)
            with self.assertRaises(ReplayExhaustedError):
                self.run_quietly(method)
            self.assertEqual(replay.remaining, 0)
            self.assertTrue(method._evaluation_executor._shutdown)

    def test_concurrent_sampling_respects_small_budget(self):
        for name in ['randsample', 'hillclimb', 'eoh', 'funsearch']:
            with self.subTest(method=name):
                llm = CountingLLM(stop_after=30)
                method = construct(self, name, llm, budget=5, workers=4)
                self.run_quietly(method)
                self.assertEqual(method._tot_sample_nums, 5)
                self.assertEqual(llm.calls, 5)
                self.assert_closed(method, llm)

    def test_evaluation_failures_consume_budget_and_close_resources(self):
        for name in ['randsample', 'hillclimb', 'eoh', 'mcts_ahd', 'lhns']:
            with self.subTest(method=name):
                llm = CountingLLM(stop_after=30)
                method = construct(self, name, llm, task=TinyEvaluation(fail_candidates=True))
                self.run_quietly(method)
                self.assertEqual(method._tot_sample_nums, 3)
                self.assertEqual(llm.calls, 6 if name == 'mcts_ahd' else 3)
                self.assert_closed(method, llm)

    def test_failed_template_closes_backend_without_sampling(self):
        for name in ['randsample', 'hillclimb', 'funsearch']:
            with self.subTest(method=name):
                llm = CountingLLM()
                method = construct(self, name, llm, task=TinyEvaluation(fail_template=True))
                with self.assertRaises(RuntimeError):
                    self.run_quietly(method)
                self.assertEqual(llm.calls, 0)
                self.assert_closed(method, llm)

    def test_replay_mismatch_propagates_to_caller_without_consumption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'responses.jsonl'
            with RecordingLLM(CountingLLM(), path) as recorder:
                recorder.draw_sample('different prompt')
            for name in ['randsample', 'eoh', 'mcts_ahd', 'reevo', 'funsearch']:
                with self.subTest(method=name):
                    replay = ReplayLLM(path)
                    method = construct(self, name, replay)
                    with self.assertRaises(ReplayMismatchError):
                        self.run_quietly(method)
                    self.assertEqual(replay.remaining, 1)
                    self.assertTrue(method._evaluation_executor._shutdown)

    def test_profiled_search_has_the_same_budget_and_closes_handlers(self):
        from llm4ad.tools.profiler import ProfilerBase
        for name in ['randsample', 'hillclimb', 'eoh', 'funsearch']:
            with self.subTest(method=name), tempfile.TemporaryDirectory() as directory:
                profiler = ProfilerBase(directory, create_random_path=False, log_style='simple')
                llm = CountingLLM()
                method = construct(self, name, llm, budget=3, profiler=profiler)
                self.run_quietly(method)
                self.assertEqual(method._tot_sample_nums, 3)
                self.assertEqual(llm.calls, 3)
                self.assertEqual(profiler._logger_txt.handlers, [])
                self.assert_closed(method, llm)

    def test_cleanup_failure_does_not_hide_original_replay_error(self):
        from unittest.mock import Mock
        from llm4ad.tools.profiler import ProfilerBase
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'responses.jsonl'
            with RecordingLLM(CountingLLM(), path):
                pass
            profiler = ProfilerBase(log_style='simple')
            profiler.finish = Mock(side_effect=OSError('Fixture cleanup failure'))
            replay = ReplayLLM(path)
            replay.close = Mock()
            method = construct(self, 'eoh', replay, profiler=profiler)
            with self.assertRaises(ReplayExhaustedError):
                self.run_quietly(method)
            profiler.finish.assert_called_once()
            replay.close.assert_called_once()

    def test_hsevo_does_not_retry_replay_as_a_different_prompt_format(self):
        from llm4ad.method.hsevo import HSEvo
        for kind in ['empty', 'mismatch']:
            with self.subTest(journal=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'responses.jsonl'
                with RecordingLLM(CountingLLM(), path) as recorder:
                    if kind == 'mismatch':
                        recorder.draw_sample('different prompt')
                replay = ReplayLLM(path)
                method = HSEvo(replay, TinyEvaluation(), profiler=None, max_sample_nums=3,
                               pop_size=2, init_pop_size=2, num_samplers=1, num_evaluators=1)
                expected = ReplayExhaustedError if kind == 'empty' else ReplayMismatchError
                with self.assertRaises(expected):
                    self.run_quietly(method)
                self.assertEqual(replay.remaining, 0 if kind == 'empty' else 1)
                self.assertTrue(method._evaluation_executor._shutdown)

    def test_generation_only_limits_work_without_a_sample_limit(self):
        for name in ['eoh', 'mles', 'partevo']:
            with self.subTest(method=name):
                llm = CountingLLM(stop_after=12)
                method = construct(self, name, llm, budget=None, max_generations=1)
                self.run_quietly(method)
                population = getattr(method, '_population', None) or getattr(method, '_pool', None)
                self.assertGreaterEqual(population.generation, 1)
                self.assertLessEqual(llm.calls, 6)
                self.assert_closed(method, llm)

    def test_partevo_local_seed_evaluations_respect_the_budget(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'seeds.json'
            path.write_text(json.dumps([{'function': f'def f():\n    return {i}\n', 'algorithm': 'seed'} for i in range(4)]), encoding='utf-8')
            llm = CountingLLM()
            method = construct(self, 'partevo', llm, budget=2, local_algo_base=str(path))
            self.run_quietly(method)
            self.assertEqual(method._tot_sample_nums, 2)
            self.assertEqual(llm.calls, 0)
            self.assert_closed(method, llm)

    def test_resume_mode_requires_loaded_state(self):
        for name in ['eoh', 'hillclimb', 'mcts_ahd']:
            with self.subTest(method=name):
                llm = CountingLLM()
                method = construct(self, name, llm, resume_mode=True)
                with self.assertRaisesRegex(ValueError, 'Resume mode requires'):
                    self.run_quietly(method)
                self.assertEqual(llm.calls, 0)
                self.assert_closed(method, llm)

    def test_zero_budget_does_not_call_backend(self):
        for name in ['randsample', 'eoh', 'mcts_ahd']:
            with self.subTest(method=name):
                llm = CountingLLM()
                method = construct(self, name, llm, budget=0)
                self.run_quietly(method)
                self.assertEqual(llm.calls, 0)
                self.assert_closed(method, llm)

    def test_negative_budget_is_rejected_with_cleanup(self):
        llm = CountingLLM()
        method = construct(self, 'eoh', llm, budget=-1)
        with self.assertRaises(ValueError):
            self.run_quietly(method)
        self.assert_closed(method, llm)


# Separate test names keep optional dependency skips visible per method/contract.
for method_name in METHODS:
    for contract in ['malformed_budget', 'valid_budget', 'replay_failure']:
        def check(self, name=method_name, contract_name=contract):
            getattr(self, 'check_' + contract_name)(name)
        setattr(SearchContracts, 'test_' + method_name + '_' + contract, check)


if __name__ == '__main__':
    unittest.main()
