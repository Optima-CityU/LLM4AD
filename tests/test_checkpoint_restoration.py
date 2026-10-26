# SPDX-License-Identifier: BSD-2-Clause
"""Checkpoint restoration preserves saved values and never rewrites history."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from llm4ad.base import TextFunctionProgramConverter as Converter
from llm4ad.tools.checkpoint import latest_snapshot, restore_attempt_count
from llm4ad.tools.profiler import ProfilerBase
from test_search_contracts import CountingLLM, construct


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in Path(directory).rglob('*.json')}


class CheckpointRestorationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def test_latest_snapshot_ignores_unrelated_files_and_sorts_numerically(self):
        folder = self.path / 'population'
        folder.mkdir()
        for name in ['pop_2.json', 'pop_10.json', '.DS_Store', 'notes.json', 'pop_final.json']:
            (folder / name).write_text('[]', encoding='utf-8')
        self.assertEqual(latest_snapshot(self.path), (str(folder / 'pop_10.json'), 10))
        with self.assertRaisesRegex(ValueError, 'No db snapshot'):
            latest_snapshot(self.path, 'prog_db', 'db')

    def check_population(self, name):
        try:
            module = importlib.import_module(f'llm4ad.method.{name}.resume')
        except ModuleNotFoundError as error:
            if error.name.split('.')[0] in {'pymoo', 'codebleu'}:
                self.skipTest(f'Optional dependency {error.name} is unavailable')
            raise
        folder = self.path / 'population'
        folder.mkdir()
        vector = name in {'meoh', 'moead', 'nsga2'}
        rows = [{'function': 'def f():\n    return 0\n', 'score': [0, 1] if vector else 0, 'algorithm': 'first'},
                {'function': 'def f():\n    return -2\n', 'score': [-2, 3] if vector else -2, 'algorithm': 'second'}]
        (folder / 'pop_7.json').write_text(json.dumps(rows), encoding='utf-8')
        before = snapshot(self.path)
        template = Converter.text_to_program('import math\ndef f():\n    return 0\n')
        population = module._resume_pop(str(self.path), 4, template_program=template)
        self.assertEqual(population.generation, 7)
        self.assertEqual([str(f) for f in population.population], [str(Converter.text_to_function(row['function'])) for row in rows])
        self.assertEqual([f.score for f in population.population], [row['score'] for row in rows])
        self.assertEqual([f.algorithm for f in population.population], ['first', 'second'])
        self.assertEqual(len(population._next_gen_pop), 0)
        self.assertIn('import math', population.population[0].entire_code)
        self.assertEqual(snapshot(self.path), before)

    def test_corrupt_population_is_rejected(self):
        from llm4ad.method.eoh.resume import _resume_pop
        folder = self.path / 'population'
        folder.mkdir()
        path = folder / 'pop_1.json'
        for payload in [{}, [], [{'function': 'invalid', 'score': 1}],
                        [{'function': 'def f():\n    return 0\n', 'score': 'bad'}],
                        [{'function': 'def f():\n    return 0\n', 'score': float('nan')}],
                        [{'function': 'def f():\n    return 0\n', 'score': 0, 'program': 42}]]:
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaises(ValueError):
                _resume_pop(str(self.path), 2)

    def test_resume_preserves_failed_scores_and_history_bytes(self):
        from llm4ad.method.randsample.resume import resume_randsample
        samples = self.path / 'samples'
        samples.mkdir()
        rows = [{'sample_order': 1, 'function': 'def f():\n    return 0\n', 'score': 0},
                {'sample_order': 2, 'function': 'def f():\n    return -1\n', 'score': None}]
        (samples / 'samples_1~200.json').write_text(json.dumps(rows), encoding='utf-8')
        before = snapshot(self.path)
        profiler = ProfilerBase(str(self.path), create_random_path=False, log_style='simple')
        llm = CountingLLM()
        method = construct(self, 'randsample', llm, budget=2, profiler=profiler)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            resume_randsample(method)
        self.assertEqual(method._tot_sample_nums, 2)
        self.assertEqual(profiler._evaluate_success_program_num, 1)
        self.assertEqual(profiler._evaluate_failed_program_num, 1)
        self.assertEqual(profiler._cur_best_program_score, 0)
        self.assertEqual(snapshot(self.path), before)
        method.run()
        self.assertEqual(llm.calls, 0)

    def test_persisted_attempts_include_malformed_unlogged_responses(self):
        from llm4ad.method.randsample.resume import resume_randsample
        llm = CountingLLM(malformed=True)
        profiler = ProfilerBase(str(self.path), create_random_path=False, log_style='simple')
        first = construct(self, 'randsample', llm, budget=3, profiler=profiler)
        with contextlib.redirect_stdout(io.StringIO()):
            first.run()
        state = json.loads((self.path / 'search_state.json').read_text(encoding='utf-8'))
        self.assertEqual(state['claimed_attempts'], 3)
        self.assertEqual(state['completed_attempts'], 3)
        self.assertEqual(state['history_samples'], 1)  # The evaluated template only.
        second_llm = CountingLLM()
        second_profiler = ProfilerBase(str(self.path), create_random_path=False, log_style='simple')
        second = construct(self, 'randsample', second_llm, budget=5, profiler=second_profiler)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            resume_randsample(second)
            second.run()
        self.assertEqual(second_llm.calls, 2)
        self.assertEqual(second._tot_sample_nums, 5)
        self.assertEqual(len(json.loads(next((self.path / 'samples').glob('samples_*~*.json')).read_text(encoding='utf-8'))), 3)

    def test_hillclimb_restores_zero_best_score_and_can_continue(self):
        from llm4ad.method.hillclimb.resume import resume_hillclimb
        samples = self.path / 'samples'
        samples.mkdir()
        rows = [{'sample_order': 1, 'function': 'def f():\n    return 0\n', 'score': 0},
                {'sample_order': 2, 'function': 'def f():\n    return -2\n', 'score': -2}]
        (samples / 'samples_1~200.json').write_text(json.dumps(rows), encoding='utf-8')
        llm = CountingLLM()
        profiler = ProfilerBase(str(self.path), create_random_path=False, log_style='simple')
        method = construct(self, 'hillclimb', llm, budget=3, profiler=profiler)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            resume_hillclimb(method)
            self.assertEqual(method._best_function_found.score, 0)
            method.run()
        self.assertEqual(llm.calls, 1)
        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual(method._best_function_found.score, 1)

    def test_eoh_final_population_checkpoint_round_trip(self):
        from llm4ad.method.eoh import EoHProfiler
        from llm4ad.method.eoh.resume import resume_eoh
        profiler = EoHProfiler(str(self.path), create_random_path=False, log_style='simple')
        first = construct(self, 'eoh', CountingLLM(), budget=5, profiler=profiler)
        with contextlib.redirect_stdout(io.StringIO()):
            first.run()
        expected = [(str(f), f.score, f.algorithm) for f in first._population.population]
        generation = first._population.generation
        before = snapshot(self.path / 'samples')
        llm = CountingLLM()
        second_profiler = EoHProfiler(str(self.path), create_random_path=False, log_style='simple')
        second = construct(self, 'eoh', llm, budget=5, profiler=second_profiler)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            resume_eoh(second, str(self.path))
            self.assertEqual([(str(f), f.score, f.algorithm) for f in second._population.population], expected)
            self.assertEqual(second._population.generation, generation)
            self.assertEqual(second._tot_sample_nums, 5)
            second.run()
        self.assertEqual(llm.calls, 0)
        self.assertEqual(snapshot(self.path / 'samples'), before)

    def test_stale_or_wrong_method_counters_fail_explicitly(self):
        path = self.path / 'search_state.json'
        state = {'schema_version': 1, 'method': 'EoH', 'history_samples': 2,
                 'completed_attempts': 3, 'claimed_attempts': 4}
        path.write_text(json.dumps(state), encoding='utf-8')
        self.assertEqual(restore_attempt_count(self.path, 2, 'EoH'), 4)
        with self.assertRaises(ValueError):
            restore_attempt_count(self.path, 2, 'HillClimb')
        # Counters saved just before/after a sample write (hard crash) stay resumable.
        for count, expected in [(1, 4), (3, 4), (6, 6)]:
            with self.assertWarns(RuntimeWarning):
                self.assertEqual(restore_attempt_count(self.path, count, 'EoH'), expected)
        path.write_text(json.dumps({**state, 'claimed_attempts': 1}), encoding='utf-8')
        with self.assertRaises(ValueError):
            restore_attempt_count(self.path, 2, 'EoH')

    def test_unsupported_lhns_checkpoint_has_a_clear_error(self):
        from llm4ad.method.lhns.resume import resume_lhns, resume_eoh
        for restore in [resume_lhns, resume_eoh]:
            with self.assertRaisesRegex(NotImplementedError, 'LHNS does not yet save'):
                restore(None, str(self.path))

    def test_mcts_resume_does_not_generate_a_new_initial_population(self):
        from llm4ad.method.mcts_ahd.resume import resume_ma
        folder = self.path / 'population'
        folder.mkdir()
        rows = [{'function': 'def f():\n    return 0\n', 'score': 0, 'algorithm': 'first'},
                {'function': 'def f():\n    return -2\n', 'score': -2, 'algorithm': 'second'}]
        (folder / 'pop_3.json').write_text(json.dumps(rows), encoding='utf-8')
        samples = self.path / 'samples'
        samples.mkdir()
        (samples / 'samples_1.json').write_text(json.dumps([{**row, 'sample_order': i + 1} for i, row in enumerate(rows)]), encoding='utf-8')
        llm = CountingLLM()
        profiler = ProfilerBase(str(self.path), create_random_path=False, log_style='simple')
        method = construct(self, 'mcts_ahd', llm, budget=2, profiler=profiler)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            resume_ma(method, str(self.path))
            method.run()
        self.assertEqual(llm.calls, 0)
        self.assertEqual(method._population.generation, 3)


for method_name in ['eoh', 'reevo', 'mcts_ahd', 'meoh', 'moead', 'nsga2']:
    def check(self, name=method_name):
        self.check_population(name)
    setattr(CheckpointRestorationTests, 'test_' + method_name + '_population_values_and_order', check)


if __name__ == '__main__':
    unittest.main()
