# SPDX-License-Identifier: BSD-2-Clause
"""Offline recording, provenance and full-search replay checks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import random
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock

import numpy as np

from llm4ad.base import LLM, Evaluation
from llm4ad.method.eoh import EoH
from llm4ad.tools.llm.replay import RecordingLLM, ReplayLLM
from llm4ad.tools.reproducibility import seed_all, sha256_file, write_run_manifest, redact


class FixtureLLM(LLM):
    """A deterministic test double; not a model or scientific benchmark."""
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.closed = False

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        return '{Simple constant candidate}\ndef f():\n    return ' + str(self.calls) + '\n'

    def close(self):
        self.closed = True


class FixtureEvaluation(Evaluation):
    def __init__(self):
        super().__init__('def f():\n    return 0\n', safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func()


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'responses.jsonl'

    def test_record_replay_preserves_requests_and_raw_response(self):
        backend = FixtureLLM()
        prompt = [{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}]
        with RecordingLLM(backend, self.path, model_metadata={'model': 'fixture', 'api_key': 'private'}) as recording:
            expected = recording.draw_sample(prompt, image64s=['image'], messages=None)
        self.assertTrue(backend.closed)
        self.assertNotIn('private', self.path.read_text(encoding='utf-8'))
        replay = ReplayLLM(self.path)
        self.assertEqual(replay.remaining, 1)
        self.assertEqual(replay.draw_sample(prompt, image64s=['image'], messages=None), expected)
        self.assertEqual(replay.remaining, 0)
        with self.assertRaises(EOFError):
            replay.draw_sample(prompt)

    def test_mismatch_does_not_consume_response(self):
        with RecordingLLM(FixtureLLM(), self.path) as recorder:
            recorder.draw_sample('original')
        replay = ReplayLLM(self.path)
        with self.assertRaises(ValueError):
            replay.draw_sample('changed')
        self.assertEqual(replay.remaining, 1)
        self.assertIn('def f', replay.draw_sample('original'))

    def test_existing_recording_is_not_overwritten(self):
        self.path.write_text('keep', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            RecordingLLM(FixtureLLM(), self.path)
        self.assertEqual(self.path.read_text(encoding='utf-8'), 'keep')

    def test_records_failure_without_exception_secrets(self):
        backend = FixtureLLM()
        backend.draw_sample = Mock(side_effect=ValueError('private api key'))
        with RecordingLLM(backend, self.path) as recorder:
            with self.assertRaises(ValueError):
                recorder.draw_sample('request')
        self.assertNotIn('private api key', self.path.read_text(encoding='utf-8'))
        with self.assertRaisesRegex(RuntimeError, 'ValueError'):
            ReplayLLM(self.path).draw_sample('request')

    def test_non_string_response_and_closed_recorder(self):
        backend = FixtureLLM()
        backend.draw_sample = Mock(return_value=None)
        recorder = RecordingLLM(backend, self.path)
        with self.assertRaises(TypeError):
            recorder.draw_sample('request')
        recorder.close()
        recorder.close()
        with self.assertRaises(ValueError):
            recorder.draw_sample('request')

    def test_concurrent_records_are_complete_and_ordered_by_start(self):
        backend = FixtureLLM()
        backend.draw_sample = lambda prompt: prompt + ' response'
        with RecordingLLM(backend, self.path) as recorder:
            with ThreadPoolExecutor(max_workers=4) as executor:
                list(executor.map(recorder.draw_sample, [str(i) for i in range(12)]))
        replay = ReplayLLM(self.path)
        rows = sorted((row for row in map(json.loads, self.path.read_text(encoding='utf-8').splitlines()) if row['type'] == 'sample'), key=lambda row: row['sequence'])
        for row in rows:
            prompt = row['request']['prompt']
            self.assertEqual(replay.draw_sample(prompt), prompt + ' response')

    def test_rejects_corrupt_journals(self):
        with RecordingLLM(FixtureLLM(), self.path) as recorder:
            recorder.draw_sample('request')
        original = [json.loads(row) for row in self.path.read_text(encoding='utf-8').splitlines()]
        mutations = [lambda rows: rows[0].update(schema_version=9),
                     lambda rows: rows[1].update(sequence=3),
                     lambda rows: rows[1].update(request_sha256='wrong'),
                     lambda rows: rows[1].update(response=123)]
        for mutate in mutations:
            rows = json.loads(json.dumps(original))
            mutate(rows)
            self.path.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
            with self.assertRaises(ValueError):
                ReplayLLM(self.path)

    def test_missing_footer_or_final_sample_is_rejected(self):
        with RecordingLLM(FixtureLLM(), self.path) as recorder:
            recorder.draw_sample('one')
            recorder.draw_sample('two')
        rows = self.path.read_text(encoding='utf-8').splitlines()
        for truncated in [rows[:-1], rows[:2] + rows[-1:]]:
            self.path.write_text('\n'.join(truncated), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                ReplayLLM(self.path)

    def test_redacted_arguments_still_require_exact_values(self):
        with RecordingLLM(FixtureLLM(), self.path) as recorder:
            expected = recorder.draw_sample('request', api_key='original-private')
        self.assertNotIn('original-private', self.path.read_text(encoding='utf-8'))
        replay = ReplayLLM(self.path)
        with self.assertRaises(ValueError):
            replay.draw_sample('request', api_key='changed-private')
        self.assertEqual(replay.draw_sample('request', api_key='original-private'), expected)

    def test_close_waits_for_outstanding_request(self):
        entered, release, close_started = Event(), Event(), Event()
        backend = FixtureLLM()
        def response(prompt):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test did not release the backend')
            return 'complete'
        backend.draw_sample = response
        recorder = RecordingLLM(backend, self.path)
        def close():
            close_started.set()
            recorder.close()
        with ThreadPoolExecutor(max_workers=2) as executor:
            sample = executor.submit(recorder.draw_sample, 'request')
            self.assertTrue(entered.wait(5))
            closing = executor.submit(close)
            try:
                self.assertTrue(close_started.wait(5))
                self.assertFalse(closing.done())
            finally:
                release.set()
            self.assertEqual(sample.result(5), 'complete')
            closing.result(5)
        self.assertTrue(backend.closed)
        self.assertEqual(ReplayLLM(self.path).draw_sample('request'), 'complete')

    def test_invalid_metadata_does_not_create_a_recording(self):
        for metadata in [[], {'invalid': object()}, {'invalid': float('nan')}]:
            with self.assertRaises((ValueError, TypeError)):
                RecordingLLM(FixtureLLM(), self.path, run_metadata=metadata)
            self.assertFalse(self.path.exists())

    def test_legacy_v1_recording_remains_readable(self):
        import hashlib
        request = {'prompt': 'old', 'args': [], 'kwargs': {}}
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        records = [{'type': 'header', 'schema_version': 1, 'do_auto_trim': True, 'model': {}},
                   {'type': 'sample', 'sequence': 0, 'request': request, 'request_sha256': digest, 'response': 'legacy'}]
        self.path.write_text('\n'.join(map(json.dumps, records)), encoding='utf-8')
        self.assertEqual(ReplayLLM(self.path).draw_sample('old'), 'legacy')

    def test_full_eoh_search_replays_without_online_backend(self):
        seed_all(17)
        backend = FixtureLLM()
        recording = RecordingLLM(backend, self.path)
        first = EoH(recording, FixtureEvaluation(), profiler=None, max_sample_nums=6,
                    max_generations=None, pop_size=2, num_samplers=1, num_evaluators=1)
        with contextlib.redirect_stdout(io.StringIO()):
            first.run()
        self.assertEqual(first._tot_sample_nums, 6)
        self.assertTrue(backend.closed)
        seed_all(17)
        replay = ReplayLLM(self.path)
        second = EoH(replay, FixtureEvaluation(), profiler=None, max_sample_nums=6,
                     max_generations=None, pop_size=2, num_samplers=1, num_evaluators=1)
        with contextlib.redirect_stdout(io.StringIO()):
            second.run()
        self.assertEqual(second._tot_sample_nums, 6)
        self.assertEqual(replay.remaining, 0)
        self.assertEqual([(str(f), f.score) for f in first._population.population],
                         [(str(f), f.score) for f in second._population.population])


class ExampleReplayTests(unittest.TestCase):
    def test_example_matches_results_and_rejects_changed_run_configuration(self):
        from example.reproducibility.run_record_replay import main
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, invalid = root / 'first', root / 'second', root / 'invalid'
            base = ['demo', '--samples', '2', '--seed', '17']
            with contextlib.redirect_stdout(io.StringIO()), patch('sys.argv', base + ['--output', str(first)]):
                main()
            replay = ['--replay', str(first / 'responses.jsonl')]
            with contextlib.redirect_stdout(io.StringIO()), patch('sys.argv', base + replay + ['--output', str(second)]):
                main()
            self.assertEqual((first / 'results.json').read_bytes(), (second / 'results.json').read_bytes())
            for changed in [['--seed', '18'], ['--samples', '1']]:
                with contextlib.redirect_stderr(io.StringIO()), patch('sys.argv', base + replay + ['--output', str(invalid)] + changed):
                    with self.assertRaises(SystemExit):
                        main()
                self.assertFalse(invalid.exists())


class ManifestTests(unittest.TestCase):
    def test_seeds_hashes_manifest_and_redaction(self):
        seed_all(11)
        expected = random.random(), np.random.random()
        seed_all(11)
        self.assertEqual(expected, (random.random(), np.random.random()))
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'dataset.txt'
            data.write_bytes(b'fixture')
            path = Path(directory) / 'run.json'
            manifest = write_run_manifest(path, config={'llm': {'key': 'private'}, 'workers': 1},
                                          model={'id': 'fixture', 'revision': 'fixed'}, seed=11, datasets=[data])
            self.assertEqual(manifest, json.loads(path.read_text(encoding='utf-8')))
            self.assertEqual(manifest['datasets'][0]['sha256'], sha256_file(data))
            self.assertEqual(len(manifest['installed_source_sha256']), 64)
            self.assertNotIn('private', path.read_text(encoding='utf-8'))
            self.assertEqual(redact({'_key': 'x', 'items': [{'authorization': 'y'}]}),
                             {'_key': '[REDACTED]', 'items': [{'authorization': '[REDACTED]'}]})


if __name__ == '__main__':
    unittest.main()
