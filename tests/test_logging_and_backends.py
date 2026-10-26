# SPDX-License-Identifier: BSD-2-Clause
"""Log integrity, restored zero scores, and fake HTTP transports."""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from llm4ad.base import Function
from llm4ad.tools.checkpoint import load_samples
from llm4ad.tools.profiler import ProfilerBase
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.ollama_http import OllamaHTTP


class LogTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)

    def test_best_program_and_credentials_and_zero_score(self):
        profiler = ProfilerBase(log_dir=str(self.path), create_random_path=False, log_style='simple')
        source = 'def f():\n    return 0\n'
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                profiler.record_parameters(SimpleNamespace(_key='secret-key', _kwargs={'api_key': 'nested-secret', 'temperature': 0}),
                                           SimpleNamespace(api_key='problem-secret'), SimpleNamespace(config={'password': 'method-secret'}))
                profiler.register_function(Function('f', '', '    return 0', score=0), program=source)
            text = (self.path / 'run_log.txt').read_text(encoding='utf-8')
            self.assertNotIn('secret-key', text)
            self.assertNotIn('nested-secret', text)
            self.assertNotIn('problem-secret', text)
            self.assertNotIn('method-secret', text)
            best = json.loads((self.path / 'samples/samples_best.json').read_text(encoding='utf-8'))
            self.assertEqual(best[0]['program'], source)
            self.assertEqual(load_samples(self.path)[0]['score'], 0)
            self.assertEqual(ProfilerBase.load_logfile(str(self.path), valid_only=True)[1], [0])
            from llm4ad.method.randsample.resume import _get_all_samples_and_scores
            self.assertEqual(_get_all_samples_and_scores(str(self.path))[1], [0])
        finally:
            for handler in list(profiler._logger_txt.handlers):
                handler.close()
                profiler._logger_txt.removeHandler(handler)

    def test_legacy_and_batched_sample_files(self):
        samples = self.path / 'samples'
        samples.mkdir()
        one = {'sample_order': 1, 'function': 'def f():\n    return 0\n', 'score': 0}
        two = {**one, 'sample_order': 2, 'score': [0, 1]}
        (samples / 'samples_1.json').write_text(json.dumps(one), encoding='utf-8')
        (samples / 'samples_2~200.json').write_text(json.dumps([two]), encoding='utf-8')
        (samples / 'samples_best.json').write_text('not loaded', encoding='utf-8')
        self.assertEqual(load_samples(self.path), [one, two])

    def test_rejects_duplicate_orders_or_missing_fields(self):
        samples = self.path / 'samples'
        samples.mkdir()
        path = samples / 'samples_1.json'
        row = {'sample_order': 1, 'function': 'code', 'score': 0}
        path.write_text(json.dumps([row, row]), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_samples(self.path)
        path.write_text(json.dumps([{'function': 'code'}]), encoding='utf-8')
        with self.assertRaises(ValueError):
            load_samples(self.path)

    def _assert_multiobjective_resume(self, method):
        samples = self.path / 'samples'
        samples.mkdir()
        row = {'sample_order': 1, 'function': 'def f():\n    return 0\n', 'score': [0, 1]}
        (samples / 'samples_0~200.json').write_text(json.dumps([row]), encoding='utf-8')
        try:
            module = importlib.import_module(f'llm4ad.method.{method}.resume')
        except ModuleNotFoundError as error:
            if error.name.split('.')[0] in {'pymoo', 'codebleu'}:
                self.skipTest(f'Optional dependency {error.name} is not installed')
            raise
        self.assertEqual(module._get_all_samples_and_scores(str(self.path))[1:], ([[0, 1]], 1))

    def test_nsga2_resume_reader(self):
        self._assert_multiobjective_resume('nsga2')

    def test_moead_resume_reader(self):
        self._assert_multiobjective_resume('moead')

    def test_meoh_resume_reader(self):
        self._assert_multiobjective_resume('meoh')

    def test_no_directory_and_root_logger_isolation(self):
        root = logging.getLogger()
        handlers = root.handlers[:]
        profiler = ProfilerBase(log_style='simple')
        profiler.record_parameters(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
        with contextlib.redirect_stdout(io.StringIO()):
            profiler.register_function(Function('f', '', '    return 0', score=0))
        profiler.finish()
        self.assertEqual(root.handlers, handlers)
        self.assertEqual(profiler._num_samples, 1)

    def test_vector_scores_and_malformed_logs(self):
        samples = self.path / 'samples'
        samples.mkdir()
        path = samples / 'samples_1.json'
        row = {'sample_order': 1, 'function': 'code', 'score': [0, 1]}
        path.write_text(json.dumps([row, {**row, 'sample_order': 2, 'score': [0, float('inf')]}]), encoding='utf-8')
        self.assertEqual(ProfilerBase.load_logfile(self.path, valid_only=True)[1], [[0, 1]])
        path.write_text('broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            ProfilerBase.load_logfile(self.path)
        for update in [{'sample_order': []}, {'sample_order': True}, {'score': 'invalid'}, {'score': []}]:
            path.write_text(json.dumps([{**row, **update}]), encoding='utf-8')
            with self.assertRaises(ValueError):
                load_samples(self.path)

    def test_optional_profilers_preserve_program_and_do_not_rewrite_resume_logs(self):
        from llm4ad.tools.profiler.tensorboard_profiler import TensorboardProfiler
        from llm4ad.tools.profiler.wandb_profiler import WandBProfiler
        for cls in [TensorboardProfiler, WandBProfiler]:
            with self.subTest(profiler=cls.__name__):
                path = self.path / cls.__name__
                target = 'llm4ad.tools.profiler.' + ('tensorboard_profiler._get_summary_writer' if cls is TensorboardProfiler else 'wandb_profiler._get_wandb')
                with patch(target, return_value=Mock()):
                    args = {} if cls is TensorboardProfiler else {'wandb_project_name': 'offline-fixture'}
                    profiler = cls(log_dir=str(path), create_random_path=False, log_style='simple', **args)
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        profiler.record_parameters(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())
                        profiler.register_function(Function('f', '', '    return 0', score=0), program='complete program')
                    before = {p.name: p.read_bytes() for p in (path / 'samples').iterdir()}
                    self.assertEqual(json.loads(before['samples_best.json'])[0]['program'], 'complete program')
                    profiler.register_function(Function('f', '', '    return 1', score=1), resume_mode=True)
                    self.assertEqual(before, {p.name: p.read_bytes() for p in (path / 'samples').iterdir()})
                finally:
                    profiler.finish()


class BackendTests(unittest.TestCase):
    def test_http_generation_parameters_and_multimodal_payload(self):
        module = 'llm4ad.tools.llm.llm_api_https'
        connection = Mock()
        connection.getresponse.return_value.read.return_value = b'{"choices":[{"message":{"content":"code"}}]}'
        backend = HttpsApi('example.invalid', 'private', 'fixed-model', temperature=0.2, max_tokens=20)
        with patch(module + '.http.client.HTTPSConnection', return_value=connection):
            self.assertEqual(backend.draw_sample('hello', image64s=['image']), 'code')
        payload = json.loads(connection.request.call_args.args[2])
        self.assertEqual(payload['temperature'], 0.2)
        self.assertEqual(payload['max_tokens'], 20)
        self.assertEqual(payload['messages'][0]['content'][1]['type'], 'image_url')
        self.assertNotIn('private', connection.request.call_args.args[2])

    def test_provider_adapters_accept_generation_configuration(self):
        for module_name, class_name in [('qwen', 'QwenAPI'), ('zhipu', 'ZhipuAPI'),
                                       ('volcengine', 'VolcengineAPI'), ('baiduqianfan', 'BaiduQianfanAPI'),
                                       ('tencentcloud', 'TencentCloudAPI')]:
            with self.subTest(provider=module_name):
                module = importlib.import_module('llm4ad.tools.llm.llm_api_' + module_name)
                backend = getattr(module, class_name)(key='private', model='model', temperature=0.2)
                self.assertEqual(backend._kwargs['temperature'], 0.2)

    def test_ollama_text_messages_and_digest_metadata(self):
        backend = OllamaHTTP('local:fixed', seed=3)
        backend._json = Mock(side_effect=[
            {'models': [{'name': 'local:fixed', 'digest': 'sha256:example'}]},
            {'version': 'fixture'}, {'message': {'content': 'code'}},
        ])
        self.assertEqual(backend.model_info()['digest'], 'sha256:example')
        self.assertEqual(backend.draw_sample('hello'), 'code')
        request = backend._json.call_args.args[1]
        self.assertFalse(request['stream'])
        self.assertEqual(request['options'], {'seed': 3, 'temperature': 0.0})

    def test_ollama_rejects_missing_model_and_unsupported_inputs(self):
        backend = OllamaHTTP('missing')
        backend._json = Mock(return_value={'models': []})
        with self.assertRaises(ValueError):
            backend.model_info()
        with self.assertRaises(ValueError):
            backend.draw_sample('hello', image64s=['image'])


if __name__ == '__main__':
    unittest.main()
