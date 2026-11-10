# SPDX-License-Identifier: BSD-2-Clause
"""LiteLLMAPI request building, retries and errors, with litellm stubbed out."""
from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from llm4ad.base import LLM, SampleTrimmer
from llm4ad.tools.llm.llm_api_litellm import LiteLLMAPI


def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _FakeLiteLLM:
    """Installs a stand-in `litellm` module whose completion records each request."""

    def __init__(self, *results):
        self.module = types.ModuleType('litellm')
        self.module.completion = mock.MagicMock(side_effect=list(results) or [_response('ok')])
        self._patch = mock.patch.dict(sys.modules, {'litellm': self.module})

    def __enter__(self):
        self._patch.start()
        return self.module.completion

    def __exit__(self, *exc):
        self._patch.stop()


def _error(name, message='boom'):
    error = type(name, (Exception,), {})(message)
    error.message = message
    return error


class LiteLLMAPITests(unittest.TestCase):
    def test_string_prompt_request(self):
        with _FakeLiteLLM(_response('def f(x):\n    return x')) as completion:
            llm = LiteLLMAPI(model='anthropic/claude-sonnet-4-5', timeout=30)
            self.assertIsInstance(llm, LLM)
            self.assertEqual(llm.draw_sample('  write f  '), 'def f(x):\n    return x')
        request = completion.call_args.kwargs
        self.assertEqual(request['model'], 'anthropic/claude-sonnet-4-5')
        self.assertEqual(request['messages'], [{'role': 'user', 'content': 'write f'}])
        self.assertEqual(request['max_tokens'], 8192)
        self.assertEqual(request['temperature'], 1.0)
        self.assertEqual(request['timeout'], 30)
        self.assertTrue(request['drop_params'])
        # Unset, LiteLLM reads the provider's own key and endpoint.
        self.assertNotIn('api_key', request)
        self.assertNotIn('api_base', request)
        self.assertNotIn('top_p', request)

    def test_key_base_and_sampling_kwargs(self):
        with _FakeLiteLLM() as completion:
            llm = LiteLLMAPI(model='litellm_proxy/alias', api_key='sk-proxy', api_base='http://localhost:4000',
                             max_tokens=512, temperature=0.2, top_p=0.9, drop_params=False,
                             reasoning_effort='low', do_auto_trim=False, debug_mode=True)
            llm.draw_sample('hi')
        request = completion.call_args.kwargs
        self.assertEqual(request['api_key'], 'sk-proxy')
        self.assertEqual(request['api_base'], 'http://localhost:4000')
        self.assertEqual((request['max_tokens'], request['temperature'], request['top_p']), (512, 0.2, 0.9))
        self.assertFalse(request['drop_params'])
        self.assertEqual(request['reasoning_effort'], 'low')
        # LLM options are kept on the instance, not sent to the provider.
        self.assertNotIn('do_auto_trim', request)
        self.assertNotIn('debug_mode', request)
        self.assertFalse(llm.do_auto_trim)
        self.assertTrue(llm.debug_mode)

    def test_messages_list_prompt_and_images(self):
        messages = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'u'}]
        with _FakeLiteLLM(_response('a'), _response('b'), _response('c')) as completion:
            llm = LiteLLMAPI(model='gemini/gemini-2.5-flash')
            llm.draw_sample('ignored', messages={'role': 'user', 'content': 'one'})
            llm.draw_sample(messages)
            llm.draw_sample('describe', image64s=['QUJD'])
        sent = [call.kwargs['messages'] for call in completion.call_args_list]
        self.assertEqual(sent[0], [{'role': 'user', 'content': 'one'}])
        self.assertEqual(sent[1], messages)
        self.assertEqual(sent[2][0]['content'][0], {'type': 'text', 'text': 'describe'})
        self.assertEqual(sent[2][0]['content'][1]['image_url']['url'], 'data:image/png;base64,QUJD')

    def test_transient_errors_are_retried(self):
        with _FakeLiteLLM(_error('RateLimitError'), _error('APIConnectionError'), _response('done')) as completion:
            llm = LiteLLMAPI(model='openai/gpt-4o-mini')
            with mock.patch('llm4ad.tools.llm.llm_api_litellm.time.sleep') as sleep, \
                    mock.patch('builtins.print'):
                self.assertEqual(llm.draw_sample('hi'), 'done')
        self.assertEqual(completion.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_configuration_errors_are_not_retried(self):
        with _FakeLiteLLM(_error('AuthenticationError', 'invalid x-api-key')) as completion:
            llm = LiteLLMAPI(model='anthropic/claude-sonnet-4-5')
            with self.assertRaisesRegex(RuntimeError, 'AuthenticationError: invalid x-api-key'):
                llm.draw_sample('hi')
        self.assertEqual(completion.call_count, 1)

    def test_debug_mode_stops_after_ten_failures(self):
        with _FakeLiteLLM(*[_error('Timeout')] * 10) as completion:
            llm = LiteLLMAPI(model='openai/gpt-4o-mini', debug_mode=True)
            with self.assertRaises(RuntimeError):
                llm.draw_sample('hi')
        self.assertEqual(completion.call_count, 10)

    def test_trimmer_and_draw_samples(self):
        with _FakeLiteLLM(_response('Here:\ndef f(x):\n    return x + 1'), _response('    return 2')):
            llm = LiteLLMAPI(model='openai/gpt-4o-mini')
            samples = SampleTrimmer(llm).draw_samples(['a', 'b'])
        self.assertEqual([s.rstrip('\n') for s in samples], ['    return x + 1', '    return 2'])

    def test_requires_model(self):
        with _FakeLiteLLM():
            with self.assertRaisesRegex(ValueError, 'requires model'):
                LiteLLMAPI()

    def test_missing_package(self):
        with mock.patch.dict(sys.modules, {'litellm': None}):
            with self.assertRaisesRegex(ImportError, r'llm4ad\[litellm\]'):
                LiteLLMAPI(model='openai/gpt-4o-mini')

    def test_lazy_registry_exposes_the_class(self):
        import llm4ad.tools.llm as llm_tools
        self.assertIn('LiteLLMAPI', llm_tools.__all__)
        self.assertIs(llm_tools.LiteLLMAPI, LiteLLMAPI)


if __name__ == '__main__':
    unittest.main()
