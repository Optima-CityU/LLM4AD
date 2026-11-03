# SPDX-License-Identifier: BSD-2-Clause
"""A dependency-free local Ollama backend for the reproducibility example."""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

from llm4ad.base import LLM


class OllamaHTTP(LLM):
    """Use a separately installed Ollama server without downloading models.

    Args:
        model: An already installed model name.
        endpoint: Local server URL.
        seed: Server sampling seed; not a guarantee across inference versions.
        temperature: Generation temperature.
        timeout: Request deadline in seconds. Errors propagate to the caller.
    """
    def __init__(self, model, endpoint='http://127.0.0.1:11434', *, seed=17,
                 temperature=0.0, timeout=120):
        super().__init__()
        self._model = model
        self._host = endpoint.rstrip('/')
        self._timeout = timeout
        self._options = {'seed': seed, 'temperature': temperature}

    def _json(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode('utf-8')
        request = Request(self._host + path, data=data, headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=self._timeout) as response:
            return json.load(response)

    def model_info(self):
        """Return the installed model digest and server version for provenance."""
        models = self._json('/api/tags')['models']
        match = next((m for m in models if m.get('name') == self._model or m.get('model') == self._model), None)
        if match is None:
            raise ValueError('Model is not installed on the selected Ollama server')
        return {'model': self._model, 'digest': match['digest'],
                'server_version': self._json('/api/version')['version'], 'options': self._options.copy()}

    def draw_sample(self, prompt, *args, **kwargs):
        if args or any(value is not None for value in kwargs.values()):
            raise ValueError('OllamaHTTP supports text or explicit message-list prompts only')
        messages = [{'role': 'user', 'content': prompt}] if isinstance(prompt, str) else prompt
        result = self._json('/api/chat', {'model': self._model, 'messages': messages,
                                        'stream': False, 'options': self._options})
        content = result['message']['content']
        if not isinstance(content, str):
            raise TypeError('Ollama must return string message content')
        return content
