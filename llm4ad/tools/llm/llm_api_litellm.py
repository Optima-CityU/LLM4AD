# SPDX-License-Identifier: BSD-2-Clause
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2026/11/11
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Licensed under BSD-2-Clause; see the project LICENSE file.
# Academic citation is appreciated, but is not a license condition.
# Suggested reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For voluntary commercial collaboration inquiries, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import time
from typing import Any
import traceback
from ...base import LLM

# Errors that the same request will hit again: a wrong key, model or request.
# They are raised at once instead of being retried.
_NOT_RETRIED = ('AuthenticationError', 'PermissionDeniedError', 'NotFoundError',
                'BadRequestError', 'UnsupportedParamsError', 'ContextWindowExceededError')


class LiteLLMAPI(LLM):
    """LLM interface for any provider LiteLLM supports, or a LiteLLM proxy."""

    def __init__(self, model=None, api_key=None, api_base=None, timeout=60, **kwargs):
        """LiteLLM API: one interface for 100+ LLM providers through the LiteLLM SDK.
        Args:
            model    : LiteLLM model name, '<provider>/<model>', e.g. 'anthropic/claude-sonnet-4-5',
                       'gemini/gemini-2.5-flash', 'bedrock/...', 'vertex_ai/...', 'azure/<deployment>',
                       or 'litellm_proxy/<model>' for a model served by a LiteLLM proxy.
            api_key  : API key. Optional: LiteLLM reads the provider's own variable
                       (e.g. ANTHROPIC_API_KEY, GEMINI_API_KEY) when it is not given.
            api_base : API base URL. Optional, e.g. the URL of a LiteLLM proxy.
            timeout  : API timeout.
            kwargs   : max_tokens, top_p and temperature as in the other APIs; any other
                       LiteLLM completion argument (e.g. reasoning_effort, drop_params) is passed on.
        """
        if model is None:
            raise ValueError('LiteLLMAPI requires model.')
        try:
            import litellm
        except ImportError:
            raise ImportError('LiteLLMAPI requires litellm: pip install "llm4ad[litellm]"') from None
        super().__init__(do_auto_trim=kwargs.pop('do_auto_trim', True),
                         debug_mode=kwargs.pop('debug_mode', False))
        litellm.suppress_debug_info = True
        self._completion = litellm.completion
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """
        Sends a request to the LLM through LiteLLM and retrieves the generated response.

        This method supports multiple input formats for backward compatibility:
        1. Explicit 'messages' list via kwargs.
        2. A message list passed directly as the 'prompt'.
        3. Multimodal inputs (text + base64 images).
        4. Simple string prompts.

        Args:
            prompt: The text prompt or a list of message dictionaries.
            **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

        Returns:
            The string content of the LLM's response.
        """
        image64s = kwargs.get('image64s', None)  # List[str]
        messages_input = kwargs.get('messages', None)

        # --- 1. Priority: Explicit messages list ---
        if messages_input is not None:
            if isinstance(messages_input, dict):
                messages = [messages_input]
            else:
                messages = messages_input

        # --- 2. Legacy Support: prompt passed as a pre-constructed list ---
        elif not isinstance(prompt, str):
            messages = prompt

        # --- 3. Construction from String + Optional Images ---
        else:
            text_content = prompt.strip()

            if image64s:
                # Construct multimodal content structure
                content = [{
                    "type": "text",
                    "text": text_content
                }]
                for image in image64s:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                        }
                    })
                messages = [{'role': 'user', 'content': content}]

            else:
                # Construct standard text-only message
                messages = [{'role': 'user', 'content': text_content}]

        request = {
            'max_tokens': 8192,
            'temperature': 1.0,
            # Parameters a provider does not accept (e.g. top_p on some reasoning
            # models) are dropped instead of failing the request.
            'drop_params': True,
            **self._kwargs,
            'model': self._model,
            'messages': messages,
            'timeout': self._timeout,
        }
        if self._api_key:
            request['api_key'] = self._api_key
        if self._api_base:
            request['api_base'] = self._api_base
        # Unset sampling parameters keep each provider's default.
        request = {k: v for k, v in request.items() if v is not None}

        # Retry loop for handling network or API transient errors
        while True:
            try:
                response = self._completion(**request)
                # Extract content from the standard response format
                content = response.choices[0].message.content
                # Reset error counter on success
                if self.debug_mode:
                    self._cumulative_error = 0
                return content

            except Exception as e:
                if type(e).__name__ in _NOT_RETRIED:
                    raise RuntimeError(f'{self.__class__.__name__} error: {type(e).__name__}: '
                                       f'{getattr(e, "message", e)}. '
                                       f'You may check your model name, API key and API base.') from e
                self._cumulative_error += 1

                # In debug mode, crash after consecutive failures to allow debugging
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                                           f'You may check your model name, API key and API base.')
                else:
                    print(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                          f'You may check your model name, API key and API base.')
                    time.sleep(2)
                continue
