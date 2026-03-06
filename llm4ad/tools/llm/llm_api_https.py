# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import http.client
import json
import time
from typing import Any
import traceback
from ...base import LLM


class HttpsApi(LLM):
    def __init__(self, host, key, model, timeout=60, **kwargs):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
        """
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """
        Handle message construction with backward compatibility:
        1. If 'messages' in kwargs: use it directly.
        2. If 'prompt' is a list: use it directly (Legacy support).
        3. If 'prompt' is a str:
           - With images: Construct multimodal message.
           - Without images: Construct simple text message (Legacy behavior).
        """
        image64s = kwargs.get('image64s', None)  # List[str]
        messages_input = kwargs.get('messages', None)

        # 1. 优先处理显式传入的 messages
        if messages_input is not None:
            if isinstance(messages_input, dict):
                messages = [messages_input]
            else:
                messages = messages_input

        # 2. 兼容旧逻辑：如果 prompt 本身就是列表（对话历史），直接使用
        elif not isinstance(prompt, str):
            messages = prompt

        # 3. 处理 prompt 为字符串的情况 (新旧逻辑结合)
        else:
            text_content = prompt.strip()

            # 如果有图片，必须使用 list[dict] 结构
            if image64s:
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

            # 如果没有图片，保持原有的简单字符串结构 (最大化兼容性)
            else:
                messages = [{'role': 'user', 'content': text_content}]

        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)

                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 8192),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': messages
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', '/v1/chat/completions', payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)

                # print(data)
                response = data['choices'][0]['message']['content']
                if self.debug_mode:
                    self._cumulative_error = 0
                return response

            except Exception as e:
                self._cumulative_error += 1
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                                           f'You may check your API host and API key.')
                else:
                    print(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                          f'You may check your API host and API key.')
                    time.sleep(2)
                continue

    # def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
    #     """
    #     Handle message construction:
    #     - If 'messages' is explicitly provided, use it as the payload.
    #     - If 'messages' is None, build it from 'prompt' and 'images':
    #         a) Text only: Wrap prompt in a standard user message format.
    #         b) Multimodal: Combine prompt text and image URLs into a single user message content list.
    #     """
    #     image64s = kwargs.get('image64s', None)  # List[str]
    #     messages_input = kwargs.get('messages', None)   # messages
    #
    #     if messages_input is not None:
    #         if isinstance(messages_input, dict):
    #             messages = [messages_input]  # 单消息包装为列表
    #         else:
    #             messages = messages_input
    #     else:
    #         content = []
    #         content.append({
    #                 "type": "text",
    #                 "text": prompt.strip()
    #             })
    #
    #         if image64s is not None:
    #             for image in image64s:
    #                 content.append({
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/png;base64,{image}",
    #                     }
    #                 })
    #
    #         messages = [{
    #             'role': 'user',
    #             'content': content
    #         }]
    #
    #     while True:
    #         try:
    #             conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
    #             payload = json.dumps({
    #                 'max_tokens': self._kwargs.get('max_tokens', 8192),
    #                 'top_p': self._kwargs.get('top_p', None),
    #                 'temperature': self._kwargs.get('temperature', 1.0),
    #                 'model': self._model,
    #                 'messages': messages
    #             })
    #             headers = {
    #                 'Authorization': f'Bearer {self._key}',
    #                 'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    #                 'Content-Type': 'application/json'
    #             }
    #             conn.request('POST', '/v1/chat/completions', payload, headers)
    #             res = conn.getresponse()
    #             data = res.read().decode('utf-8')
    #             data = json.loads(data)
    #             # print(data)
    #             response = data['choices'][0]['message']['content']
    #             if self.debug_mode:
    #                 self._cumulative_error = 0
    #             return response
    #         except Exception as e:
    #             self._cumulative_error += 1
    #             if self.debug_mode:
    #                 if self._cumulative_error == 10:
    #                     raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
    #                                        f'You may check your API host and API key.')
    #             else:
    #                 print(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
    #                       f'You may check your API host and API key.')
    #                 time.sleep(2)
    #             continue
