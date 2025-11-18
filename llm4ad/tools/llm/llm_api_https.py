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

import ssl

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
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        try:
            # 这适用于大多数情况
            unverified_context = ssl._create_unverified_context()
        except AttributeError:
            # 以防旧版本的Python没有 _create_unverified_context
            unverified_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            unverified_context.check_hostname = False
            unverified_context.verify_mode = ssl.CERT_NONE

        while True:
            start_time = time.time()
            try:
                print(f"[{time.strftime('%H:%M:%S', time.localtime(start_time))}] --- 诊断信息 ---")
                print(f"HOST: {self._host}, MODEL: {self._model}, TIMEOUT: {self._timeout}s")

                # --- 步骤 1: 建立连接 ---
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 尝试建立 HTTPS 连接...")

                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout, context=unverified_context)
                # 注：实际连接通常在第一个 I/O 操作（如 request）时发生，但这一行是为了设置对象。
                # 尝试提前调用 connect() 来隔离连接建立时间
                conn.connect()
                connect_end_time = time.time()
                print(
                    f"[{time.strftime('%H:%M:%S', time.localtime())}] 连接建立成功。耗时: {connect_end_time - start_time:.2f}s")

                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 4096),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': prompt
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }

                print(f"--- 诊断信息 ---")
                print(f"HOST: {self._host}")
                print(f"MODEL: {self._model}")
                print(f"PROMPT (Snippet): {prompt[0]['content'][:50]}...")
                print(f"PAYLOAD Size: {len(payload)} bytes")
                print(f"------------------")

                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 尝试发送 POST 请求...")
                conn.request('POST', '/v1/chat/completions', payload, headers)

                request_end_time = time.time()
                print(
                    f"[{time.strftime('%H:%M:%S', time.localtime())}] 请求发送完成。耗时: {request_end_time - connect_end_time:.2f}s")

                # --- 步骤 3: 获取响应 ---
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 等待接收响应头...")
                res = conn.getresponse()

                response_start_time = time.time()
                print(
                    f"[{time.strftime('%H:%M:%S', time.localtime())}] 收到响应头。耗时: {response_start_time - request_end_time:.2f}s")

                # 检查状态码和处理数据... (保持上次的逻辑)
                print(f"HTTP Status: {res.status}, Reason: {res.reason}")

                # 确保状态码是成功的 (200 OK)
                if res.status != 200:
                    data = res.read().decode('utf-8')
                    print(f"!!! HTTP ERROR !!!")
                    print(f"Failed Status Code: {res.status}")
                    print(f"Raw Response Body: {data}")
                    # 抛出异常以便被下面的except捕获，或自行处理错误
                    raise RuntimeError(f"API request failed with status {res.status}: {res.reason}. Response: {data}")

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
