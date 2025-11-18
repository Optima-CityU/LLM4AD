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
import socket  # <--- 新增：用于底层代理连接
import ssl  # <--- 新增：用于 SSL 握手和跳过验证
from typing import Any
import traceback


class LLM:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        raise NotImplementedError


class HttpsApi(LLM):
    def __init__(self, host, key, model, timeout=60, proxy_host=None, proxy_port=None, **kwargs):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
            proxy_host: 代理服务器主机名或IP（可选）。
            proxy_port: 代理服务器端口（可选）。
        """
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

        # --- 新增：代理配置 ---
        self._proxy_host = proxy_host
        self._proxy_port = proxy_port
        # ----------------------

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        # 创建不验证 SSL 证书的上下文（相当于 curl -k）
        try:
            unverified_context = ssl._create_unverified_context()
        except AttributeError:
            unverified_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            unverified_context.check_hostname = False
            unverified_context.verify_mode = ssl.CERT_NONE

        while True:
            start_time = time.time()
            conn = None
            try:
                # -----------------------------------------------------------------
                # --- 连接建立逻辑：区分代理和直接连接 ---
                # -----------------------------------------------------------------

                if self._proxy_host and self._proxy_port:
                    # ====== 启用代理连接 ======
                    print(
                        f"[{time.strftime('%H:%M:%S', time.localtime())}] 尝试通过代理 {self._proxy_host}:{self._proxy_port} 建立连接...")

                    target_host = self._host
                    target_port = 443
                    if ':' in target_host:
                        target_host, target_port = target_host.split(':')
                        target_port = int(target_port)

                    # 1. 连接到代理服务器
                    s = socket.create_connection((self._proxy_host, self._proxy_port), timeout=self._timeout)

                    # 2. 建立 CONNECT 隧道
                    connect_msg = f"CONNECT {target_host}:{target_port} HTTP/1.1\r\nHost: {target_host}\r\n\r\n"
                    s.sendall(connect_msg.encode('ascii'))

                    # 3. 接收代理响应（检查 200 OK）
                    proxy_response = s.recv(4096).decode('ascii')

                    if not proxy_response.startswith('HTTP/1.1 200'):
                        s.close()
                        raise RuntimeError(f"Proxy tunnel failed. Response: {proxy_response.splitlines()[0]}")

                    # 4. 将 socket 转换为 SSL socket
                    conn_socket = unverified_context.wrap_socket(s, server_hostname=target_host)

                    # 5. 替换 http.client.HTTPSConnection 对象
                    conn = http.client.HTTPSConnection(target_host, port=target_port, timeout=self._timeout)
                    conn.sock = conn_socket  # 传入已连接且 SSL 包装过的 socket

                else:
                    # ====== 无代理直接连接 (原有逻辑) ======
                    print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 尝试建立直接 HTTPS 连接...")
                    conn = http.client.HTTPSConnection(self._host, timeout=self._timeout, context=unverified_context)
                    conn.connect()  # 原有代码在这里隐式或显式地连接

                connect_end_time = time.time()
                print(
                    f"[{time.strftime('%H:%M:%S', time.localtime())}] 连接建立成功。耗时: {connect_end_time - start_time:.2f}s")

                # -----------------------------------------------------------------
                # --- 发送请求和获取响应 (原有逻辑保持不变) ---
                # -----------------------------------------------------------------

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

                conn.request('POST', '/v1/chat/completions', payload, headers)
                res = conn.getresponse()

                # 诊断：打印状态码
                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] 收到响应头。状态: {res.status}")

                if res.status != 200:
                    data = res.read().decode('utf-8')
                    raise RuntimeError(f"API request failed with status {res.status}: {res.reason}. Response: {data}")

                data = res.read().decode('utf-8')

                # 解析 JSON 和提取内容 (原有逻辑)
                data_json = json.loads(data)
                response = data_json['choices'][0]['message']['content']

                if self.debug_mode:
                    self._cumulative_error = 0
                return response

            except Exception as e:
                # 错误处理逻辑 (原有逻辑加上诊断)
                self._cumulative_error += 1

                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] !!! 捕获到异常 !!!")
                print(f"异常类型: {type(e).__name__}. 错误信息: {e}")

                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                                           f'请检查 API host, API key, 和代理设置。')
                else:
                    print(f'{self.__class__.__name__} error: {traceback.format_exc()}.')
                    time.sleep(2)
                continue
            finally:
                if conn:
                    # 确保无论成功与否，连接都被关闭
                    conn.close()