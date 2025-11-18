from __future__ import annotations

import json
import time
from typing import Any
import traceback
import requests  # <--- 引入 requests 库


# 假设 LLM 基类来自 '...base'
class LLM:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        raise NotImplementedError


class RequestsApi(LLM):
    def __init__(self, host, key, model, timeout=60, proxy_url=None, **kwargs):  # <--- 简化代理参数，只接收完整的 URL
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
            proxy_url: 完整的代理 URL，例如 'http://user:pass@proxy.com:8080'（可选）。
        """
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

        # --- 代理配置 ---
        self._proxies = None
        if proxy_url:
            # requests 使用字典配置代理，HTTPS 请求通过 HTTP 代理需要这样配置
            self._proxies = {
                'https': proxy_url
            }
        # ----------------------

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]

        # API 的完整 URL
        api_url = f'https://{self._host}/v1/chat/completions'

        # 请求体 (Payload)
        request_data = {
            'max_tokens': self._kwargs.get('max_tokens', 4096),
            'top_p': self._kwargs.get('top_p', None),
            'temperature': self._kwargs.get('temperature', 1.0),
            'model': self._model,
            'messages': prompt
        }

        # 请求头 (Headers)
        headers = {
            'Authorization': f'Bearer {self._key}',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json'
        }

        while True:
            start_time = time.time()
            try:
                print(f"[{time.strftime('%H:%M:%S', time.localtime(start_time))}] --- 诊断信息 ---")
                print(f"URL: {api_url}, MODEL: {self._model}, TIMEOUT: {self._timeout}s")
                if self._proxies:
                    print(f"使用代理: {self._proxies['https']}")

                # --- 使用 requests.post 发送请求 ---
                # verify=False 实现了 curl -k (跳过 SSL 验证)
                # proxies=self._proxies 实现了代理支持
                response = requests.post(
                    url=api_url,
                    headers=headers,
                    json=request_data,  # requests 自动处理 dict 到 JSON 的序列化
                    timeout=self._timeout,
                    verify=False,  # 相当于 curl -k
                    proxies=self._proxies  # 代理支持
                )

                end_time = time.time()

                # --- 检查 HTTP 状态码 ---
                # raise_for_status() 会对 4xx/5xx 状态码抛出异常
                response.raise_for_status()

                # --- 提取回复内容 ---
                data_json = response.json()

                # 诊断信息
                print(
                    f"[{time.strftime('%H:%M:%S', time.localtime())}] 请求成功。HTTP Status: {response.status_code}. 总耗时: {end_time - start_time:.2f}s")

                # 尝试提取回复内容
                try:
                    llm_response = data_json['choices'][0]['message']['content']
                except (KeyError, IndexError):
                    print(f"!!! API Structure Error !!! Full JSON: {json.dumps(data_json, indent=2)}")
                    raise RuntimeError("API response missing expected 'choices' or 'content' keys.")

                if self.debug_mode:
                    self._cumulative_error = 0
                return llm_response

            except requests.exceptions.RequestException as e:
                # requests 库的异常处理更专业，能捕获连接、超时、HTTP错误等
                self._cumulative_error += 1

                print(f"[{time.strftime('%H:%M:%S', time.localtime())}] !!! 捕获到请求异常 !!!")
                print(f"异常类型: {type(e).__name__}")

                if isinstance(e, requests.exceptions.HTTPError):
                    print(f"HTTP Status: {e.response.status_code}. Raw Response: {e.response.text}")
                elif isinstance(e, requests.exceptions.Timeout):
                    print(f"连接或读取超时: {self._timeout}秒。")
                elif isinstance(e, requests.exceptions.ProxyError) and self._proxies:
                    print(f"代理错误。请检查代理地址是否正确，以及认证是否成功。")

                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(f'{self.__class__.__name__} 错误: {e}. 请检查 API host/key/代理设置。')
                else:
                    print(f'{self.__class__.__name__} 错误: {traceback.format_exc()}.')
                    print('等待 2 秒后重试...')
                    time.sleep(2)
                continue
            except Exception as e:
                # 捕获其他非 requests 异常，如 JSONDecodeError
                self._cumulative_error += 1
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(f'{self.__class__.__name__} 错误: {e}.')
                else:
                    print(f'{self.__class__.__name__} 错误: {traceback.format_exc()}.')
                    time.sleep(2)
                continue


# -------------------- 示例用法 --------------------

if __name__ == '__main__':
    # --- 代理模式配置示例（带认证）---
    # 注意：这里的 '!' 不需要转义，直接输入即可。
    PROXY_URL = "http://XXXX:XXXXX!@proxy.XXXX.com:8080"

    # --- 直接连接配置示例（无代理）---
    # PROXY_URL = None

    # LLM API配置
    LLM_HOST = "api.openai.com"  # 假设是 OpenAI 兼容的 API Host
    LLM_KEY = "your-secret-api-key"
    LLM_MODEL = "gpt-3.5-turbo"

    try:
        api = HttpsApi(
            host=LLM_HOST,
            key=LLM_KEY,
            model=LLM_MODEL,
            timeout=10,
            proxy_url=PROXY_URL,  # 传入完整的代理 URL
            debug_mode=True
        )

        test_prompt = "请用一句话描述人工智能的发展趋势。"
        response = api.draw_sample(test_prompt)
        print("\n--- LLM 回复 ---")
        print(response)

    except RuntimeError as e:
        print(f"\n操作失败: {e}")
    except Exception as e:
        print(f"\n发生未知错误: {e}")