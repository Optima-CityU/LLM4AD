import time
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError

API_KEY = "你的API密钥"  # 替换为你的API密钥
BASE_URL = "https://api.lkeap.cloud.tencent.com/coding/v3" # 替换为你的API地址
MODEL = "glm-5"  # 如果失败可以换 minimax-m2.5 / kimi-k2.5 等

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=20,
    max_retries=0,
)

def main():
    print("=== LLM API 测试开始 ===")
    print("BASE_URL:", BASE_URL)
    print("MODEL:", MODEL)

    start = time.time()

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {"role": "user", "content": "Long time no see! Can you briefly introduce yourself? "}
            ],
            temperature=0,
        )

        elapsed = time.time() - start

        print("\n=== SUCCESS ===")
        print("耗时:", round(elapsed, 2), "秒")
        print("返回内容:", resp.choices[0].message.content)

    except APITimeoutError:
        print("\n=== ERROR: TIMEOUT ===")

    except APIConnectionError as e:
        print("\n=== ERROR: CONNECTION ===")
        print(e)

    except APIStatusError as e:
        print("\n=== ERROR: STATUS ===")
        print("状态码:", e.status_code)
        print("响应体:", getattr(e, "body", e))

    except Exception as e:
        print("\n=== UNKNOWN ERROR ===")
        print(type(e).__name__, e)


if __name__ == "__main__":
    main()