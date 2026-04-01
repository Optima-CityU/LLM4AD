import time
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError
from config import get_saved_llm_parameters
from llm4ad.prompts import load_prompt_text

llm_config = get_saved_llm_parameters()
API_KEY = llm_config.get('key', '')
BASE_URL = llm_config.get('host', '')
MODEL = llm_config.get('model', '')

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    timeout=20,
    max_retries=10,
)

def main():
    if not API_KEY or not BASE_URL or not MODEL:
        print("=== 配置缺失 ===")
        print("请先在项目根目录 .env 文件中设置 LLM4AD_API_KEY、LLM4AD_API_BASE_URL 和 LLM4AD_MODEL_ID。")
        return

    print("=== LLM API 测试开始 ===")
    print("BASE_URL:", BASE_URL)
    print("MODEL:", MODEL)

    start = time.time()

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": load_prompt_text('test_llm', 'system.txt')},
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
