import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation, template_program
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.mles import MLES
from llm4ad.method.mles import MLESProfiler


def main():
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='xxx',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)
    log_dir = f'logs/MLES'  # Use run_id to avoid overwriting logs
    # batch 表示不是跑全的，而是从某一代就开始跑的
    task = MoonLanderEvaluation(whocall='mles')

    # 定义JSON文件路径
    seedpath = r'pop_init.json'

    method = MLES(llm=llm,
                  profiler=MLESProfiler(log_dir=log_dir, log_style='complex'),
                  evaluation=task,
                  max_sample_nums=50,
                  max_generations=None,
                  pop_size=8,
                  num_samplers=4,
                  num_evaluators=4,
                  debug_mode=False,
                  operators=('e1', 'e2', 'm1', 'm2_M'),  # ('e1', 'e2', 'm1_M', 'm2_M')
                  seed_path=seedpath
                  )

    method.run()


if __name__ == '__main__':
    main()
