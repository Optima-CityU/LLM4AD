import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.ails2 import Ails2Evaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh_java import EoH_Java
from llm4ad.method.eoh_java import EoH_java_Profiler
import os
import json
from llm4ad.base import JavaScripts

def main():
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-0hCjhh3wBUP7H2TQF9B6D290Ee604cAc88633dDc5f68B0Ed',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=60)

    task = Ails2Evaluation()

    method = EoH_Java(llm=llm,
                      profiler=EoH_java_Profiler(log_dir='logs', log_style='complex'),
                      evaluation=task,
                      max_sample_nums=48,
                      max_generations=5,
                      pop_size=8,
                      num_samplers=4,
                      num_evaluators=8,
                      debug_mode=True)

    # 定义JSON文件路径
    seedpath = r'pop_initial.json'  # 12代

    # 检查文件是否存在
    if os.path.exists(seedpath):
        # 打开并读取JSON文件
        with open(seedpath, 'r', encoding='utf-8') as file:
            seeds = json.load(file)
    else:
        print(f"文件 {seedpath} 不存在")

    # 取出 programs database
    prog_db = method._population
    profiler = method._profiler

    # 对每个 seed function: 1) 加入种群 2) 记录到 profiler
    for seed_individual in seeds:
        seed_str = seed_individual['function']
        # 用 funsearch 中的 evaluator 评估 seed
        score, eval_time = method._evaluator.evaluate_java_record_time(program=seed_str)
        # 讲 seed 转化成 function 实例

        java_script = JavaScripts(
            algorithm='',
            java_code=seed_str,
            score=score,
            evaluate_time=eval_time,
        )

        if profiler is not None:
            profiler.register_java(java_script)
            if isinstance(profiler, EoH_java_Profiler):
                profiler.register_population(prog_db)
            method._tot_sample_nums += 1

        # register to the population
        prog_db.register_function(java_script)

    method.run()


if __name__ == '__main__':
    main()
