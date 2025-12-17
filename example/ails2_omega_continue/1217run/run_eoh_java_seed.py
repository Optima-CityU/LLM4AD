import sys

sys.path.append('../../')  # This is for finding all the modules

from ails2_omega import Ails2Evaluation
from ails2_omega import template_program
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

    task = Ails2Evaluation(dump_java_output_on_finish=False, instances_subdir='XLDemo', max_parallel_instances=4,
                           test_mode=True)

    method = EoH_Java(llm=llm,
                      profiler=EoH_java_Profiler(log_dir='logs', log_style='complex'),
                      evaluation=task,
                      max_sample_nums=20,
                      max_generations=None,
                      pop_size=6,
                      num_samplers=2,
                      num_evaluators=2,
                      debug_mode=False)

    # 定义JSON文件路径
    seedpath = r'init_pop_acc_6ind.json'  # 12代

    # 检查文件是否存在
    if os.path.exists(seedpath):
        # 打开并读取JSON文件
        with open(seedpath, 'r', encoding='utf-8') as file:
            seeds = json.load(file)

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
    else:
        print(f"文件 {seedpath} 不存在, 将原本的模块作为初始个体之一")
        # 取出 programs database
        prog_db = method._population
        profiler = method._profiler

        seed_str = template_program
        # print('初始个体为：')
        # print(seed_str)
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
