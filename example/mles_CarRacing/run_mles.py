import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.car_racing import RacingCarEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.mles import MLES
from llm4ad.method.mles import MLESProfiler


def main():
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-xxxx',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)
    log_dir = f'logs/MLES'  # Use run_id to avoid overwriting logs

    seeds = [1]
    instance_set = {}
    for id, seed in enumerate(seeds):
        instance_set[id] = seed

    # Using
    using_algo_designed_path = ""
    # Using_seeds = [i for i in range(20, 60)]
    Using_seeds = [i for i in range(10, 20)]
    # Using_seeds = seeds
    ins_to_be_solve_set = {}
    for id, seed in enumerate(Using_seeds):
        ins_to_be_solve_set[id] = seed

    run_mode = 'Training'  # Training, Using, Combined
    task = RacingCarEvaluation(whocall='mles',
                               run_mode=run_mode,
                               instance_set=instance_set,
                               ins_to_be_solve_set=ins_to_be_solve_set,
                               objective_value=100)

    # 定义JSON文件路径
    seedpath = r'pop_init.json'

    method = MLES(llm=llm,
                  profiler=MLESProfiler(log_dir=log_dir, log_style='complex', run_mode=run_mode,
                                        using_algo_designed_path=using_algo_designed_path),
                  evaluation=task,
                  max_sample_nums=100,
                  max_generations=None,
                  pop_size=16,
                  num_samplers=8,
                  num_evaluators=8,
                  debug_mode=False,
                  operators=('e1', 'e2', 'm1_M', 'm2_M'),  # ('e1', 'e2', 'm1_M', 'm2_M')
                  seed_path=seedpath
                  )

    method.run()


if __name__ == '__main__':
    main()
