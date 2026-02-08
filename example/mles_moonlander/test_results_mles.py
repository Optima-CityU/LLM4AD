import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation, moon_lander_feature
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.mles import MLES
from llm4ad.method.mles import MLESProfiler


def main(using_algo_designed_path):
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-qMAtcWpKnF64zZxWqyLcqXRQYEtwnyiriaB0nR5GBldQ7S0A',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)
    log_dir = f'logs/MLES'  # Use run_id to avoid overwriting logs

    seeds = [6, 9, 17, 29, 57,  # 全分布
             44, 18, 69, 26, 68,
             65, 23, 51, 93, 16,
             87, 92, 90, 22, 73,
             60, 10, 19, 97, 11,
             14, 99, 98, 8, 28,
             43, 56, 89, 15, 74]
    instance_set = {}
    for id, seed in enumerate(seeds):
        instance_set[id] = seed

    # Using
    Using_seeds = [i for i in range(100, 150)]
    # Using_seeds = seeds
    ins_to_be_solve_set = {}
    for id, seed in enumerate(Using_seeds):
        ins_to_be_solve_set[id] = seed

    run_mode = 'Using'  # Training, Using, Combined
    task = MoonLanderEvaluation(whocall='mles', instance_set=instance_set, run_mode=run_mode,
                                ins_to_be_solve_set=ins_to_be_solve_set, feature_pipeline=moon_lander_feature,
                                objective_value=230)

    # 定义JSON文件路径
    seedpath = r'pop_init.json'

    method = MLES(llm=llm,
                  profiler=MLESProfiler(log_dir='', log_style='complex', run_mode=run_mode,
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

    method.using_flow(worst_case_percent=10, top_k=1)


if __name__ == '__main__':
    testing_paths = [
        r"C:\0_QL_work\014_mmeoh\LLM4AD_MLES\LLM4AD\example\mles_moonlander\logs\MLES\20260208_201339",
    ]

    for path in testing_paths:
        main(path)
