import sys

sys.path.append('../../../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation, moon_lander_feature
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.partevo import PartEvo
from llm4ad.method.partevo import PartEvoProfiler
from llm4ad.tools.profiler import ProfilerBase


def main():
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-qMAtcWpKnF64zZxWqyLcqXRQYEtwnyiriaB0nR5GBldQ7S0A',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat, gpt-4o-mini
                   timeout=120)
    log_dir = f'logs/partevo'  # Use run_id to avoid overwriting logs

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
    using_algo_designed_path = ""
    # Using_seeds = [i for i in range(20, 60)]
    Using_seeds = [i for i in range(100, 150)]
    # Using_seeds = seeds
    ins_to_be_solve_set = {}
    for id, seed in enumerate(Using_seeds):
        ins_to_be_solve_set[id] = seed

    run_mode = 'Training'  # Training, Using, Combined
    task = MoonLanderEvaluation(whocall='partevo', instance_set=instance_set, run_mode=run_mode,
                                ins_to_be_solve_set=ins_to_be_solve_set, feature_pipeline=moon_lander_feature,
                                objective_value=230)

    method = PartEvo(llm=llm,
                     profiler=PartEvoProfiler(log_dir=log_dir, log_style='simple', run_mode=run_mode,
                                              using_algo_designed_path=using_algo_designed_path),
                     evaluation=task,
                     max_sample_nums=500,
                     max_generations=None,
                     pop_size=16,
                     operators=('re', 'se', 'cn', 'lge'),   # ('re', 'se', 'cn', 'lge'),
                     num_samplers=4,
                     num_evaluators=4,
                     partition_method='kmeans',
                     partition_number=4,
                     local_algo_base='./pop_init_test.json',
                     feature_used=('ast',),
                     debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
