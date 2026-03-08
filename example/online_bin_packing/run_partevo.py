import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.partevo import PartEvo
from llm4ad.method.partevo import PartEvoProfiler
from llm4ad.tools.profiler import ProfilerBase


def main():
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-qMAtcWpKnF64zZxWqyLcqXRQYEtwnyiriaB0nR5GBldQ7S0A',  # your key, e.g., sk-abcdefghijklmn
                   model='gemini-2.5-flash',  # your llm, e.g., gpt-3.5-turbo, gpt-4o-mini, deepseek-v3.2, qwen3.5-plus, gemini-2.5-flash
                   timeout=240)

    task = OBPEvaluation()

    log_dir = f'logs/partevo_gemini'  # Use run_id to avoid overwriting logs

    run_mode = 'Training'  # Training, Using, Combined

    method = PartEvo(llm=llm,
                     profiler=PartEvoProfiler(log_dir=log_dir, log_style='simple', run_mode=run_mode,
                                              using_algo_designed_path=''),
                     evaluation=task,
                     max_sample_nums=500,
                     max_generations=None,
                     pop_size=8,
                     operators=('re', 'se', 'cn', 'lge'),   # ('re', 'se', 'cn', 'lge'),
                     num_samplers=8,
                     num_evaluators=8,
                     partition_method='kmeans',
                     partition_number=4,
                     local_algo_base='',
                     feature_used=('ast',),
                     debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
