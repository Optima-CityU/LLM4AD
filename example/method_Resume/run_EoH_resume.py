import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.method.eoh.resume import resume_eoh
from llm4ad.tools.profiler import ProfilerBase


def main():

    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='xxx',  # your llm, e.g., 'gpt-3.5-turbo' 'claude-3-5-sonnet-20240620'
                   timeout=120)

    task = OBPEvaluation()

    method = EoH(llm=llm,
        profiler=EoHProfiler(log_dir='logs/eoh/20251201_130050', log_style='simple', create_random_path=False),
        evaluation=task,
        max_sample_nums=100,
        max_generations=100,
        pop_size=4,
        num_samplers=4,
        num_evaluators=4
        )
    
    resume_eoh(method, path='logs/eoh/20251201_130050')

    method.run()


if __name__ == '__main__':
    main()