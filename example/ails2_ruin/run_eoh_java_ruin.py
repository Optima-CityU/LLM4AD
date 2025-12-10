import sys

sys.path.append('../../')  # This is for finding all the modules

from evaluation import Ails2Evaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh_java import EoH_Java
from llm4ad.method.eoh_java import EoH_java_Profiler
import os
import json
from llm4ad.base import JavaScripts

def main():
    llm = HttpsApi(host='stable.gptbest.vip:9088',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-dKMuUHsISnfTuYGPDb78437190Db4bC19f968bC796260230',  # your key, e.g., sk-abcdefghijklmn
                   model='gemini-3-pro-preview',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=60)

    task = Ails2Evaluation(max_parallel_instances=4,instances_subdir="XLDemo_eohtest")

    method = EoH_Java(llm=llm,
                      profiler=EoH_java_Profiler(log_dir='logs', log_style='complex'),
                      evaluation=task,
                      max_sample_nums=20,
                      max_generations=None,
                      pop_size=4,
                      num_samplers=4,
                      num_evaluators=4,
                      debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
