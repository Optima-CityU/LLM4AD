import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.ails2 import Ails2Evaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh_java import EoH_Java
from llm4ad.method.eoh_java import EoH_java_Profiler


def main():
    llm = HttpsApi(host='api.bltcy.ai',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='sk-0hCjhh3wBUP7H2TQF9B6D290Ee604cAc88633dDc5f68B0Ed',  # your key, e.g., sk-abcdefghijklmn
                   model='gpt-4o-mini',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=60)
    response = llm.draw_sample("1+1=?")
    print(response)
    # task = Ails2Evaluation()
    #
    # method = EoH_Java(llm=llm,
    #                   profiler=EoH_java_Profiler(log_dir='logs', log_style='complex'),
    #                   evaluation=task,
    #                   max_sample_nums=48,
    #                   max_generations=5,
    #                   pop_size=8,
    #                   num_samplers=4,
    #                   num_evaluators=8,
    #                   debug_mode=True)
    #
    # method.run()


if __name__ == '__main__':
    main()
