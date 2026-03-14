from llm4ad.task.optimization.online_bin_packing_2O import OBP_2O_Evaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.meoh import MEoH, MEoHProfiler


def main():
    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='xxx',  # your llm, e.g., 'gpt-3.5-turbo' 'claude-3-5-sonnet-20240620'
                   timeout=120)

    task = OBP_2O_Evaluation()

    method = MEoH(
        llm=llm,
        profiler=MEoHProfiler(
            log_dir='logs/meoh',
            num_objs=2,
            log_style='simple'),
        evaluation=task,
        max_sample_nums=1000,
        max_generations=1000,
        pop_size=10,
        num_samplers=4,
        num_evaluators=4,
        num_objs=2,
        debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
