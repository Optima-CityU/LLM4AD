import sys
# Add the project root to the system path.
# (Not required if you installed the package via pip, but useful for local development)
sys.path.append('../../../../')  # This is for finding all the modules

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.partevo import PartEvo
from llm4ad.method.partevo import PartEvoProfiler

def main():
    # =========================================================================
    # 1. LLM Configuration
    # Set up the Large Language Model that will act as our "Algorithm Designer".
    # =========================================================================
    llm = HttpsApi(host='xxxx',
                   # Replace with your API endpoint (e.g., api.openai.com/v1/completions, api.deepseek.com)
                   key='xxx',  # Replace with your actual API key
                   model='xxx',  # Choose your model (e.g., gpt-4o, deepseek-chat)
                   timeout=120  # Maximum waiting time for LLM response
                   )

    # =========================================================================
    # 2. Task Evaluation Setup
    # Link the environment instances to our custom MoonLander evaluator.
    # =========================================================================
    run_mode = 'Training'  # Options: 'Training' (evolution), 'Using' (testing), 'Combined'
    using_algo_designed_path = ""  # Path to a saved policy if run_mode is 'Using'
    task = OBPEvaluation()

    # Directory where evolution logs and generated policies will be saved
    log_dir = f'logs/partevo'  # Use run_id to avoid overwriting logs

    # Initial population file containing base heuristic code to kickstart evolution
    local_algo_base = ''

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
                     local_algo_base=local_algo_base,
                     feature_used=('ast',),
                     debug_mode=False)

    method.run()


if __name__ == '__main__':
    main()
