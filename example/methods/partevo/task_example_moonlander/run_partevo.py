import sys
# Add the project root to the system path.
# (Not required if you installed the package via pip, but useful for local development)
sys.path.append('../../../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation, moon_lander_feature
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

    # Directory where evolution logs and generated policies will be saved
    log_dir = f'logs/partevo'  # Use run_id to avoid overwriting logs

    # =========================================================================
    # 2. Environment Configuration (Training & Testing Instances)
    # Define the random seeds to generate distinct initial states for the environment.
    # =========================================================================

    # Training Seeds: A carefully selected diverse set covering various initial conditions to ensure the evolved policy is robust.
    training_seeds = [6, 9, 17, 29, 57,
                      44, 18, 69, 26, 68,
                      65, 23, 51, 93, 16,
                      87, 92, 90, 22, 73,
                      60, 10, 19, 97, 11,
                      14, 99, 98, 8, 28,
                      43, 56, 89, 15, 74]
    instance_set = {id: seed for id, seed in enumerate(training_seeds)}

    # Testing Seeds: Unseen scenarios used purely for evaluating the final policy
    # (Only active if run_mode is set to 'Using' or 'Combined')
    testing_seeds = [i for i in range(100, 150)]
    ins_to_be_solve_set = {id: seed for id, seed in enumerate(testing_seeds)}

    # =========================================================================
    # 3. Task Evaluation Setup
    # Link the environment instances to our custom MoonLander evaluator.
    # =========================================================================

    run_mode = 'Training'  # Options: 'Training' (evolution), 'Using' (testing), 'Combined'
    using_algo_designed_path = ""  # Path to a saved policy if run_mode is 'Using'

    task = MoonLanderEvaluation(whocall='partevo', instance_set=instance_set, run_mode=run_mode,
                                ins_to_be_solve_set=ins_to_be_solve_set, feature_pipeline=moon_lander_feature,
                                objective_value=230  # Target fitness score for success
                                )

    # Initial population file containing base heuristic code to kickstart evolution
    local_algo_base = r'./pop_init.json'

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

    # Start the automated discovery process!
    print(f"Starting MLES on MoonLander. Logs will be saved to: {log_dir}")
    method.run()


if __name__ == '__main__':
    main()
