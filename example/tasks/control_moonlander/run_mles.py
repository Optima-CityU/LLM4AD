import sys

# Add the project root to the system path.
# (Not required if you installed the package via pip, but useful for local development)
sys.path.append('../../../../')

from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation, moon_lander_feature
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.mles import MLES
from llm4ad.method.mles import MLESProfiler


def main():
    # =========================================================================
    # 1. LLM Configuration
    # Set up the Large Language Model that will act as our "Algorithm Designer".
    # =========================================================================
    llm = HttpsApi(host='xxx',      # Replace with your API endpoint (e.g., api.openai.com/v1/completions, api.deepseek.com)
                   key='xxx',  # Replace with your actual API key
                   model='xxxx',  # Choose your model (e.g., gpt-4o, deepseek-chat)
                   timeout=120  # Maximum waiting time for LLM response
                   )

    # Directory where evolution logs and generated policies will be saved
    log_dir = f'logs/MLES'

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

    task = MoonLanderEvaluation(whocall='mles', instance_set=instance_set, run_mode=run_mode,
                                ins_to_be_solve_set=ins_to_be_solve_set, feature_pipeline=moon_lander_feature,
                                objective_value=230  # Target fitness score for success
                                )

    # Initial population file containing base heuristic code to kickstart evolution
    seedpath = r'./pop_init.json'

    # =========================================================================
    # 4. MLES Algorithm Configuration & Execution
    # =========================================================================
    method = MLES(llm=llm,
                  profiler=MLESProfiler(log_dir=log_dir, log_style='complex', run_mode=run_mode,
                                        using_algo_designed_path=using_algo_designed_path),
                  evaluation=task,

                  # --- Evolutionary Hyperparameters ---
                  max_sample_nums=100,      # Total number of policies to sample/evaluate
                  max_generations=None,     # Alternative stopping criterion (None = rely on max_sample_nums)
                  pop_size=8,              # Number of policies kept in the active population

                  # --- System Hyperparameters ---
                  num_samplers=4,           # Number of concurrent threads for LLM calls
                  num_evaluators=4,         # Number of concurrent threads for environment evaluations
                  debug_mode=False,

                  operators=('e1', 'e2', 'm1_M', 'm2_M'),  # ('e1', 'e2', 'm1_M', 'm2_M')

                  seed_path=seedpath
                  )

    # Start the automated discovery process!
    print(f"Starting MLES on MoonLander. Logs will be saved to: {log_dir}")
    method.run()


if __name__ == '__main__':
    main()
