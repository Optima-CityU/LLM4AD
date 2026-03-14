import sys
# Add the project root to the system path.
# (Not required if you installed the package via pip, but useful for local development)
sys.path.append('../../../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.car_racing import RacingCarEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.mles import MLES
from llm4ad.method.mles import MLESProfiler


def main():
    # =========================================================================
    # 1. LLM Configuration
    # Set up the Large Language Model that will act as our "Algorithm Designer".
    # =========================================================================
    llm = HttpsApi(host='xxxx',         # Replace with your API endpoint (e.g., api.openai.com/v1/completions, api.deepseek.com)
                   key='xxx',           # Replace with your actual API key
                   model='xxx',         # Choose your model (e.g., gpt-4o, deepseek-chat)
                   timeout=120          # Maximum waiting time for LLM response
                   )

    # Directory where evolution logs, generated policies, and visual evidence will be saved
    log_dir = f'logs/MLES'

    # =========================================================================
    # 2. Environment Configuration (Training & Testing Instances)
    # Define the random seeds to generate distinct race tracks.
    # =========================================================================

    # Training Seeds: The tracks used during the evolutionary search.
    # Using a single seed [1] here for a fast baseline experiment.
    training_seeds = [1]
    instance_set = {id: seed for id, seed in enumerate(training_seeds)}

    # Testing Seeds: Unseen tracks used purely for evaluating the final policy
    # (Only active if run_mode is set to 'Using' or 'Combined')
    testing_seeds = [i for i in range(10, 20)]
    ins_to_be_solve_set = {id: seed for id, seed in enumerate(testing_seeds)}

    # =========================================================================
    # 3. Task Evaluation Setup
    # Link the environment instances to our custom Car Racing evaluator.
    # =========================================================================
    run_mode = 'Training'  # Options: 'Training' (evolution), 'Using' (testing), 'Combined'
    using_algo_designed_path = ""   # Path to a saved policy if run_mode is 'Using'

    task = RacingCarEvaluation(whocall='mles',
                               run_mode=run_mode,
                               instance_set=instance_set,
                               ins_to_be_solve_set=ins_to_be_solve_set,
                               objective_value=100)

    # Initial population file containing base heuristic code to kickstart evolution
    seedpath = r'pop_init.json'

    # =========================================================================
    # 4. MLES Algorithm Configuration & Execution
    # =========================================================================
    method = MLES(
        llm=llm,
        profiler=MLESProfiler(
            log_dir=log_dir,
            log_style='complex',
            run_mode=run_mode,
            using_algo_designed_path=using_algo_designed_path
        ),
        evaluation=task,

        # --- Evolutionary Hyperparameters ---
        max_sample_nums=100,  # Total number of policies to sample/evaluate
        max_generations=None,  # Alternative stopping criterion (None = rely on max_sample_nums)
        pop_size=16,  # Number of policies kept in the active population

        # --- System Hyperparameters ---
        num_samplers=8,  # Number of concurrent threads for LLM calls
        num_evaluators=8,  # Number of concurrent threads for environment evaluations
        debug_mode=False,

        # --- Mutation & Crossover Operators ---
        # e1/e2: Exploration, m1_M/m2_M: Visual-feedback-driven mutations
        operators=('e1', 'e2', 'm1_M', 'm2_M'),

        seed_path=seedpath
    )

    # Start the automated discovery process!
    print(f"Starting MLES on Car Racing. Logs will be saved to: {log_dir}")
    method.run()


if __name__ == '__main__':
    main()
