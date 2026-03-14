import sys
# Add the project root to the system path.
sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.machine_learning.moon_lander import MoonLanderEvaluation, moon_lander_feature
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.mles import MLES
from llm4ad.method.mles import MLESProfiler


def main(using_algo_designed_path):
    # =========================================================================
    # 1. LLM Configuration
    # Even in testing mode, the MLES framework initializes the LLM object.
    # =========================================================================
    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., api.openai.com/v1/completions, api.deepseek.com
                   key='xxx',  # your key, e.g., sk-abcdefghijklmn
                   model='xxx',  # your llm, e.g., gpt-3.5-turbo, deepseek-chat
                   timeout=120)

    log_dir = ''

    # =========================================================================
    # 2. Environment Configuration (Training & Testing Instances)
    # =========================================================================

    # Training Seeds (Historical Context):
    # We still define the training seeds here to ensure the environment
    # configuration exactly matches what the policy was trained on.

    training_seeds = [6, 9, 17, 29, 57,
             44, 18, 69, 26, 68,
             65, 23, 51, 93, 16,
             87, 92, 90, 22, 73,
             60, 10, 19, 97, 11,
             14, 99, 98, 8, 28,
             43, 56, 89, 15, 74]
    instance_set = {id: seed for id, seed in enumerate(training_seeds)}

    # Testing Seeds (The Real Test):
    # A completely new set of seeds (100-149) that the evolved policy has NEVER seen.
    # This tests the true generalization ability of the programmatic policy.
    testing_seeds = [i for i in range(100, 150)]
    ins_to_be_solve_set = {id: seed for id, seed in enumerate(testing_seeds)}

    # =========================================================================
    # 3. Task Evaluation Setup
    # Notice the run_mode is set to 'Using' instead of 'Training'.
    # =========================================================================
    run_mode = 'Using'  # CRITICAL: Sets framework to evaluation mode
    task = MoonLanderEvaluation(whocall='mles', instance_set=instance_set, run_mode=run_mode,
                                ins_to_be_solve_set=ins_to_be_solve_set, feature_pipeline=moon_lander_feature,
                                objective_value=230)

    seedpath = ''

    # =========================================================================
    # 4. MLES Algorithm Configuration
    # =========================================================================
    method = MLES(llm=llm,
                  profiler=MLESProfiler(log_dir=log_dir, # Can be empty for 'Using' mode if not saving new logs
                                        log_style='complex',
                                        run_mode=run_mode,
                                        using_algo_designed_path=using_algo_designed_path),
                  evaluation=task,
                  max_sample_nums=100,
                  max_generations=None,
                  pop_size=16,
                  num_samplers=8,
                  num_evaluators=8,
                  debug_mode=False,
                  operators=('e1', 'e2', 'm1_M', 'm2_M'),  # ('e1', 'e2', 'm1_M', 'm2_M')
                  seed_path=seedpath
                  )

    # =========================================================================
    # 5. Execute Testing Flow
    # using_flow() specifically evaluates the loaded policy on the testing set.
    # worst_case_percent=10: Analyzes the bottom 10% of failure cases.
    # top_k=1: Selects the absolute best algorithm from the loaded path to test.
    # =========================================================================
    method.using_flow(worst_case_percent=10, top_k=1)


if __name__ == '__main__':
    # =========================================================================
    # 🚀 LOAD YOUR TRAINED POLICIES HERE
    # Replace the path below with the actual log directory generated during Training.
    # Example format: "logs/MLES_gemini/YYYYMMDD_HHMMSS"
    # =========================================================================
    testing_paths = [
        r"C:\0_QL_work\04_PartEvo_llm4ad\LLM4AD_260308_PartEvo\LLM4AD\example\methods\mles\mles_Moonlander\logs\MLES\20260308_220334",     # <-- Update this path to your specific run!
    ]

    for path in testing_paths:
        main(path)
