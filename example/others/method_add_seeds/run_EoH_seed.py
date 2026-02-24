from __future__ import annotations

import sys

sys.path.append('../../')  # This is for finding all the modules

from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler
from llm4ad.tools.profiler import ProfilerBase, TensorboardProfiler
from llm4ad.base import TextFunctionProgramConverter as tfpc

seed1 = '''
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    penalty = np.arange(len(bins), 0, -1)
    scores = bins / (bins - item) - penalty
    max_capacity_bins = np.where(bins == bins.max())[0]
    for idx in max_capacity_bins:
        scores[idx] = -np.inf
    return scores
    '''

seed2 = '''
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    scores = -bins
    return scores
    '''

if __name__ == '__main__':

    llm = HttpsApi(host='xxx',  # your host endpoint, e.g., 'api.openai.com', 'api.deepseek.com'
                   key='xxx',  # your key, e.g., 'sk-abcdefghijklmn'
                   model='xxx',  # your llm, e.g., 'gpt-3.5-turbo' 'claude-3-5-sonnet-20240620'
                   timeout=120)
    
    task = OBPEvaluation()
    method = EoH(
        llm=llm,
        profiler=EoHProfiler(log_dir='logs/eoh', log_style='simple'),
        evaluation=task,
        max_sample_nums=100,
        max_generations=100,
        pop_size=4,
        num_samplers=4,
        num_evaluators=4,
        resume_mode=True
    )

    #  seed
    seeds = [seed1, seed2]
    
    # thoughts for seed（EoH uses both thought and code）
    algorithm_ideas = [
        'the ideas behind seed1',
        'the ideas behind seed2'
    ]
    
    # get population
    pop = method._population
    profiler = method._profiler

    # For each seed function: 
    # 1) evaluate
    # 2) add to population
    # 3) add to profiler
    for seed, algo in zip(seeds, algorithm_ideas):
        # evaluate using evaluator
        score, eval_time = method._evaluator.evaluate_program_record_time(program=seed)

        # seed to function
        seed = tfpc.text_to_function(seed)

        # add time, score and thought to seed
        seed.evaluate_time = eval_time
        seed.score = score
        seed.algorithm = algo

        # add seed to population 
        pop._population.append(seed)

        # add seed to profiler 
        profiler.register_function(seed)

    method.run()
