from os import getenv
from typing import Callable

from llamea import LLaMEA
from llamea.llm import Gemini_LLM
from llamea.solution import Solution
from llamea.utils import prepare_namespace, clean_local_namespace
from llm4ad.task.optimization.cvrp_construct import CVRPEvaluation
from llm4ad.task.optimization.cvrp_construct.template import task_description, template_program


import inspect

task = CVRPEvaluation()

def evaluate_wrapper(solution: Solution, explogger=None, evaluator : Callable = task.evaluate) -> Solution:
    """
        LLaMEA anad llm4ad evaluate functions differently, this function 
        serves as an wrapper to help evaluate the functions properly.
    
    Args:
        `solution: llamea.Solution`: LLaMEA comes with a `Solution` object that have all
        the arguements necessary for LLaMEA to track it as an individual in population.

        `evaulator: Callable` here is a CVRPEvaluation.evaluate, that takes in a 
        callable function, and returns its score as float.
    
    Returns:
        `Solution` object with updated score.
    """
    code = solution.code
    possible_issue = None
    local_ns = {}
    try:
        global_ns, possible_issue = prepare_namespace(code, allowed=['pandas', 'scipy', 'numpy'])
        exec(code, global_ns, local_ns)
        local_ns = clean_local_namespace(local_ns, global_ns)

    except Exception as e:
        solution.set_scores(
            float("inf"),
            (possible_issue if possible_issue else "") + f". Exec block failed to execute.",
            e
        )
        return solution
    executable = local_ns[solution.name]
    try:
        score = evaluator(executable)
        solution.set_scores(
            score,
            f"The average distance of this heursitic is {score}.",
            None
        )
        return solution
    except Exception as e:
        solution.set_scores(
            float("inf"),
            f"Code failed to execute {e}: {inspect.getsource(executable)}",
            e
        )
        return solution


def main(): 
    api_key = getenv("GOOGLE_API_KEY")
    llm = Gemini_LLM(
        api_key,
        "gemini-2.5-flash"
    )

    # logger = ExperimentLogger("cvrp_construct")
    role_prompt = '''You are a highly skilled computer scientist in the field of natural computing. 
    Your task is to design novel metaheuristic algorithms to solve CVRP Problem.'''


    method = LLaMEA(
        f=evaluate_wrapper,
        llm=llm,
        n_parents = 1,
        n_offspring = 1,
        role_prompt=role_prompt,
        task_prompt=task_description,
        example_prompt=template_program,
        minimization=False,
        budget=10                                 #Test Case: small number of iterations.
    )

    method.run()

if __name__ == "__main__":
    main()