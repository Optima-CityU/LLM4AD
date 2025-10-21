import sys
from os import getenv, getcwd

from llamea import Gemini_LLM
from llm4ad.method.llamea import LLaMEA

from llm4ad.task.optimization.cvrp_construct import CVRPEvaluation
from llm4ad.task.optimization.cvrp_construct.template import task_description, template_program


def main(): 
    api_key = getenv("GOOGLE_API_KEY")
    llm = Gemini_LLM(
        api_key,
        "gemini-2.5-flash"
    )
    task = CVRPEvaluation()

    # logger = ExperimentLogger("cvrp_construct")
    role_prompt = '''You are a highly skilled computer scientist in the field of natural computing. 
    Your task is to design novel metaheuristic algorithms to solve CVRP Problem.'''


    method = LLaMEA(
        llm=llm,
        evaluator=task,
        n_parents = 1,
        n_offsprings = 1,
        role_prompt=role_prompt,
        task_prompt=task_description,
        example_prompt=template_program,
        minimization=False,
        iterations=10                                 #Test Case: small number of iterations.
    )

    method.run()

if __name__ == "__main__":
    main()