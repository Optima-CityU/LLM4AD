from llamea import LLaMEA as LLaMEA_Algorithm
from ...base import LLM

from .evaluation import generate_evaluator
from .sampler import LLaMEASampler

from llm4ad.base import Evaluation

class LLaMEA(LLaMEA_Algorithm):
    def __init__(
            self,
            llm : LLM,
            evaluator: Evaluation,
            iterations : int = 50,
            n_parents: int = 5,
            n_offsprings: int=5,
            role_prompt : str = "",
            task_prompt: str = "",
            example_prompt :str | None = None,
            minimization: bool = False,
            elitism: bool = True,
            **kwargs
    ):
        """
        Args:
            evaluation_function: A function for scoring the fitness of the llm generated heuristic.
            llm: An instance of llamea.LLM one of the llms in Ollama, OpenAI, Gemini, DeepSeek,
            evaluation: An instance of llm4ad.base.Evaluation, which defines the way to calculate the score of a generated function.
            iterations: Iteration Count for evolution process,
            n_parents: Number of individuals in parent population (λ),
            n_offsprings: Number of individuals in offspring population (µ),
            role_prompt: LLM role prompt like: "You are an excellent scientific programmer tasked to solve the problem of GVRP.",
            task_prompt: Task prompt is llm4ad.tasks.*.template.task_description for solving a problem,
            example_prompt: Example propmt is llm4ad.tasks.*.template.template_program for solving a problem,
            minimisation: Flag to define direction of optimality.
            elitism: A bool flag to run algorithm in (λ + µ) if set True, else (λ , µ).
        """    
        evaluation_function = generate_evaluator(evaluator)
        super().__init__(
            f=evaluation_function,
            llm=llm,
            budget=iterations,
            n_offspring=n_offsprings,
            n_parents=n_parents,
            role_prompt=role_prompt,
            task_prompt=task_prompt,
            example_prompt=example_prompt,
            minimization=minimization,
            elitism=elitism,
            **kwargs
        )
        
        self.evaluator = evaluator
        self.sampler = LLaMEASampler(llm)