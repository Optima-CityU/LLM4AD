from llamea import Solution, prepare_namespace
from llm4ad.base.evaluate import Evaluation

def generate_evaluator(for_instance: Evaluation):
    """A LLaMEA instance works on llamea.Solution object, this generator
    takes the instance of evaluation, that have evaluate member mapping 
    Callable -> float, and returns a function that takes that `float` value 
    to update the `Solution` with appropriate fitness.
    """

    def evaluator(solution: Solution, explogger=None) -> Solution:
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
            global_ns, possible_issue = prepare_namespace(code, allowed=['pandas', 'numpy', 'numbas'])
            exec(code, global_ns, local_ns)

        except Exception as e:
            solution.set_scores(
                float("-inf"),  # Always maximisation problem in llm4ad.
                (possible_issue if possible_issue else "") + f". Exec block failed to execute.",
                e
            )
            return solution
        executable = local_ns[solution.name]
        try:
            score = for_instance.evaluate(executable)
            solution.set_scores(
                score,
                f"The average distance of this heursitic is {score}.",
                None
            )
            return solution
        except Exception as e:
            solution.set_scores(
                float("-inf"),
                f"Code failed to execute {e}.",
                e
            )
            return solution
    return evaluator