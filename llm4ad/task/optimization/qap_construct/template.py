template_program = '''
import numpy as np
from typing import List

def select_next_assignment(flow_matrix: np.ndarray, distance_matrix: np.ndarray) -> List[int]:
    """
    Constructive heuristic for the Quadratic Assignment Problem (QAP).

    IMPORTANT: This version matches the evaluator's contract in `QAPEvaluation.qap_evaluate`,
    which calls the heuristic as:
        assignment = eva(flow_matrix, distance_matrix)

    So the heuristic must return a COMPLETE assignment (a permutation) in ONE call.

    Args:
        flow_matrix (np.ndarray): shape (n, n)
        distance_matrix (np.ndarray): shape (n, n)

    Returns:
        assignment (List[int]): length-n permutation of {0, ..., n-1}
            where assignment[i] is the location assigned to facility i.
            Must contain no -1 and no duplicates.
    """
    n = flow_matrix.shape[0]

    # Simple baseline: identity permutation (facility i -> location i).
    # This is always feasible and satisfies evaluator checks.
    assignment = list(range(n))
    return assignment


'''

task_description = '''
The task is to assign a set of facilities to a set of locations in such a way that the total cost of interactions between facilities is minimized.
'''
