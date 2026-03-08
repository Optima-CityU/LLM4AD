template_program = '''
from typing import Sequence, Tuple, TypedDict, TypeAlias

JobId: TypeAlias = int
MachineId: TypeAlias = int
Time: TypeAlias = int
ProcessingTime: TypeAlias = int

Operation: TypeAlias = Tuple[JobId, MachineId, ProcessingTime]  # (job_id, machine_id, proc_time)


def determine_next_operation(
    current_status: CurrentStatus,
    feasible_operations: Sequence[Operation],
) -> Operation:
    """
    Choose one operation from `feasible_operations` to schedule next.

    Input contract (must match the evaluation block)
    -----------------------------------------------
    current_status:
        Dict-like object with exactly these keys:
        - 'machine_status': list[int] in the evaluator (Sequence[int] here)
        - 'job_status': list[int] in the evaluator (Sequence[int] here)

    feasible_operations:
        A (usually non-empty) list of operations, where each operation is:
        (job_id: int, machine_id: int, processing_time: int)

        These tuples are created in the evaluator as:
        all_operations.append((job_id, machine_id, processing_times[job_id][machine_id]))

    Output contract (must match the evaluation block)
    -------------------------------------------------
    Returns:
        One `Operation` tuple of the same form:
        (job_id: int, machine_id: int, processing_time: int)

        IMPORTANT: It should be one of the tuples from `feasible_operations`
        (the evaluator does `all_operations.remove(next_operation)`).

    """
    # Example baseline heuristic: shortest processing time
    return min(feasible_operations, key=lambda op: op[2])
'''

task_description = '''
Given jobs and machines, schedule jobs on machines to minimize the total makespan. Design an algorithm to select the next operation in each step.
'''
