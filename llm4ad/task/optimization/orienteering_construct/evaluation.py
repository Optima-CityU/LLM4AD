# SPDX-License-Identifier: BSD-2-Clause
# Module Name: OrienteeringEvaluation
# Description: Evaluates constructive heuristics for the Orienteering Problem.
from __future__ import annotations

from typing import Any

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.orienteering_construct.get_instance import GetData
from llm4ad.task.optimization.orienteering_construct.template import template_program, task_description

__all__ = ['OrienteeringEvaluation']


class OrienteeringEvaluation(Evaluation):
    """Evaluator for constructive heuristics for the Orienteering Problem."""

    def __init__(
            self,
            timeout_seconds=30,
            n_instance=16,
            problem_size=50,
            max_length_ratio=0.35,
            seed=2024,
            **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_instance = int(n_instance)
        self.problem_size = int(problem_size)
        self.max_length_ratio = float(max_length_ratio)
        self.seed = int(seed)
        self._datasets = GetData(
            n_instance=self.n_instance,
            problem_size=self.problem_size,
            max_length_ratio=self.max_length_ratio,
            seed=self.seed,
        ).generate_instances()

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def get_feasible_unvisited_nodes(
            self,
            current_node: int,
            destination_node: int,
            unvisited_nodes: np.ndarray,
            distance_matrix: np.ndarray,
            remaining_budget: float,
    ) -> np.ndarray:
        travel_to_node = distance_matrix[current_node][unvisited_nodes]
        return_to_destination = distance_matrix[unvisited_nodes, destination_node]
        feasible_mask = travel_to_node + return_to_destination <= remaining_budget + 1e-12
        return unvisited_nodes[feasible_mask]

    def construct_solution(self, instance: dict, select_next_node: callable) -> tuple[list[int], float, float] | None:
        distance_matrix = instance["distance_matrix"]
        prizes = instance["prizes"]
        start_node = int(instance["start_node"])
        destination_node = int(instance["end_node"])
        remaining_budget = float(instance["max_length"])

        current_node = start_node
        unvisited_nodes = np.arange(1, len(prizes), dtype=int)
        route = [start_node]
        collected_prize = 0.0

        while len(unvisited_nodes) > 0:
            feasible_nodes = self.get_feasible_unvisited_nodes(
                current_node,
                destination_node,
                unvisited_nodes,
                distance_matrix,
                remaining_budget,
            )
            if len(feasible_nodes) == 0:
                break

            try:
                next_node = select_next_node(
                    current_node,
                    destination_node,
                    feasible_nodes,
                    distance_matrix,
                    prizes,
                    remaining_budget,
                )
                next_node = int(next_node)
            except Exception:
                return None

            if next_node not in feasible_nodes:
                return None

            remaining_budget -= float(distance_matrix[current_node][next_node])
            collected_prize += float(prizes[next_node])
            current_node = next_node
            route.append(current_node)
            unvisited_nodes = unvisited_nodes[unvisited_nodes != next_node]

        remaining_budget -= float(distance_matrix[current_node][destination_node])
        if remaining_budget < -1e-8:
            return None

        if route[-1] != destination_node:
            route.append(destination_node)

        travel_length = float(instance["max_length"] - remaining_budget)
        return route, collected_prize, travel_length

    def evaluate(self, select_next_node: callable) -> float | None:
        collected_prizes = []

        for instance in self._datasets:
            solution = self.construct_solution(instance, select_next_node)
            if solution is None:
                return None
            _, collected_prize, _ = solution
            collected_prizes.append(collected_prize)

        if not collected_prizes:
            return None

        return float(np.mean(collected_prizes))


if __name__ == '__main__':
    def select_next_node(
            current_node: int,
            destination_node: int,
            unvisited_nodes: np.ndarray,
            distance_matrix: np.ndarray,
            prizes: np.ndarray,
            remaining_budget: float) -> int:
        travel_cost = distance_matrix[current_node][unvisited_nodes]
        return_cost = distance_matrix[unvisited_nodes, destination_node]
        scores = prizes[unvisited_nodes] / (travel_cost + return_cost + 1e-12)
        return int(unvisited_nodes[np.argmax(scores)])

    evaluator = OrienteeringEvaluation(n_instance=4, problem_size=20)
    print(evaluator.evaluate(select_next_node))
