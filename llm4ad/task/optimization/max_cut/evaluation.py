# SPDX-License-Identifier: BSD-2-Clause
# Module Name: MaxCutEvaluation
# Last Revision: 2026/6.16
# Description: Evaluates the local search heuristic for Max-Cut problem.
#              Given an undirected weighted graph,
#              the goal is to partition nodes into two sets S and T
#              to maximize the total weight of edges crossing between sets.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - n_instance: Number of problem instances to generate: int (default: 16).
#    - n_nodes: Number of nodes in each graph: int (default: 100).
#    - edge_probability: Probability of edge existence between any two nodes: float (default: 0.5).
#    - max_flips: Maximum number of flips per local search run: int (default: n_nodes * 2).
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Licensed under BSD-2-Clause; see the project LICENSE file.
# Academic citation is appreciated, but is not a license condition.
# Suggested reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For voluntary commercial collaboration inquiries, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

from typing import Any
import numpy as np
from llm4ad.base import Evaluation
from llm4ad.task.optimization.max_cut.get_instance import GetData
from llm4ad.task.optimization.max_cut.template import template_program, task_description

__all__ = ['MaxCutEvaluation']


class MaxCutEvaluation(Evaluation):
    """Evaluator for maximum cut problem using local search."""

    def __init__(self,
                 timeout_seconds=30,
                 n_instance=16,
                 n_nodes=100,
                 edge_probability=0.5,
                 max_flips=None,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_instance = n_instance
        self.n_nodes = n_nodes
        self.edge_probability = edge_probability
        self.max_flips = max_flips if max_flips else n_nodes * 2

        getData = GetData(self.n_instance, self.n_nodes, self.edge_probability)
        self._datasets = getData.generate_instances()

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def compute_cut_value(self, adjacency: dict, cut_S: list, cut_T: list) -> float:
        """Compute total weight of edges crossing between S and T."""
        set_T = set(cut_T)
        cut_value = 0.0
        for node in cut_S:
            for neighbor, weight in adjacency.get(node, {}).items():
                if neighbor in set_T:
                    cut_value += weight
        return cut_value

    def compute_flip_gain(self, node: int, adjacency: dict, cut_S_set: set, cut_T_set: set) -> float:
        """Compute the gain of flipping a node (positive means improvement)."""
        gain = 0.0
        if node in cut_S_set:
            for neighbor, weight in adjacency.get(node, {}).items():
                if neighbor in cut_S_set:
                    gain += weight
                elif neighbor in cut_T_set:
                    gain -= weight

        else:
            for neighbor, weight in adjacency.get(node, {}).items():
                if neighbor in cut_T_set:
                    gain += weight
                elif neighbor in cut_S_set:
                    gain -= weight
        return gain

    def get_improving_candidates(self, adjacency: dict, cut_S: list, cut_T: list) -> list:
        """Get all nodes whose flip would improve (or not worsen) the cut."""
        cut_S_set = set(cut_S)
        cut_T_set = set(cut_T)
        candidates = []
        all_nodes = cut_S + cut_T

        for node in all_nodes:
            gain = self.compute_flip_gain(node, adjacency, cut_S_set, cut_T_set)
            if gain >= -1e-9:
                candidates.append(node)
        return candidates

    def evaluate(self, eva: callable) -> float:

        cut_values = np.ones(self.n_instance)
        n_ins = 0

        for adjacency, _ in self._datasets:

            # Random initialization
            all_nodes = list(adjacency.keys())
            np.random.shuffle(all_nodes)
            mid = len(all_nodes) // 2
            cut_S = all_nodes[:mid]
            cut_T = all_nodes[mid:]

            # Local search loop
            for flip_step in range(self.max_flips):

                candidates = self.get_improving_candidates(adjacency, cut_S, cut_T)

                if len(candidates) == 0:
                    break

                try:
                    selected_idx = eva(candidates, cut_S, cut_T, adjacency)
                    selected_idx = int(selected_idx)
                    if selected_idx < 0 or selected_idx >= len(candidates):
                        selected_idx = 0
                except Exception:
                    selected_idx = 0

                selected_node = candidates[selected_idx]

                # Flip the selected node
                if selected_node in cut_S:
                    cut_S.remove(selected_node)
                    cut_T.append(selected_node)
                else:
                    cut_T.remove(selected_node)
                    cut_S.append(selected_node)

            # Compute final cut value
            final_cut = self.compute_cut_value(adjacency, cut_S, cut_T)
            cut_values[n_ins] = final_cut

            n_ins += 1
            if n_ins == self.n_instance:
                break

        ave_cut = np.average(cut_values)
        return ave_cut


if __name__ == '__main__':
    import sys

    print(sys.path)


    def select_node_to_flip(candidates: list, cut_S: list, cut_T: list, adjacency: dict) -> int:
        """
        Design a novel algorithm to select which node to flip in each step of local search.

        Args:
        candidates: List of candidate node IDs that can be flipped.
        cut_S: List of node IDs currently assigned to set S.
        cut_T: List of node IDs currently assigned to set T.
        adjacency: Dictionary where key is node ID, value is a dictionary,
                   {neighbor_id: edge_weight} representing the graph structure.

        Return:
        The index of the selected node from the candidates list.
        """
        best_idx = 0
        best_gain = -float('inf')

        cut_S_set = set(cut_S)
        cut_T_set = set(cut_T)

        for i, node in enumerate(candidates):
            gain = 0.0
            if node in cut_S_set:
                for neighbor, w in adjacency.get(node, {}).items():
                    if neighbor in cut_S_set:
                        gain += w
                    elif neighbor in cut_T_set:
                        gain -= w
            else:
                for neighbor, w in adjacency.get(node, {}).items():
                    if neighbor in cut_T_set:
                        gain += w
                    elif neighbor in cut_S_set:
                        gain -= w
            if gain > best_gain:
                best_gain = gain
                best_idx = i

        return best_idx

    maxcut = MaxCutEvaluation()
    maxcut.evaluate_program('_', select_node_to_flip)
