# Module Name: PartEvo
# Last Revision: 2026/3/8
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Qinglong Hu and Qingfu Zhang.
#       "Partition to evolve: Niching-enhanced evolution with llms for automated algorithm discovery."
#       In Thirty-ninth Annual Conference on Neural Information Processing Systems (NeurIPS). 2025.
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple, Literal, Optional, Dict, Any
from llm4ad.base import Function
from .base import Evoind
from threading import RLock
import itertools
import random


class ClusterUnit:
    """
    An Algorithm Pool Unit that manages a sub-population of a specific algorithmic branch (niche).

    Responsibilities:
    - Performs parent selection for the specific cluster using various strategies.
    - Manages the local population including registration, deduplication, and survival of the fittest.
    - Tracks stagnation and improvement metrics for the niche.
    """

    # Mapping of evolutionary operators to their default selection strategies
    OPERATOR_SELECTION_MAP = {
        're': 'tournament',
        'se': 'tournament',
        'cc': 'tournament',
        'lge': 'random'
    }

    def __init__(self, cluster_id: int,
                 max_pop_size: int,
                 intra_operators: Tuple[str, ...],  # ('re', 'se', 'cc', 'lge')
                 intra_operators_parent_num: Dict[str, int],
                 pop: List[Evoind] | ClusterUnit | None = None,
                 ):
        """
        Initializes the ClusterUnit (Niche).

        Args:
            cluster_id: Unique identifier for this cluster/niche.
            max_pop_size: Maximum number of individuals allowed in the sub-population.
            intra_operators: Tuple of operator names (e.g., 're', 'se', 'cc', 'lge').
            intra_operators_parent_num: Dictionary mapping operators to required parent counts.
            pop: Initial population or existing ClusterUnit to clone from.
        """
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop.copy()
        elif hasattr(pop, '_population'):
            self._population = pop._population.copy()
        else:
            self._population = []

        self.cluster_id = cluster_id
        self._max_pop_size = max_pop_size

        self.cumulative_Non_improvement_count = 0
        self._history_best_score = float('-inf')

        self._lock = RLock()

        self.intra_operators = intra_operators
        # Default requirement: cc needs 1 internal parent (the other is provided externally).
        self.intra_operators_parent_num = intra_operators_parent_num or {'re': 1, 'se': 1, 'cc': 1, 'lge': 1}
        self._intra_operators_iterator = itertools.cycle(self.intra_operators)

    def __len__(self):
        with self._lock: return len(self._population)

    def __getitem__(self, item) -> Evoind:
        with self._lock: return self._population[item]

    def __setitem__(self, key, value):
        with self._lock: self._population[key] = value

    @property
    def population(self):
        """Returns a thread-safe copy of the current sub-population list."""
        with self._lock: return self._population.copy()

    def _calculate_dynamic_scores(self, population: List[Evoind],
                                  help_inter: bool,
                                  target_instances: Optional[List[Any]]) -> List[Tuple[Evoind, float]]:
        """
        Calculates fitness scores for selection.

        Strategy:
        1. Default: Use the global individual score (ind.function.score).
        2. Specialized (Inter-cluster Help): If this cluster is acting as a helper for specific
           instances, calculate a targeted score based on performance on target_instances.
        """
        candidates_with_score = []
        for ind in population:
            # 1. Default Baseline: Overall score
            eff_score = ind.function.score

            # 2. Targeted Logic: Override if specific instances are targeted by a requester
            if help_inter and target_instances:
                score_sum = 0
                valid_cnt = 0
                if ind.function.all_ins_performance:
                    for ins_id in target_instances:
                        perf = ind.function.all_ins_performance.get(ins_id)
                        if perf and perf.get('score') is not None:
                            s = perf['score']
                            if s != float('-inf'):
                                score_sum += s
                                valid_cnt += 1

                # If target instances are specified but individual hasn't run them, penalize
                eff_score = score_sum if valid_cnt > 0 else float('-inf')

            if eff_score is not None and not math.isinf(eff_score):
                candidates_with_score.append((ind, eff_score))

        return candidates_with_score

    def selection(self, existing_functions: Optional[List[Function]] = None,
                  best_must: bool = False,
                  mode: str = None,
                  help_inter: bool = False,
                  help_number: int = 1,
                  tournament_k: int = 3,
                  target_instances: Optional[List[Any]] = None
                  ) -> Tuple[List[Function], str | None, bool]:
        """
        Unified selection interface for the clusterUnit.

        Args:
            existing_functions: List of functions to exclude from selection.
            best_must: If True, ensures the current cluster-best is included in selection.
            mode: Selection strategy ('top', 'tournament', 'random', 'exp', etc.).
            help_inter: Set to True if this cluster is providing individuals for another cluster.
            help_number: Number of individuals to provide in help mode.
            target_instances: Specific problem instances to prioritize performance on.

        Returns:
            Tuple: (selected_parent_functions, current_operator_name, need_external_collaboration)
        """
        number = 0
        current_operator = None
        need_external_help = False

        with self._lock:
            # --- Step 0: Determine Role and Operator ---
            if help_inter:
                # [Helper Mode]: Provide individuals without evolving local population
                number = help_number
                current_operator = None
                need_external_help = False
                if mode is None:
                    mode = 'tournament'
            else:
                # [Requester Mode]: Evolve locally, cycle through operators
                current_operator = next(self._intra_operators_iterator)
                number = self.intra_operators_parent_num.get(current_operator, 1)

                # Determine if the operator requires external collaboration (e.g., Crossover)
                if current_operator == 'cc':
                    need_external_help = True
                elif current_operator == 'lge':
                    need_external_help = True

                # Dynamic Strategy Selection based on operator
                if mode is None:
                    mode = self.OPERATOR_SELECTION_MAP.get(current_operator, 'tournament')

            # --- Step 1: Population Filtering ---
            full_population = self._population.copy()
            valid_pop = full_population

            if existing_functions:
                existing_ids = {id(func) for func in existing_functions}
                valid_pop = [ind for ind in full_population if id(ind.function) not in existing_ids]

            # Fallback: if filtering leaves no one, revert to full population
            if not valid_pop and full_population:
                valid_pop = full_population

            if not valid_pop:
                return [], current_operator, False

            # --- Step 2: Fitness Calculation ---
            candidates_with_score = self._calculate_dynamic_scores(valid_pop, help_inter, target_instances)

            # Fallback for insufficient candidates
            if len(candidates_with_score) < number:
                candidates_with_score = self._calculate_dynamic_scores(full_population, help_inter, target_instances)
                if len(candidates_with_score) < number:
                    return [], current_operator, False

            # --- Step 3: Execute Selection Strategy ---
            selected_individuals = []

            # A. Top-N (Greedy)
            if mode == 'top':
                candidates_with_score.sort(key=lambda x: x[1], reverse=True)
                selected_individuals = [x[0] for x in candidates_with_score[:number]]

            # B. Tournament Selection
            elif mode == 'tournament':
                pool_indices = list(range(len(candidates_with_score)))
                for _ in range(number):
                    if not pool_indices: break
                    k = min(len(pool_indices), tournament_k)
                    chosen_indices = np.random.choice(pool_indices, size=k, replace=False)
                    best_idx = max(chosen_indices, key=lambda i: candidates_with_score[i][1])
                    selected_individuals.append(candidates_with_score[best_idx][0])
                    pool_indices.remove(best_idx)

            # C. Uniform Random (High Diversity)
            elif mode == 'random':
                if len(candidates_with_score) >= number:
                    chosen_tuples = random.sample(candidates_with_score, number)
                    selected_individuals = [t[0] for t in chosen_tuples]
                else:
                    selected_individuals = [t[0] for t in candidates_with_score]

            # D. Probabilistic Sampling (Roulette/Exponential)
            elif mode in ['roulette', 'linear', 'exp']:
                candidates_with_score.sort(key=lambda x: x[1], reverse=True)
                sorted_inds = [x[0] for x in candidates_with_score]

                if mode == 'exp':
                    p = np.exp(-np.arange(len(sorted_inds)) / 2.0)
                elif mode == 'linear':
                    weights = np.arange(len(sorted_inds), 0, -1)
                    p = weights
                else:
                    # Softmax to handle negative scores
                    raw_scores = np.array([x[1] for x in candidates_with_score])
                    p = np.exp(raw_scores - np.max(raw_scores))

                probabilities = p / np.sum(p)
                try:
                    selected_individuals = list(
                        np.random.choice(sorted_inds, size=number, p=probabilities, replace=False))
                except ValueError:
                    selected_individuals = sorted_inds[:number]

            else:
                # Default to Top-N
                candidates_with_score.sort(key=lambda x: x[1], reverse=True)
                selected_individuals = [x[0] for x in candidates_with_score[:number]]

            # --- Step 4: Elitism Enforcement ---
            if best_must and len(selected_individuals) > 0 and len(valid_pop) > 0:
                global_best = max(valid_pop, key=lambda ind: ind.function.score)
                if not any(ind is global_best for ind in selected_individuals):
                    selected_individuals[-1] = global_best
        return [ind.function for ind in selected_individuals], current_operator, need_external_help

    def has_duplicate_function(self, func: str | Evoind) -> bool:
        """Checks if an individual with identical code already exists in the cluster."""
        target_code = func.function.body if isinstance(func, Evoind) else str(func)
        for ind in self._population:
            if ind.function.body == target_code:
                return True
        return False

    def register_individual(self, new_individual: Evoind):
        """
        Registers a new individual into the population.
        Implements 'Identical Score Replacement' to maintain niche diversity.
        """
        with self._lock:
            new_score = new_individual.function.score
            if new_score is None or math.isinf(new_score):
                return False

            # --- 1. Deduplication and Replacement Logic ---
            replaced = False
            for i, ind in enumerate(self._population):
                is_same_code = (ind.function.body == new_individual.function.body)
                is_same_score = (abs(ind.function.score - new_score) < 1e-9)

                if is_same_code or is_same_score:
                    # Only replace if the new individual is better or equal (rejuvenation)
                    if new_score >= ind.function.score:
                        self._population[i] = new_individual
                        replaced = True
                        if new_score > self._history_best_score + 1e-6:
                            self._history_best_score = new_score
                            self.cumulative_Non_improvement_count = 0
                        break
                    else:
                        return False

            # --- 2. Addition of New Blood ---
            if not replaced:
                if new_score > self._history_best_score + 1e-6:
                    self._history_best_score = new_score
                    self.cumulative_Non_improvement_count = 0
                else:
                    self.cumulative_Non_improvement_count += 1

                self._population.append(new_individual)

            # --- 3. Population Size Control ---
            if len(self._population) > self._max_pop_size:
                self.do_pop_management()

            return True

    def do_pop_management(self):
        """
        Cleans the population by removing duplicates and truncating to max size.
        Survival is based strictly on score.
        """
        if not self._population:
            return

        # Final deduplication based on function body
        unique_map: Dict[str, Evoind] = {}
        for ind in self._population:
            code_key = ind.function.body
            if code_key not in unique_map or ind.function.score >= unique_map[code_key].function.score:
                unique_map[code_key] = ind

        # Sort by score descending and truncate
        sorted_pop = list(unique_map.values())
        sorted_pop.sort(key=lambda ind: ind.function.score, reverse=True)
        self._population = sorted_pop[:self._max_pop_size]

    def get_best_individual(self) -> Evoind | None:
        """Returns the highest-scoring individual in the Niche."""
        with self._lock:
            if not self._population: return None
            return max(self._population, key=lambda ind: ind.function.score)