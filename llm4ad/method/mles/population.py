# Module Name: MLES
# Last Revision: 2026/2/9
# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Reference:
#   - Qinglong Hu, Xialiang Tong, Mingxuan Yuan, Fei Liu, Zhichao Lu, and Qingfu Zhang.
#       "Multimodal LLM-assisted Evolutionary Search for Programmatic Control Policies."
#       The Fourteenth International Conference on Learning Representations (ICLR). 2026.

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
from threading import Lock
from typing import List
import numpy as np

from ...base import *


class Population:
    """
    Manages a collection of designed algorithms.
    Handles individual registration, duplicate detection, environmental selection,
    and rank-based parent selection for evolution.
    """

    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        """
        Initialize the population container.
        :param pop_size: Maximum number of individuals allowed in the active population.
        :param generation: Initial generation counter.
        :param pop: Optional initial set of individuals.
        """
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()     # Ensures thread-safe registration during parallel sampling
        self._next_gen_pop = []     # Buffer for individuals waiting for environmental selection
        self._generation = generation
        self._pop_register_number = 1      # Unique global ID for each sampled individual

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        """Returns the current active individuals (the 'survivors')."""
        return self._population

    @property
    def generation(self):
        """Returns the current evolutionary generation count."""
        return self._generation

    def register_function(self, func: Function):
        """
        Registers a new individual into the population.
        Triggers environmental selection once the buffer reaches pop_size.
        """
        # Phase 0: Validation - Only valid functions allowed in the initial pool
        if self._generation == 0 and func.score is None:
            return

        # Default score for invalid/failed evaluations
        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            # Phase 1: Quality Control - Penalize duplicates to maintain diversity
            if self.has_duplicate_function(func):
                func.score = float('-inf')
            func.pop_register_number = self._pop_register_number
            self._pop_register_number += 1
            # Phase 2: Buffering
            self._next_gen_pop.append(func)
            # Phase 3: Environmental Selection (Survival of the Fittest)
            # Occurs when the candidate buffer is full
            if len(self._next_gen_pop) >= self._pop_size:
                pop = self._population + self._next_gen_pop
                # Sort all candidates by score and keep the top-N individuals
                pop = sorted(pop, key=lambda f: f.score, reverse=True)
                self._population = pop[:self._pop_size]
                # Reset buffer and increment generation
                self._next_gen_pop = []
                self._generation += 1
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        """Checks for redundancy based on code string equality or identical fitness scores."""
        # Check against current population
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        # Check against the current buffer
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    def selection(self, number=1, best_must=False, mode='exp', pressure=0.5) -> List[Function]:
        """
        Selects parents for the next evolutionary step using rank-based probability distributions.

        :param number: Number of individuals to select.
        :param best_must: If True, ensures the global best is included in the output.
        :param mode: Probability distribution mode ('exp' for exponential, 'linear', or uniform).
        :param pressure: Selection pressure for 'exp' mode (higher = more likely to pick top individuals).
        """
        # Filter out individuals with invalid scores (-inf)
        valid_funcs = [f for f in self._population if not math.isinf(f.score)]
        if not valid_funcs:
            return []
        # Rank individuals: Index 0 is the best performing
        sorted_funcs = sorted(valid_funcs, key=lambda f: f.score, reverse=True)
        n = len(sorted_funcs)

        ranks = np.arange(n)
        # Define selection probabilities (P) based on rank
        if mode == 'exp':
            p = np.exp(-pressure * ranks)
        elif mode == 'linear':
            p = 1.0 / (ranks + n)
        else:
            p = np.ones(n)  # Uniform sampling

        # Normalize probabilities to sum to 1.0
        p = p / np.sum(p)

        # Handle cases where requested sample size exceeds available individuals
        use_replace = False
        if number > n:
            use_replace = True

        # Perform weighted random sampling
        selected = list(np.random.choice(sorted_funcs, size=number, p=p, replace=use_replace))

        # Elitism Strategy: Force inclusion of the best individual if required
        if best_must:
            best_ind = sorted_funcs[0]
            if best_ind not in selected:
                selected[-1] = best_ind

        return selected
