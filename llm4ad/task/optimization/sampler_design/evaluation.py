# SPDX-License-Identifier: BSD-2-Clause
# Module Name: SamplerDesignEvaluation
# Last Revision: 2026/11/17
# Description: Evaluates a sampler for a one-dimensional two-peak target density
#              p(x) proportional to exp(-4 * (x**2 - 1)**2).
#              The sampler may only use the provided uniform(k) function as randomness.
#              A sampler is valid if its samples pass a Kolmogorov-Smirnov check against
#              the target; valid samplers are scored by the uniform numbers used per sample.
#              This module is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
#
# Parameters:
#    - timeout_seconds: Maximum allowed time (in seconds) for the evaluation process: int (default: 30).
#    - n_samples: Number of samples requested in each run: int (default: 10000).
#    - n_runs: Number of runs with different random seeds: int (default: 3).
#    - max_uniforms_per_sample: Uniform numbers allowed per sample before a run fails: int (default: 100).
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
from llm4ad.task.optimization.sampler_design.template import template_program, task_description

__all__ = ['SamplerDesignEvaluation', 'target_density']


def target_density(x):
    """Unnormalized target density with peaks at x = -1 and x = 1."""
    return np.exp(-4.0 * (np.asarray(x, dtype=float) ** 2 - 1.0) ** 2)


class _UniformSource:
    """The only randomness given to a sampler; counts the uniform numbers it draws."""

    def __init__(self, seed: int, limit: int):
        self._rng = np.random.default_rng(seed)
        self.limit = limit
        self.used = 0

    def __call__(self, k: int = 1) -> np.ndarray:
        k = int(k)
        if k < 0:
            raise ValueError('k must be non-negative')
        self.used += k
        if self.used > self.limit:
            raise RuntimeError(f'the sampler used more than {self.limit} uniform numbers')
        return self._rng.random(k)


class SamplerDesignEvaluation(Evaluation):
    """Evaluator for designing a sampler of a simple two-peak density."""

    def __init__(self,
                 timeout_seconds=30,
                 n_samples=10000,
                 n_runs=3,
                 max_uniforms_per_sample=100,
                 **kwargs):

        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds
        )

        self.n_samples = n_samples
        self.n_runs = n_runs
        self.max_uniforms_per_sample = max_uniforms_per_sample
        # Accept a run if its KS distance is below 2 / sqrt(n); an exact sampler fails
        # this check with probability below 0.1%.
        self.ks_threshold = 2.0 / np.sqrt(n_samples)

        # The target CDF on a fine grid; the density outside [-3, 3] is below exp(-250).
        self._grid = np.linspace(-3.0, 3.0, 60001)
        density = target_density(self._grid)
        cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2)])
        self._cdf = cdf / cdf[-1]

    def evaluate_program(self, program_str: str, callable_func: callable) -> Any | None:
        return self.evaluate(callable_func)

    def ks_distance(self, samples: np.ndarray) -> float:
        """Kolmogorov-Smirnov distance between the samples and the target distribution."""
        samples = np.sort(samples)
        n = len(samples)
        cdf = np.interp(samples, self._grid, self._cdf)
        return float(max(np.max(np.arange(1, n + 1) / n - cdf), np.max(cdf - np.arange(n) / n)))

    def run_sampler(self, eva: callable, seed: int):
        """Run the sampler once; return (samples, uniforms used) or None if the output is invalid."""
        source = _UniformSource(seed, self.max_uniforms_per_sample * self.n_samples)
        samples = np.asarray(eva(self.n_samples, source), dtype=float)
        if samples.shape != (self.n_samples,) or not np.all(np.isfinite(samples)):
            return None
        return samples, source.used

    def evaluate(self, eva: callable) -> float | None:
        runs = []
        for seed in range(self.n_runs):
            result = self.run_sampler(eva, seed)
            if result is None or self.ks_distance(result[0]) > self.ks_threshold:
                return None
            runs.append(result)

        # Samples must come from uniform(k): the same seed must reproduce the same samples,
        # and different seeds must give different samples.
        repeat = self.run_sampler(eva, 0)
        if repeat is None or not np.array_equal(repeat[0], runs[0][0]):
            return None
        if len(runs) > 1 and np.array_equal(runs[0][0], runs[1][0]):
            return None
        used = [n_uniforms for _, n_uniforms in runs]

        # Any sampler needs at least one uniform number per sample, so -1 is the best score.
        uniforms_per_sample = np.mean(used) / self.n_samples
        return -max(float(uniforms_per_sample), 1.0)


if __name__ == '__main__':
    def sample(n: int, uniform: callable) -> np.ndarray:
        """Inverse-CDF sampling with a lookup table: one uniform number per sample."""
        grid = np.linspace(-3.0, 3.0, 20001)
        density = np.exp(-4.0 * (grid ** 2 - 1.0) ** 2)
        cdf = np.concatenate([[0.0], np.cumsum((density[1:] + density[:-1]) / 2)])
        cdf /= cdf[-1]
        return np.interp(uniform(n), cdf, grid)

    task = SamplerDesignEvaluation()
    print('inverse CDF:', task.evaluate_program('_', sample))
