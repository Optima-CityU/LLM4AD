# SPDX-License-Identifier: BSD-2-Clause
template_program = '''
import numpy as np

def sample(n: int, uniform: callable) -> np.ndarray:
    """
    Draw n samples from the target density p(x) proportional to exp(-4 * (x**2 - 1)**2).

    Args:
        n: Number of samples to return.
        uniform: uniform(k) returns a numpy array of k independent Uniform[0, 1) numbers.
                 This is the only source of randomness you may use.

    Return:
        A numpy array of shape (n,) with the samples.
    """
    # Rejection sampling with a flat proposal on [-2, 2]; p(x) <= 1 everywhere.
    samples = []
    while len(samples) < n:
        x = -2.0 + 4.0 * uniform(1)[0]
        if uniform(1)[0] < np.exp(-4.0 * (x ** 2 - 1.0) ** 2):
            samples.append(x)
    return np.array(samples)
'''

task_description = "Design a sampling algorithm for the one-dimensional target density \
p(x) proportional to exp(-4 * (x**2 - 1)**2), which has two peaks at x = -1 and x = 1. \
The algorithm may only use the provided uniform(k) function as its source of randomness. \
The samples must follow the target distribution, and the algorithm should use as few uniform random numbers per sample as possible."
