# SPDX-License-Identifier: BSD-2-Clause
import numpy as np


class GetData:
    def __init__(
            self,
            n_instance: int,
            problem_size: int,
            max_length_ratio: float = 0.35,
            seed: int = 2024,
    ):
        self.n_instance = int(n_instance)
        self.problem_size = int(problem_size)
        self.max_length_ratio = float(max_length_ratio)
        self.seed = int(seed)

    def generate_instances(self):
        rng = np.random.default_rng(self.seed)
        instance_data = []

        for _ in range(self.n_instance):
            coordinates = rng.random((self.problem_size, 2))
            coordinates[0] = np.array([0.5, 0.5])

            distances = np.linalg.norm(
                coordinates[:, np.newaxis] - coordinates,
                axis=2,
            )

            prizes = rng.uniform(0.1, 1.0, self.problem_size)
            prizes[0] = 0.0

            max_length = float(self.max_length_ratio * self.problem_size)

            instance_data.append({
                "coordinates": coordinates,
                "distance_matrix": distances,
                "prizes": prizes,
                "start_node": 0,
                "end_node": 0,
                "max_length": max_length,
            })

        return instance_data
