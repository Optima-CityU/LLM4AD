# SPDX-License-Identifier: BSD-2-Clause
import numpy as np


class GetData():
    def __init__(self, n_instance, n_nodes, edge_probability=0.5):
        self.n_instance = n_instance
        self.n_nodes = n_nodes
        self.edge_probability = edge_probability

    def generate_instances(self):
        np.random.seed(2026)
        instance_data = []
        for _ in range(self.n_instance):
            adjacency = {}
            for i in range(self.n_nodes):
                adjacency[i] = {}

            for i in range(self.n_nodes):
                for j in range(i + 1, self.n_nodes):
                    if np.random.rand() < self.edge_probability:
                        weight = np.random.uniform(0.5, 2.0)
                        # weight = 1.0
                        adjacency[i][j] = weight
                        adjacency[j][i] = weight

            nodes = list(range(self.n_nodes))
            instance_data.append((adjacency, nodes))

        return instance_data
