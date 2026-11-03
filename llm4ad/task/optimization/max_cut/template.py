# SPDX-License-Identifier: BSD-2-Clause
template_program = '''
import numpy as np
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
    selected_index = 0

    return selected_index
'''

task_description = "Given an undirected weighted graph, you need to find a maximum cut (Max-Cut), \
which partitions all nodes into two sets S and T such that the total weight of edges between S and T is maximized. \
The task is solved by iteratively flipping nodes between S and T using local search. \
In each step, given a set of candidate nodes that can improve the cut, help me design a novel algorithm \
that selects the best node to flip, different from classic strategies like choosing the node with the highest gain."
