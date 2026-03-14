import json
import os
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import numpy as np
import seaborn as sns

import base64
from io import BytesIO
from PIL import Image


def load_all_individuals(directory_path):
    """
    Loads individual data from all JSON files in the specified directory.

    Parameters:
        directory_path: Directory path containing the JSON files.

    Returns:
        A dictionary where the key is the sample_order and the value is the individual data.
"""

    individuals = {}

    clean_path = directory_path.replace('\x00', '').strip()

    try:
        for filename in os.listdir(clean_path):
            if filename.endswith('.json') and filename.startswith('samples_'):
                file_path = os.path.join(clean_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for ind in data:
                                if 'sample_order' in ind:
                                    individuals[ind['sample_order']] = ind
                        elif isinstance(data, dict) and 'sample_order' in data:
                            individuals[data['sample_order']] = data
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                    print(f"Warning: Could not process file {filename}: {e}")
    except FileNotFoundError:
        print(f"Error: Directory not found: {clean_path}")
    except Exception as e:
        print(f"Error accessing directory {clean_path}: {e}")

    return individuals


def extract_lineage(individuals, target_order):
    """
    Extracts the main lineage path from the initial individual to the target individual (selecting the older parent).

    Parameters:
        individuals: Dictionary of all individuals.
        target_order: Sample order of the target individual.

    Returns:
        A list of individuals on the main lineage path, where each element is a (order, generation, score) tuple.
"""

    lineage = []
    current_order = target_order

    while current_order in individuals:
        ind = individuals[current_order]
        generation = ind.get('generation', 0)
        score = ind.get('score', 0)

        lineage.append((current_order, generation, score))

        parents = ind.get('parents', [])
        if not parents:
            break

        parent_orders = []
        for p in parents:
            if p in individuals:
                parent_orders.append(p)

        if not parent_orders:
            break

        parent_generations = [individuals[p].get('generation', generation - 1) for p in parent_orders]


        min_gen = min(parent_generations)
        min_gen_parents = [p for p, gen in zip(parent_orders, parent_generations) if gen == min_gen]


        current_order = min_gen_parents[0]


    lineage.reverse()
    return lineage

def extract_lineage_score(individuals, target_order, mode='high'):
    """
    Extracts the main lineage path from the initial individual to the target individual (selecting the parent with the lower score).

    Parameters:
        individuals: Dictionary of all individuals.
        target_order: Sample order of the target individual.

    Returns:
        A list of individuals on the main lineage path, where each element is a (order, generation, score) tuple.
"""

    lineage = []
    current_order = target_order

    while current_order in individuals:
        ind = individuals[current_order]
        generation = ind.get('generation', 0)
        score = ind.get('score', 0)

        lineage.append((current_order, generation, score))

        parents = ind.get('parents', [])
        if not parents:
            break

        parent_orders = []
        parent_scores = []
        for p in parents:
            if p in individuals:
                parent_orders.append(p)
                parent_scores.append(individuals[p].get('score', 0))

        if not parent_orders:
            break

        if mode == 'high':
            min_score = max(parent_scores)
        else:
            min_score = min(parent_scores)
        min_score_parents = [p for p, s in zip(parent_orders, parent_scores) if s == min_score]


        current_order = min_score_parents[0]


    lineage.reverse()
    return lineage

def find_ancestors(individuals, target_order, initial_generation=0):
    """
    Finds all ancestors of a given individual.

    Parameters:
        individuals: Dictionary of all individuals.
        target_order: Sample order of the target individual.

    Returns:
        A set containing all ancestor nodes, where each node is a (sample_order, generation) tuple.
"""

    if target_order not in individuals:
        raise ValueError(f"Individual with sample_order {target_order} not found")

    ancestors = set()
    queue = deque()

    target_ind = individuals[target_order]
    queue.append((target_order, target_ind.get('generation', 0)))

    while queue:
        order, gen = queue.popleft()
        if (order, gen) in ancestors:
            continue

        ancestors.add((order, gen))

        if order in individuals:
            parents = individuals[order].get('parents', [])
            for parent_order in parents:
                if parent_order in individuals:
                    parent_gen = individuals[parent_order].get('generation', gen - 1)
                    queue.append((parent_order, parent_gen))
                else:
                    print(f"Warning: Parent {parent_order} not found in records")

    return ancestors

def print_structured_family_tree(G, target_order):
    """
    Prints structured family tree information.

    Parameters:
        G: Directed graph object.
        target_order: Sample order of the target individual.
"""

    print("\n" + "=" * 50)
    print("STRUCTURED FAMILY TREE ANALYSIS")
    print("=" * 50)

    generations = nx.get_node_attributes(G, 'generation')
    nodes_by_generation = {}
    for node, gen in generations.items():
        if gen not in nodes_by_generation:
            nodes_by_generation[gen] = []
        nodes_by_generation[gen].append(node)


    for gen in sorted(nodes_by_generation.keys()):
        print(f"\nGeneration {gen}:")
        for node in sorted(nodes_by_generation[gen], key=lambda x: int(x.split('I')[1])):
            node_data = G.nodes[node]
            order = node_data['order']
            score = node_data['score']
            operator = node_data['operator']


            parents = list(G.predecessors(node))
            parent_orders = [p.split('I')[1] for p in parents]


            target_marker = " (TARGET)" if order == target_order else ""

            print(f"  Individual {order}{target_marker}:")
            print(f"    Score: {score:.4f}")
            print(f"    Operator: {operator}")
            print(f"    Parents: {parent_orders if parent_orders else 'None'}")


    print("\n" + "-" * 50)
    print("FAMILY TREE STATISTICS:")
    print(f"Total individuals in tree: {len(G.nodes())}")
    print(f"Number of generations: {len(nodes_by_generation)}")


    print("\nAverage score by generation:")
    for gen in sorted(nodes_by_generation.keys()):
        scores = [G.nodes[node]['score'] for node in nodes_by_generation[gen]]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"  Generation {gen}: {avg_score:.4f} (from {len(scores)} individuals)")


    operators = {}
    for node in G.nodes():
        op = G.nodes[node]['operator']
        if op not in operators:
            operators[op] = 0
        operators[op] += 1

    print("\nOperator usage frequency:")
    for op, count in sorted(operators.items(), key=lambda x: x[1], reverse=True):
        print(f"  {op}: {count} times ({count / len(G.nodes()) * 100:.1f}%)")


def build_family_tree(individuals, ancestors):
    """
    Constructs a directed graph of the family tree.

    Parameters:
        individuals: Dictionary of all individuals.
        ancestors: Set containing all ancestor nodes.

    Returns:
        A directed graph object.
"""

    G = nx.DiGraph()

    for order, gen in ancestors:
        if order not in individuals:
            continue

        ind = individuals[order]
        node_label = f"G{gen}I{order}"
        G.add_node(node_label,
                   generation=gen,
                   order=order,
                   score=ind.get('score', 0),
                   operator=ind.get('operator', ''))

        parents = ind.get('parents', [])
        for parent_order in parents:
            if parent_order in individuals:
                parent_gen = individuals[parent_order].get('generation', gen - 1)
                parent_label = f"G{parent_gen}I{parent_order}"
                G.add_edge(parent_label, node_label)

    return G


def draw_family_tree(G, target_order, candidate):
    """
    Plots the family tree with different colors for different operators, correctly handling cross-generation parent-child relationships.

    Parameters:
        G: Directed graph object.
        target_order: Sample order of the target individual.
"""


    sns.set(style="whitegrid", font_scale=1.2)
    sns.set_palette("husl")


    operator_colors = {
        'm1_M': sns.color_palette("husl", 8)[0],
        'm2_M': sns.color_palette("husl", 8)[1],
        'e1': sns.color_palette("husl", 8)[2],
        'e2': sns.color_palette("husl", 8)[3],
        'm1': sns.color_palette("husl", 8)[4],
        'm2': sns.color_palette("husl", 8)[5],
        'Initialization': sns.color_palette("husl", 8)[6],
        'Template': sns.color_palette("husl", 8)[7],
    }


    generations = nx.get_node_attributes(G, 'generation')
    if not generations:
        print("No generation information found in the graph.")
        return


    gen_scores = defaultdict(list)
    for node in G.nodes():
        gen = generations[node]
        score = G.nodes[node]['score']
        gen_scores[gen].append(score)


    gen_min_max = {}
    for gen in gen_scores:
        gen_min_max[gen] = (min(gen_scores[gen]), max(gen_scores[gen]))


    pos = {}
    gen_offset = {}


    max_gen = max(generations.values())
    min_gen = min(generations.values())


    for gen in sorted(generations.values()):

        gen_offset[gen] = (max_gen - gen) * 1.5


    for node in G.nodes():
        gen = generations[node]
        score = G.nodes[node]['score']
        min_score, max_score = gen_min_max[gen]


        if max_score != min_score:
            normalized_score = (score - min_score) / (max_score - min_score)
        else:
            normalized_score = 0.5


        pos[node] = (gen, gen_offset[gen] + normalized_score * 0.8)


    node_colors = []
    for node in G.nodes():
        order = int(node.split('I')[1])
        operator = G.nodes[node].get('operator', '')


        if not operator:
            operator = 'Initialization' if G.nodes[node]['generation'] == 0 else 'Template'


        if order == target_order:
            node_colors.append('red')
        else:
            node_colors.append(operator_colors.get(operator, sns.color_palette("husl")[-1]))

    plt.figure(figsize=(14, 10))


    nx.draw_networkx_edges(G, pos, arrowsize=15, width=1.5, alpha=0.7, connectionstyle="arc3,rad=0.1",
                           edge_color='gray')


    nx.draw_networkx_nodes(G, pos, node_size=800, node_color=node_colors, alpha=0.9, linewidths=2, edgecolors='white')


    labels = {}
    for node in G.nodes():
        parts = node.split('I')
        gen = parts[0][1:]
        order = parts[1]
        node_data = G.nodes[node]
        score = node_data.get('score', 'N/A')
        operator = node_data.get('operator', 'N/A')
        labels[node] = f"{parts[0]}\nI{order}\n{score:.4f}\n{operator}"

    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family='sans-serif')


    ax = plt.gca()
    for gen in gen_offset:

        ax.axhline(y=gen_offset[gen], color='gray', linestyle='--', alpha=0.3)

        ax.text(-0.5, gen_offset[gen] + 0.4, f'Generation {gen}',
                ha='right', va='center', color='gray')


    all_x = [pos[0] for pos in pos.values()]
    all_y = [pos[1] for pos in pos.values()]

    x_margin = 4
    min_x = min(all_x) - x_margin
    max_x = max(all_x) + x_margin

    y_margin = 4
    min_y = min(all_y) - y_margin
    max_y = max(all_y) + y_margin

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)


    plt.title(f"Family Tree of Individual {target_order} in {candidate} experiment\n"
              f"(Colors represent different operators)",
              fontsize=16, pad=20)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Score (normalized within generation)", fontsize=14)


    legend_elements = []

    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor='red', markersize=10,
                                      label='Target Individual'))

    for op, color in operator_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=10,
                                          label=op))

    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=sns.color_palette("husl")[-1], markersize=10,
                                      label='Unknown Operator'))

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, framealpha=0.9)


    plt.grid(True, linestyle='--', alpha=0.3)


    plt.tight_layout()
    plt.show()


def analyze_population_data(directory):
    """
    Analyzes population data and returns the score distribution and operator usage for each generation.

    Parameters:
        directory: Directory containing the population JSON files.

    Returns:
        generation_stats: Statistical information for each generation.
        operator_stats: Operator usage statistics.
        all_scores: List of scores for all individuals in each generation.
"""

    generation_stats = defaultdict(list)
    operator_stats = defaultdict(lambda: defaultdict(int))
    all_scores = defaultdict(list)

    for filename in os.listdir(directory):
        if filename.startswith('pop_') and filename.endswith('.json'):
            try:
                with open(os.path.join(directory, filename), 'r') as f:
                    content = f.read().strip()
                    if not content.startswith('['):
                        content = '[' + content + ']'

                    content = content.replace('},]', '}]')

                    data = json.loads(content)


                    gen_number = int(filename.split('_')[1].split('.')[0]) - 1


                    scores = []
                    operators = []

                    for individual in data:
                        if isinstance(individual, dict):
                            score = individual.get('score', 0)
                            regis_num = individual.get('pop_register_number', None)


                            operator = individual.get('operator')
                            if operator is None:
                                operator = "Template"
                            elif operator == "":
                                operator = "Initialization"

                            scores.append(score)
                            operators.append(operator)


                    if scores:
                        avg_score = sum(scores) / len(scores)
                        max_score = max(scores)
                        min_score = min(scores)

                        generation_stats[gen_number] = {
                            'avg_score': avg_score,
                            'max_score': max_score,
                            'min_score': min_score,
                            'count': len(scores)
                        }


                        all_scores[gen_number] = scores


                        for op in operators:
                            operator_stats[gen_number][op] += 1

            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
                continue

    return generation_stats, operator_stats, all_scores


def plot_generation_stats_with_all_lineages(generation_stats, all_scores, all_lineages, highlight_lineage,
                                            candidate=""):
    """
    Plots the score distribution and all lineage paths, with one path highlighted.

    Parameters:
        generation_stats: Statistical information for each generation.
        all_scores: List of scores for all individuals in each generation.
        all_lineages: List of all lineage paths.
        highlight_lineage: The lineage path(s) to highlight (list of tuples).
        candidate: Name of the experiment.
"""
    generations = sorted(generation_stats.keys())
    avg_scores = [generation_stats[g]['avg_score'] for g in generations]
    max_scores = [generation_stats[g]['max_score'] for g in generations]
    min_scores = [generation_stats[g]['min_score'] for g in generations]


    sns.set(style="whitegrid")


    plt.figure(figsize=(14, 8))


    for gen in generations:
        scores = all_scores[gen]
        x = [gen] * len(scores)
        sns.scatterplot(x=x, y=scores, color='gray', alpha=0.3, s=20,
                        label='Population Scores' if gen == generations[0] else "")


    # plt.plot(generations, avg_scores, label='Average Score', marker='o', color='blue', markersize=4)
    plt.plot(generations, max_scores, label='Max Score', linestyle='--', color='red', markersize=8)
    plt.plot(generations, min_scores, label='Min Score', linestyle='--', color='green', markersize=8)


    added_highlight_legend = False
    added_other_legend = False


    for i, lineage in enumerate(all_lineages):

        x_line = [m[1] for m in lineage]  # generation
        y_line = [m[2] for m in lineage]  # score
        orders_line = [m[0] for m in lineage]  # order


        is_highlight = all(a[0] == b[0] for a, b in zip(lineage, highlight_lineage))


        if is_highlight:

            line_style = {'color': 'blue', 'linewidth': 3, 'alpha': 1.0}
            point_style = {'color': 'blue', 's': 80, 'marker': 's', 'alpha': 1.0}
            text_style = {'color': 'blue', 'weight': 'bold', 'fontsize': 10}


            label = 'Highlighted Lineage' if not added_highlight_legend else None
            added_highlight_legend = True
        else:

            normal_alpha = 0.5
            line_style = {'color': 'gold', 'linewidth': 1, 'alpha': normal_alpha}
            point_style = {'color': 'gold', 's': 30, 'marker': 'o', 'alpha': normal_alpha}
            text_style = {'color': 'gold', 'weight': 'normal', 'fontsize': 8, 'alpha': normal_alpha}


            label = 'Other Lineages' if not added_other_legend else None
            added_other_legend = True


        plt.plot(x_line, y_line, '-', label=label, **line_style)


        plt.scatter(x_line, y_line, **point_style)


        if is_highlight:
            for j, order in enumerate(orders_line):
                plt.annotate(f"{order}", (x_line[j], y_line[j]), textcoords="offset points",
                             xytext=(0, 15), ha='center', **text_style)
        else:
            if orders_line:
                plt.annotate(f"{orders_line[0]}", (x_line[0], y_line[0]), textcoords="offset points",
                             xytext=(0, 15), ha='center', **text_style)

                plt.annotate(f"{orders_line[-1]}", (x_line[-1], y_line[-1]), textcoords="offset points",
                             xytext=(0, -15), ha='center', **text_style)

    plt.legend()

    plt.title(f'{candidate} Score Statistics with All Lineages (Highlighted in Red)')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.savefig("lineage_plot_higtlight.png", dpi=900, bbox_inches='tight')  # bbox_inches 避免裁剪
    plt.show()


def print_lineage_responses_and_algorithms(individuals, lineage):
    """
    Print the 'response' and 'algorithm' fields for all individuals in a lineage

    Parameters:
        individuals: Dictionary of all individuals
        lineage: List of tuples (order, generation, score) representing the lineage
    """
    print("\n" + "=" * 50)
    print("LINEAGE RESPONSES AND ALGORITHMS")
    print("=" * 50)

    for order, gen, score in lineage:
        if order in individuals:
            ind = individuals[order]
            response = ind.get('response', 'N/A')
            algorithm = ind.get('algorithm', 'N/A')
            operator_it = ind.get('operator', 'N/A')
            code_it = ind.get('function', 'N/A')
            print(f"\nIndividual {order} (Generation {gen}, Score {score:.4f}, Operator {operator_it}):")
            print(f"Algorithm: {algorithm}")
            if order == 2 or order == 1:
                print(f'Code:\n{code_it}')
            print("Response:")
            print(response)
        else:
            print(f"\nWarning: Individual {order} not found in records")




def plot_generation_stats_with_all_lineages_norm(generation_stats, all_scores, all_lineages, highlight_lineage,
                                            candidate=""):
    """
    Plots the score distribution and all lineage paths.

    Parameters:
        generation_stats: Statistical information for each generation.
        all_scores: List of scores for all individuals in each generation.
        all_lineages: List of all lineage paths.
        highlight_lineage: The lineage path(s) to highlight (list of tuples).
        candidate: Name of the experiment.
"""


    generations = sorted(generation_stats.keys())
    avg_scores = [generation_stats[g]['avg_score'] for g in generations]
    max_scores = [generation_stats[g]['max_score'] for g in generations]
    min_scores = [generation_stats[g]['min_score'] for g in generations]


    sns.set(style="whitegrid")


    plt.figure(figsize=(14, 8))


    for gen in generations:
        scores = all_scores[gen]
        x = [gen] * len(scores)
        sns.scatterplot(x=x, y=scores, color='gray', alpha=0.3, s=20,
                        label='Population Scores' if gen == generations[0] else "")


    # plt.plot(generations, avg_scores, label='Average Score', marker='o', color='blue', markersize=4)
    plt.plot(generations, max_scores, label='Max Score', linestyle='--', color='red', markersize=8)
    plt.plot(generations, min_scores, label='Min Score', linestyle='--', color='green', markersize=8)


    added_highlight_legend = False
    added_other_legend = False


    for i, lineage in enumerate(all_lineages):

        x_line = [m[1] for m in lineage]  # generation
        y_line = [m[2] for m in lineage]  # score
        orders_line = [m[0] for m in lineage]  # order

        is_highlight = all(a[0] == b[0] for a, b in zip(lineage, highlight_lineage))


        if is_highlight:

            line_style = {'color': 'blue', 'linewidth': 3, 'alpha': 1.0}
            point_style = {'color': 'blue', 's': 80, 'marker': 's', 'alpha': 1.0}
            text_style = {'color': 'blue', 'weight': 'bold', 'fontsize': 10}


            label = 'Highlighted Lineage' if not added_highlight_legend else None
            added_highlight_legend = True
        else:

            normal_alpha = 0.05
            line_style = {'color': 'gold', 'linewidth': 1, 'alpha': normal_alpha}
            point_style = {'color': 'gold', 's': 30, 'marker': 'o', 'alpha': normal_alpha}
            text_style = {'color': 'gold', 'weight': 'normal', 'fontsize': 8, 'alpha': normal_alpha}


            label = 'Lineages' if not added_other_legend else None
            added_other_legend = True


        plt.plot(x_line, y_line, '-', label=label, **line_style)


        plt.scatter(x_line, y_line, **point_style)

    legend = plt.legend(bbox_to_anchor=(0.22, 1))
    for line in legend.get_lines():
        line.set_alpha(1.0)

    plt.grid(False)
    plt.title(f'{candidate} Score Statistics with All Lineages (Highlighted in Red)')
    plt.xlabel('Generation')
    plt.ylabel('Score')

    plt.savefig("lineage_plot_highlight_nonumber.png", dpi=900, bbox_inches='tight')
    plt.show()
    plt.show()


def plot_generation_stats_with_family_members(generation_stats, all_scores, family_members, lineage_members, candidate=""):
    """
    Plots the score distribution and marks the members of the family tree.

    Parameters:
        generation_stats: Statistical information for each generation.
        all_scores: List of scores for all individuals in each generation.
        family_members: List of family tree members, where each member is a (order, generation, score) tuple.
"""

    generations = sorted(generation_stats.keys())
    avg_scores = [generation_stats[g]['avg_score'] for g in generations]
    max_scores = [generation_stats[g]['max_score'] for g in generations]
    min_scores = [generation_stats[g]['min_score'] for g in generations]


    sns.set(style="whitegrid")


    plt.figure(figsize=(14, 8))


    for gen in generations:
        scores = all_scores[gen]
        x = [gen] * len(scores)
        sns.scatterplot(x=x, y=scores, color='gray', alpha=0.3, s=20,
                        label='Population Scores' if gen == generations[0] else "",
                        zorder=1)

    plt.plot(generations, max_scores, label='Max Score', linestyle='--', color='green', markersize=8,
             zorder=2)
    plt.plot(generations, min_scores, label='Min Score', linestyle='--', color='red', markersize=8, zorder=2)


    family_generations = set(m[1] for m in family_members)
    for gen in family_generations:

        members = [m for m in family_members if m[1] == gen]
        x = [gen] * len(members)
        y = [m[2] for m in members]
        orders = [m[0] for m in members]

        sns.scatterplot(x=x, y=y, color='red', s=100, marker='*',
                        label='Family Members' if gen == min(family_generations) else "",
                        zorder=3)


        for i, order in enumerate(orders):
            plt.annotate(f"{order}", (x[i], y[i]), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8, color='red',
                         zorder=4)

    if lineage_members:
        x_line = [m[1] for m in lineage_members]
        y_line = [m[2] for m in lineage_members]
        orders_line = [m[0] for m in lineage_members]

        plt.plot(x_line, y_line, 'b-', linewidth=2, label='Lineage Path')

        sns.scatterplot(x=x_line, y=y_line, color='blue', s=150, marker='s',
                        label='Lineage Members' if len(lineage_members) > 0 else "")

        for i, order in enumerate(orders_line):
            plt.annotate(f"{order}", (x_line[i], y_line[i]), textcoords="offset points",
                         xytext=(0, 15), ha='center', fontsize=10, color='blue', weight='bold')

    plt.title(f'{candidate} Score Statistics Across Generations with Family Members Highlighted')
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


def find_all_lineages(individuals, target_order):
    """
    Finds all lineage paths from the initial individual to the target individual.

    Parameters:
        individuals: Dictionary of all individuals.
        target_order: Sample order of the target individual.

    Returns:
        A list of all lineage paths, where each path is a list of (order, generation, score) tuples.
"""


    def dfs(current_order, path):
        current_ind = individuals[current_order]
        current_gen = current_ind.get('generation', 0)
        current_score = current_ind.get('score', 0)
        new_path = path + [(current_order, current_gen, current_score)]

        parents = current_ind.get('parents', [])
        valid_parents = [p for p in parents if p in individuals]

        if not valid_parents:
            all_paths.append(new_path)
            return

        for parent in valid_parents:
            dfs(parent, new_path)

    all_paths = []

    dfs(target_order, [])

    reversed_paths = [list(reversed(path)) for path in all_paths]

    return reversed_paths

def trace_family_tree_and_analyze(samples_dir, population_dir, target_order, candidate=''):
    """
    Main function: Tracks the family tree and analyzes the position within the population.

    Parameters:
        samples_dir: Directory path containing the samples JSON files.
        population_dir: Directory path containing the population JSON files.
        target_order: Sample order of the target individual.
"""


    print(f"Loading individuals from directory: {samples_dir}")
    individuals = load_all_individuals(samples_dir)
    if not individuals:
        print("No individuals loaded. Please check the directory path and file contents.")
        return

    print(f"Loaded {len(individuals)} individuals")


    print(f"Finding ancestors of individual {target_order}")
    try:
        ancestors = find_ancestors(individuals, target_order)
        print(f"Found {len(ancestors)} ancestors in the family tree")
    except ValueError as e:
        print(e)
        return


    print("Building family tree...")
    G = build_family_tree(individuals, ancestors)

    if len(G.nodes()) == 0:
        print("No family tree could be built.")
        return


    print_structured_family_tree(G, target_order)


    print("\nDrawing family tree...")
    draw_family_tree(G, target_order, candidate)


    print("\nExtracting main lineage path...")
    highlight_lineage = extract_lineage_score(individuals, target_order, mode='high')
    # highlight_lineage = extract_lineage_low_score(individuals, target_order)
    print(f"Highlight lineage path extracted with {len(highlight_lineage)} members:")
    for order, gen, score in highlight_lineage:
        print(f"  Generation {gen}, Order {order}, Score {score:.4f}")


    print("\nFinding all lineage paths...")
    all_lineages = find_all_lineages(individuals, target_order)
    print(f"Found {len(all_lineages)} lineage paths:")
    for i, lineage in enumerate(all_lineages):
        print(f"Path {i + 1}: {[m[0] for m in lineage]}")


    print("\nAnalyzing population data...")
    gen_stats, op_stats, all_scores = analyze_population_data(population_dir)


    family_members = []
    for node in G.nodes():
        node_data = G.nodes[node]
        family_members.append((
            node_data['order'],
            node_data['generation'],
            node_data['score']
        ))

    base64image=None
    if base64image is not None:

        image_data = base64.b64decode(base64image)


        image = Image.open(BytesIO(image_data))


        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    family_members = []
    for node in G.nodes():
        node_data = G.nodes[node]
        family_members.append((
            node_data['order'],
            node_data['generation'],
            node_data['score']
        ))

    print("\nPlotting population with all lineages (highlighting main path)...")
    plot_generation_stats_with_all_lineages(gen_stats, all_scores, all_lineages, highlight_lineage, candidate)

    print("\nPlotting population with family members and main lineage path...")
    plot_generation_stats_with_family_members(gen_stats, all_scores, family_members, highlight_lineage, candidate)

    print_lineage_responses_and_algorithms(individuals, highlight_lineage)

if __name__ == "__main__":
    directories = {
        'car_MLES': r"C:\0_QL_work\014_mmeoh\LLM4AD_MLES\LLM4AD\example\mles_moonlander\logs\MLES\20260208_215147",
        }

    see_samples = {
                    'car_MLES': 100,
                    }

    candidates = ['car_MLES']
    index = 0
    candidate = candidates[index]

    directory = directories[candidate]
    samples_directory = directory + r"\samples"
    population_directory = directory + r"\population"

    target_order = see_samples[candidate]

    trace_family_tree_and_analyze(samples_directory, population_directory, target_order, candidate)
