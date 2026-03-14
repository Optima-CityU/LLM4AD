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
import numpy as np
import random
import traceback
import os
from typing import List, Dict, Tuple, Any, Optional
from threading import RLock

from .clusterunit import ClusterUnit
from .externalArchive import ExternalArchive
from llm4ad.base import Function
from .base import Evoind
from codebleu import calc_codebleu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


def get_bert_embeddings(texts: List[str], model_path: str = None) -> np.ndarray:
    """
    An optional algorithm feature mapping for algorithm clustering.
    Generates text embeddings using a BERT model for semantic analysis of algorithm descriptions (Thoughts).

    Args:
        texts: A list of strings to be embedded.
        model_path: Local path to BERT model. Defaults to 'bert-base-uncased'.

    Returns:
        A numpy array of shape (len(texts), 768) containing the CLS token embeddings."""
    try:
        from transformers import BertTokenizer, BertModel
        import torch
    except ImportError as e:
        print(f"⚠️ [Warning in clustermanager.py] Optional dependencies missing: {e}. Feature extraction might fail.")

    target_path = model_path if model_path and os.path.exists(model_path) else 'bert-base-uncased'

    try:
        tokenizer = BertTokenizer.from_pretrained(target_path)
        model = BertModel.from_pretrained(target_path)
    except Exception as e:
        print(f"❌ [Error] Failed to load BERT from {target_path}: {e}")
        return np.random.rand(len(texts), 768)

    embeddings = []
    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
            # Use the CLS token (index 0) as the sentence-level representation
            cls_embedding = hidden_states[0, 0, :].numpy()
            embeddings.append(cls_embedding)
        except Exception:
            embeddings.append(np.random.rand(768))

    return np.array(embeddings)


def individual_feature(population: List[Evoind],
                       feature_type: Tuple[str, ...] = ('ast',),
                       save_path: str = '',
                       bert_model_path: str = None):
    """
    Calculates multi-modal features for the population to facilitate niching/clustering.
    Supported types:
    - 'ast': Structural similarity via CodeBLEU.
    - 'language': Semantic similarity via BERT.
    - 'random': Gaussian noise baseline.
    - 'objective': Performance-based features.
    """
    if not population:
        return

    print(f'[Feature Extraction] Processing feature types: {feature_type}')
    population_size = len(population)
    features = [[] for _ in range(population_size)]

    # 1. AST Structural Features (CodeBLEU)
    if 'ast' in feature_type:
        AST = np.zeros((population_size, population_size))
        codes = [ind.function.to_code_without_docstring() for ind in population]
        for i in range(population_size):
            for j in range(i, population_size):
                if i == j:
                    score = 1.0
                else:
                    try:
                        cal_result = calc_codebleu([codes[i]], [codes[j]], lang='python',
                                                   weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
                        # Hybrid score of syntax match and dataflow match
                        score = 0.5 * cal_result['syntax_match_score'] + 0.5 * cal_result['dataflow_match_score']
                    except Exception:
                        score = 0.0

                AST[i, j] = score
                AST[j, i] = score

        # Add AST as one of the used features
        for i in range(population_size):
            features[i].extend(AST[i, :].tolist())

    # 2. Semantic Features (BERT)
    if 'language' in feature_type:
        texts = [ind.function.to_code_without_docstring() for ind in population]
        embeddings = get_bert_embeddings(texts, model_path=bert_model_path)
        for i in range(population_size):
            features[i].extend(embeddings[i, :].tolist())

    # 3. Random Baseline Features
    if 'random' in feature_type:
        random_features = np.random.normal(size=(population_size, 20))
        for i in range(population_size):
            features[i].extend(random_features[i, :].tolist())

    # 4. Objective/Fitness Features
    if 'objective' in feature_type:
        for i in range(population_size):
            # 处理 None 的情况
            obj = population[i].function.score if population[i].function.score is not None else 0
            features[i].append(obj)

    # --- Dimension Reduction & Embedding ---
    try:
        all_features = np.array(features)
        all_features = np.nan_to_num(all_features)
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)
        # PCA reduction for clustering efficiency
        n_components = min(10, population_size, all_features.shape[1])
        pca = PCA(n_components=n_components)
        all_features_reduced = pca.fit_transform(all_features)

        for i, ind in enumerate(population):
            ind.set_feature(all_features_reduced[i])

        if save_path:
            try:
                plt.figure(figsize=(8, 6))
                sns.heatmap(all_features_reduced, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
                plt.title('PCA of All Features')
                save_file = f"{save_path}_PCA_features.png"
                plt.savefig(save_file)
                plt.close()
                print(f"PCA Heatmap saved to {save_file}")
            except Exception as e:
                print(f"Plotting failed: {e}")

    except Exception as e:
        print(f"[Error] Feature processing failed: {e}")
        for ind in population:
            ind.set_feature(np.zeros(10))


class ClusterManager:
    """
    Global Orchestrator for the Partition-to-Evolve (PartEvo) framework.

    Workflow:
    1. Maintains a global population and an offspring buffer.
    2. Executes 'Cold Start' until sufficient individuals are collected.
    3. Performs Clustering based on extracted code features.
    4. Manages ClusterUnits, which handle sub-population evolution and operator selection.
    5. Implements resource allocation (resource tilt) based on cluster performance.
    """

    def __init__(self,
                 pop_size: int = 16,
                 n_clusters: int = 4,
                 intra_operators: Tuple[str, ...] = ('re', 'se', 'cc', 'lge'),
                 intra_operators_parent_num: Dict[str, int] = None,
                 intra_operators_frequency: Optional[Dict] = None,
                 use_resource_tilt: bool = False,
                 resource_tilt_alpha: float = 2.0,
                 feature_type: Tuple[str, ...] = ('ast',),

                 bert_model_path: str = None,
                 debug_flag: bool = False,
                 ):
        """
        Initialize the Cluster Manager.

        Args:
            pop_size: Target size for the global population.
            n_clusters: Number of niches/clusters to maintain.
            intra_operators: List of available evolutionary operators.
            use_resource_tilt: If True, high-performing clusters get more sampling opportunities.
        """

        self.debug_flag = debug_flag
        self.pop_size = pop_size
        self.generation = 0
        self.n_clusters = n_clusters
        self.bert_model_path = bert_model_path

        self.feature_type = feature_type

        self.is_initialized = False

        # --- Operator Weighted Sequence ---
        default_parent_num = {'re': 1, 'se': 1, 'cc': 1, 'lge': 1}
        self.intra_cluster_operators_parent_num = intra_operators_parent_num or default_parent_num

        # Expand operators based on frequency for weighted random selection
        freq_config = intra_operators_frequency or {op: 1 for op in intra_operators}

        expanded_ops = []
        for op in intra_operators:
            freq = freq_config.get(op, 1)
            expanded_ops.extend([op] * freq)
        self.intra_cluster_operators = tuple(expanded_ops)

        if intra_operators_parent_num:
            missing_keys = set(intra_operators) - set(intra_operators_parent_num.keys())
            if missing_keys:
                raise ValueError(f"intra_cluster_operators_parent_num is missing entries for: {sorted(missing_keys)}")

        print(f'[Manager] Operators Expanded Sequence: {self.intra_cluster_operators}')
        print(f'[Manager] Parent Requirements: {self.intra_cluster_operators_parent_num}')

        self.external_archive = ExternalArchive(max_elites=5,
                                                max_hard_negatives=30,
                                                summary_update_interval=int(self.n_clusters * 3))

        self.use_resource_tilt = use_resource_tilt
        self.resource_tilt_alpha = resource_tilt_alpha
        self.cluster_units: Dict[int, ClusterUnit] = {}
        self.global_best_ind: Optional[Function] = None
        self.population: List[Function] = []
        self.next_pop: List[Function] = []
        self._lock = RLock()

    def initial_population_clustering(self):
        """
        Creating initial niching.
        Converts Functions to Evoinds, extracts multi-modal features, and initializes ClusterUnits via Clustering (such as, K-Means)
        """
        valid_funcs = [f for f in self.population if f.score is not None]

        if len(valid_funcs) < self.n_clusters:
            print(f"❌ [Manager] Not enough individuals to cluster ({len(valid_funcs)}). Waiting...")
            return

        print(f"⏳ [Manager] Initializing Clustering with {len(valid_funcs)} individuals...")

        # Initialize the Evoind container for the subsequent algorithm allocation to various niches.
        temp_evo_pop = [Evoind(function=f) for f in valid_funcs]

        save_path = "init_debug" if self.debug_flag else ""
        individual_feature(temp_evo_pop, feature_type=self.feature_type,
                           save_path=save_path, bert_model_path=self.bert_model_path)

        features = []
        for ind in temp_evo_pop:
            if hasattr(ind, 'feature') and ind.feature is not None and len(ind.feature) > 0:
                features.append(ind.feature)
            else:
                features.append(np.zeros(10))
        features = np.array(features)

        try:
            if len(features) >= self.n_clusters:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
            else:
                labels = np.random.randint(0, self.n_clusters, size=len(temp_evo_pop))
        except Exception as e:
            print(f"⚠️ [Manager] KMeans failed: {e}. Using random assignment.----")
            labels = np.random.randint(0, self.n_clusters, size=len(temp_evo_pop))

        self.cluster_units.clear()
        grouped_pop = {i: [] for i in range(self.n_clusters)}

        for ind, label in zip(temp_evo_pop, labels):
            ind.cluster_id = label
            grouped_pop[label].append(ind)

        for c_id in range(self.n_clusters):
            unit_pop = grouped_pop[c_id]
            new_unit = ClusterUnit(
                cluster_id=c_id,
                max_pop_size=self.pop_size,
                intra_operators=self.intra_cluster_operators,
                intra_operators_parent_num=self.intra_cluster_operators_parent_num,
                pop=unit_pop
            )
            self.cluster_units[c_id] = new_unit

        self._update_global_best(temp_evo_pop)

        self.is_initialized = True
        print("🎉 [Manager] Clustering Finished. System Online. 👍")

    def _calculate_selection_probs(self) -> Tuple[List[int], List[float]]:
        """
        Calculates selection probabilities for clusters.
        Implements Softmax-based resource tilt towards high-performing clusters.
        """
        cluster_ids = []
        scores = []
        for c_id, unit in self.cluster_units.items():
            cluster_ids.append(c_id)
            if self.use_resource_tilt:
                best_ind = unit.get_best_individual()
                if best_ind and best_ind.function.score is not None:
                    scores.append(best_ind.function.score)
                else:
                    scores.append(-1e9)

        if not self.use_resource_tilt or not scores:
            n = len(cluster_ids)
            return cluster_ids, [1.0 / n] * n

        scores_arr = np.array(scores)
        exp_scores = np.exp((scores_arr - np.max(scores_arr)) * self.resource_tilt_alpha)
        sum_exp = np.sum(exp_scores)

        if sum_exp == 0:
            probs = [1.0 / len(cluster_ids)] * len(cluster_ids)
        else:
            probs = exp_scores / sum_exp
        return cluster_ids, probs

    def select_parent(self) -> Tuple[List[Function], str, int]:
        """
        Selects parent individuals for the next evolutionary step.
        Handles both 'Cold Start' (random selection) and 'Clustered' (unit-specific selection) phases.
        """
        with self._lock:
            # --- Phase A: Cold Start (Global Random) ---
            if not self.is_initialized:
                pool = self.population + self.next_pop
                valid_pool = [f for f in pool if f.score is not None]
                if not valid_pool:
                    return [], 'error', -1
                parent = random.choice(valid_pool)
                return [parent], 're', -1

            # --- Phase B: Evolutionary Niching ---
            cluster_ids, probs = self._calculate_selection_probs()
            if not cluster_ids:
                return [], 'error', -1

            chosen_c_id = np.random.choice(cluster_ids, p=probs)
            target_unit = self.cluster_units[chosen_c_id]

            # parents: List[Function], operator: str, need_external: bool
            parents, operator, need_external = target_unit.selection(help_inter=False)

            # --- Cross-Niches Collaboration ---
            if need_external:
                if operator == 'cn':    # Crossover with external help
                    other_units = [u for uid, u in self.cluster_units.items() if uid != chosen_c_id and len(u) > 0]
                    helper_func = None

                    if other_units:
                        helper_unit = random.choice(other_units)
                        h_parents, _, _ = helper_unit.selection(help_inter=True, mode='tournament', help_number=1)
                        if h_parents:
                            helper_func = h_parents[0]

                    if not helper_func and self.global_best_ind:
                        helper_func = self.global_best_ind

                    if helper_func:
                        parents.append(helper_func)

                elif operator == 'lge':
                    # Add Global Best Function
                    if self.global_best_ind:
                        if not any(f.body == self.global_best_ind.body for f in parents):
                            parents.append(self.global_best_ind)

                    # Add Cluster Best Function
                    cluster_best_evo = target_unit.get_best_individual()  # return: Evoind
                    if cluster_best_evo:
                        if not any(f.body == cluster_best_evo.function.body for f in parents):
                            parents.append(cluster_best_evo.function)

            return parents, operator, chosen_c_id

    def register_function(self, offspring: Function, from_which_cluster: int = None):
        """
        Registers a new function into the global population, external archive and according niches.
        Handles deduplication and elitist replacement.
        """
        with self._lock:
            if offspring.score is None:
                return

            try:
                # --- Deduplication & Rejuvenation Logic ---
                is_duplicate_or_neutral = False

                for i, existing_func in enumerate(self.population):
                    # Code same
                    code_match = (existing_func.body == offspring.body)
                    # Score same
                    score_match = (abs(existing_func.score - offspring.score) < 1e-9)

                    if code_match or score_match:
                        if offspring.score >= existing_func.score:
                            # Replace old version with new candidate to maintain diversity
                            print(
                                f"♻️ [Manager] Replacing existing individual (Score: {existing_func.score:.4f}) with new candidate.")
                            self.population[i] = offspring
                            is_duplicate_or_neutral = True
                            break
                        else:
                            return  # Drop inferior duplicates

                if is_duplicate_or_neutral:
                    self._update_global_best([Evoind(offspring)])
                    return

                if self.global_best_ind is None or offspring.score > self.global_best_ind.score:
                    self.global_best_ind = offspring

                self.next_pop.append(offspring)

                # Route to specific ClusterUnit (Niche) if applicable
                target_id = from_which_cluster
                if self.is_initialized and target_id is not None and target_id in self.cluster_units:
                    evo_offspring = Evoind(function=offspring, cluster_id=target_id)
                    if hasattr(offspring, 'reflection'):
                        evo_offspring.set_reflection(offspring.reflection)
                    self.cluster_units[target_id].register_individual(evo_offspring)

                self.external_archive.register(offspring)

                if len(self.next_pop) >= self.pop_size:
                    self._manager_pop_management()

            except Exception as e:
                print(f"🚨 [Manager] Error in register_offspring: {e}")
                traceback.print_exc()
                return

    def _manager_pop_management(self):
        """
        Maintains the global population via elitist selection and truncation.
        Triggers niching once the cold-start threshold is met.
        """
        with self._lock:
            candidates = self.population + self.next_pop

            unique_map: Dict[str, Function] = {}

            for func in candidates:
                if func.score is None:
                    continue
                code_key = func.body

                if code_key not in unique_map:
                    unique_map[code_key] = func
                else:
                    if func.score > unique_map[code_key].score:
                        unique_map[code_key] = func

            valid_funcs = list(unique_map.values())

            valid_funcs.sort(key=lambda x: x.score, reverse=True)

            self.population = valid_funcs[:self.pop_size]
            self.next_pop = []
            self.generation += 1

            # Log
            best_score = self.population[0].score if self.population else None
            print(f"[Manager] 📢 Global Population Updated. Current Size: {len(self.population)}")

            if not self.is_initialized and len(self.population) >= self.n_clusters:
                print(f"[Manager] 🚀 Sufficient individuals collected ({len(self.population)} >= {self.n_clusters}). "
                      f"Triggering Initial Clustering...")
                self.initial_population_clustering()

    def _update_global_best(self, population: List[Evoind]):
        """Helper to track the highest scoring function globally."""
        for ind in population:
            if self.global_best_ind is None or ind.function.score > self.global_best_ind.score:
                self.global_best_ind = ind.function

    def debug_status(self):
        """Prints diagnostic information about cluster health and scores."""
        print(f"\n=== Manager Status ===")
        print(f"Global Pop (Funcs): {len(self.population)}, Buffer: {len(self.next_pop)}")

        if self.global_best_ind:
            print(f"Global Best: {self.global_best_ind.score:.4f}")

        if self.is_initialized:
            c_ids, probs = self._calculate_selection_probs()
            for c_id, prob in zip(c_ids, probs):
                unit = self.cluster_units[c_id]
                best = unit.get_best_individual()
                score = best.function.score if best else -999
                print(f"  Cluster {c_id}: Size={len(unit)}, Best={score:.4f}, Prob={prob:.2%}")
        else:
            print("  [Cold Start] Waiting for population buffer to fill...")
