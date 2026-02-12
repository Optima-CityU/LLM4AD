# cluster_manager.py

from __future__ import annotations
import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional
from threading import RLock
import os

# 假设引入了之前的 ClusterUnit 和相关的类
from .clusterunit import ClusterUnit
from llm4ad.base import Function
from .base import Evoind
from codebleu import calc_codebleu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


try:
    from transformers import BertTokenizer, BertModel
    import torch
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"[Warning] Optional dependencies missing: {e}. Feature extraction might fail.")

# 使用BERT模型生成文本嵌入
def get_bert_embeddings(texts: List[str], model_path: str = None) -> np.ndarray:
    """使用BERT模型生成文本嵌入"""
    # 默认路径处理，增强代码移植性
    target_path = model_path if model_path and os.path.exists(model_path) else 'bert-base-uncased'

    try:
        tokenizer = BertTokenizer.from_pretrained(target_path)
        model = BertModel.from_pretrained(target_path)
    except Exception as e:
        print(f"[Error] Failed to load BERT from {target_path}: {e}")
        # 返回随机向量防止程序崩溃 (仅用于调试流程)
        return np.random.rand(len(texts), 768)

    embeddings = []
    # 批量处理建议：如果数据量大，建议分batch处理，这里简化为逐个处理
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
        # 使用[CLS] token的嵌入表示整个句子
        cls_embedding = hidden_states[0, 0, :].numpy()
        embeddings.append(cls_embedding)

    return np.array(embeddings)

def individual_feature(population: List[Evoind],
                       feature_type: Tuple[str, ...] = ('AST',),
                       save_path: str = '',
                       bert_model_path: str = None):
    """
    计算种群特征：AST, random, language, objective
    """
    if not population:
        return

    print(f'[Feature Extraction] Processing feature types: {feature_type}')
    population_size = len(population)
    features = [[] for _ in range(population_size)]

    # 1. AST 特征 (CodeBLEU)
    if 'AST' in feature_type:
        AST = np.zeros((population_size, population_size))
        # 提取所有代码文本
        codes = [ind.function.to_code_without_docstring() for ind in population]

        # 优化：CodeBLEU 计算可能较慢，此处为 O(N^2)
        for i in range(population_size):
            for j in range(i, population_size):  # 利用对称性
                if i == j:
                    score = 1.0
                else:
                    try:
                        cal_result = calc_codebleu([codes[i]], [codes[j]], lang='python',
                                                   weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
                        # 组合语法匹配和数据流匹配
                        score = 0.5 * cal_result['syntax_match_score'] + 0.5 * cal_result['dataflow_match_score']
                    except Exception:
                        score = 0.0  # 容错

                AST[i, j] = score
                AST[j, i] = score

        # 将AST相似度向量作为特征
        for i in range(population_size):
            features[i].extend(AST[i, :].tolist())

    # 2. 自然语言嵌入特征 (BERT)
    if 'language' in feature_type:
        texts = [ind.function.algorithm for ind in population]
        embeddings = get_bert_embeddings(texts, model_path=bert_model_path)
        for i in range(population_size):
            features[i].extend(embeddings[i, :].tolist())

    # 3. 随机特征 (Baseline)
    if 'random' in feature_type:
        random_features = np.random.normal(size=(population_size, 20))
        for i in range(population_size):
            features[i].extend(random_features[i, :].tolist())

    # 4. 目标值特征
    if 'objective' in feature_type:
        for i in range(population_size):
            # 处理 None 的情况
            obj = population[i].function.score if population[i].function.score is not None else 0
            features[i].append(obj)

    # === 特征降维与绑定 ===
    try:
        all_features = np.array(features)
        # 数据清洗：替换 NaN 和 Inf
        all_features = np.nan_to_num(all_features)
        # 标准化
        scaler = StandardScaler()
        all_features = scaler.fit_transform(all_features)
        # PCA 降维 (如果特征维度 > 10)
        n_components = min(10, population_size, all_features.shape[1])
        pca = PCA(n_components=n_components)
        all_features_reduced = pca.fit_transform(all_features)
        # 为每个个体设置特征
        for i, ind in enumerate(population):
            ind.set_feature(all_features_reduced[i]) # 确保 Evoind 有 .feature 属性或 .set_feature 方法
        # === 绘图 (可选) ===
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
        # 兜底：赋予零向量防止后续聚类崩溃
        for ind in population:
            ind.set_feature(np.zeros(10))

class ClusterManager:
    """
    ClusterManager: HIE 算法的宏观调度器。

    数据流说明：
    1. Manager 维护全局 self.population (List[Function])。
    2. register_offspring 接收 Function，存入 self.next_pop (List[Function])。
    3. 当 next_pop 满时，触发 _manager_pop_management，更新 self.population。
    4. 如果未初始化，尝试在 self.population 上进行聚类。
    5. 聚类时，将 Function 包装为 Evoind 进行特征计算和分发。
    6. ClusterUnit 内部维护 List[Evoind]。
    """

    def __init__(self,
                 pop_size: int = 16,
                 n_clusters: int = 4,
                 intra_operators: Tuple[str, ...] = ('re', 'se', 'cc', 'lge'),
                 intra_operators_parent_num: Dict[str, int] = None,
                 intra_operators_frequence: Optional[Dict] = None,
                 use_resource_tilt: bool = True,  # 是否开启资源倾斜
                 resource_tilt_alpha: float = 2.0,  # 倾斜强度 (仅在 True 时生效)

                 bert_model_path: str = None,  # [新增] 指定 BERT 路径
                 debug_flag: bool = False,
                 ):

        self.debug_flag = debug_flag
        self.pop_size = pop_size
        self.n_clusters = n_clusters
        self.bert_model_path = bert_model_path

        # --- Operators Config ---
        # 1. 设置父代数量需求 (Default)
        default_parent_num = {'re': 1, 'se': 1, 'cc': 1, 'lge': 1}
        self.intra_cluster_operators_parent_num = intra_operators_parent_num or default_parent_num

        # 2. 设置算子频率 (Frequency)
        # 如果未提供，默认所有算子频率为 1
        freq_config = intra_operators_frequence or {op: 1 for op in intra_operators}

        # 3. [关键逻辑] 生成加权后的算子元组
        # 例如: {'re': 2, 'cc': 1} -> ('re', 're', 'cc')
        # 这样 ClusterUnit 轮询时，re 出现的概率就是 cc 的两倍
        expanded_ops = []
        for op in intra_operators:
            freq = freq_config.get(op, 1)
            expanded_ops.extend([op] * freq)
        self.intra_cluster_operators = tuple(expanded_ops)

        # 校验
        if intra_operators_parent_num:
            missing_keys = set(intra_operators) - set(intra_operators_parent_num.keys())
            if missing_keys:
                raise ValueError(f"intra_cluster_operators_parent_num is missing entries for: {sorted(missing_keys)}")

        print(f'[Manager] Operators Expanded Sequence: {self.intra_cluster_operators}')
        print(f'[Manager] Parent Requirements: {self.intra_cluster_operators_parent_num}')

        # 资源调度配置
        self.use_resource_tilt = use_resource_tilt
        self.resource_tilt_alpha = resource_tilt_alpha

        # 核心存储
        self.cluster_units: Dict[int, ClusterUnit] = {}
        self.global_best_ind: Optional[Evoind] = None

        self.population: List[Function] = []
        self.next_pop: List[Function] = []

        self._lock = RLock()

        # 记录所有 ClusterUnit 产生的最近一次操作，用于后续注册
        self.last_selected_cluster_id = None

    def initial_population_clustering(self, feature_type=('AST',)):
        """
        初始化流程：
        1. 计算特征 (individual_feature)
        2. K-Means 聚类
        3. 创建 ClusterUnits
        """
        print(f"[Manager] Initializing {len(initial_pop)} individuals into {self.n_clusters} clusters...")

        # 1. 计算特征 (In-place 修改 ind.feature)
        # 这里的 save_path 可以在 debug 模式下设置
        save_path = "init_debug" if self.debug_flag else ""
        individual_feature(initial_pop, feature_type=feature_type,
                           save_path=save_path, bert_model_path=self.bert_model_path)

        # 2. 准备聚类数据
        features = []
        valid_pop = []
        for ind in initial_pop:
            if hasattr(ind, 'feature') and ind.feature is not None:
                features.append(ind.feature)
                valid_pop.append(ind)
            else:
                # 如果计算特征失败，暂时给个随机或零向量
                print(f"[Warning] Individual {ind} has no feature, assigning zeros.")
                features.append(np.zeros(10))
                valid_pop.append(ind)

        features = np.array(features)

        # 3. 执行 K-Means
        if len(features) >= self.n_clusters:
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
        else:
            print("[Warning] Population size < n_clusters, using random assignment.")
            labels = np.random.randint(0, self.n_clusters, size=len(valid_pop))

        # 4. 创建 ClusterUnits
        self.cluster_units.clear()
        grouped_pop = {i: [] for i in range(self.n_clusters)}

        for ind, label in zip(valid_pop, labels):
            ind.cluster_id = label
            grouped_pop[label].append(ind)

        for c_id in range(self.n_clusters):
            unit_pop = grouped_pop[c_id]
            # [关键] 将加权后的 operator tuple 传给 Unit
            new_unit = ClusterUnit(
                cluster_id=c_id,
                max_pop_size=self.pop_size,
                intra_operators=self.intra_cluster_operators,
                intra_operators_parent_num=self.intra_cluster_operators_parent_num,
                pop=unit_pop
            )
            self.cluster_units[c_id] = new_unit

        # 更新全局最优
        self._update_global_best(valid_pop)
        print("[Manager] Clustering and Initialization finished.")

    def _calculate_selection_probs(self) -> Tuple[List[int], List[float]]:
        """
        计算被选概率: 资源倾斜 vs 均匀分布
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
                    # 给一个非常小的值，保证有概率（e^x > 0），但概率极低
                    # 注意：如果 objective 可能为负数，这个值要比最小可能的 objective 还要小
                    scores.append(-1e9)

                    # 策略 A: 均匀分布
        if not self.use_resource_tilt:
            n = len(cluster_ids)
            return cluster_ids, [1.0 / n] * n

        # 策略 B: 资源倾斜 (Softmax)
        scores_arr = np.array(scores)
        # 防止溢出
        exp_scores = np.exp((scores_arr - np.max(scores_arr)) * self.resource_tilt_alpha)
        sum_exp = np.sum(exp_scores)

        if sum_exp == 0:
            probs = [1.0 / len(cluster_ids)] * len(cluster_ids)
        else:
            probs = exp_scores / sum_exp

        return cluster_ids, probs

    def select_parent(self) -> Tuple[List[Function], str, int]:
        """
        外部调用的主接口。
        """
        with self._lock:
            # 1. 依概率选择 Cluster
            cluster_ids, probs = self._calculate_selection_probs()
            chosen_c_id = np.random.choice(cluster_ids, p=probs)
            target_unit = self.cluster_units[chosen_c_id]
            self.last_selected_cluster_id = chosen_c_id

            # 2. 调用 Unit 内部轮询
            parents, operator, need_external = target_unit.selection(
                help_inter=False,
            )

            # 3. 处理外部协作 (Crossover 补位)
            if need_external:
                if operator == 'cc':
                    # 获取所有其他 Unit
                    other_units = [u for uid, u in self.cluster_units.items() if uid != chosen_c_id and len(u) > 0]
                    helper_func = None

                    if other_units:
                        # 随机选一个外簇
                        helper_unit = random.choice(other_units)
                        # 从该外簇中选一个最好的 (Top-1)
                        # Unit.selection 返回 (funcs, op, bool)
                        helper_parents, _, _ = helper_unit.selection(
                            help_inter=True,  # 标记为 Helper 模式
                            mode='tournament',
                            help_number=1
                        )
                        if helper_parents:
                            helper_func = helper_parents[0]
                    # 如果找不到外援 (比如只有一个簇)，尝试用全局最优
                    if helper_func is None and self.global_best_ind:
                        helper_func = self.global_best_ind.function

                    if helper_func:
                        parents.append(helper_func)

                # === Case 2: Layered/Global Evolution (lge) ===
                # 需要注入全局视角信息：[Global Best, Cluster Best]
                elif operator == 'lge':
                    # 1. 注入 Global Best
                    if self.global_best_ind:
                        # 查重：避免重复添加
                        if not any(f.algorithm == self.global_best_ind.function.algorithm for f in parents):
                            parents.append(self.global_best_ind.function)

                    # 2. 注入 Cluster Best (如果它不在 parents 里)
                    cluster_best = target_unit.get_best_individual()
                    if cluster_best:
                        if not any(f.algorithm == cluster_best.function.algorithm for f in parents):
                            parents.append(cluster_best.function)

            return parents, operator, chosen_c_id

    def register_offspring(self, offspring: Function, from_which_cluster: int = None):
        """注册新个体"""
        with self._lock:
            target_id = cluster_id if cluster_id is not None else self.last_selected_cluster_id

            if target_id is not None and target_id in self.cluster_units:
                success = self.cluster_units[target_id].register_individual(offspring)
                if success:
                    if self.global_best_ind is None or offspring.function.score > self.global_best_ind.function.score:
                        self.global_best_ind = offspring
            else:
                print(f"[Manager] Warning: Unknown cluster id {target_id}, offspring discarded.")

    def _update_global_best(self, population: List[Evoind]):
        for ind in population:
            if self.global_best_ind is None or ind.function.score > self.global_best_ind.function.score:
                self.global_best_ind = ind

    def debug_status(self):
        """打印状态"""
        c_ids, probs = self._calculate_selection_probs()
        print(f"\n=== Manager Status (Tilt={self.use_resource_tilt}) ===")
        print(f"Global Best: {self.global_best_ind.cluster_score if self.global_best_ind else 'None'}")
        for c_id, prob in zip(c_ids, probs):
            unit = self.cluster_units[c_id]
            best = unit.get_best_individual()
            score = best.cluster_score if best else -999
            print(f"Cluster {c_id}: Size={len(unit)}, Best={score:.4f}, Prob={prob:.2%}")