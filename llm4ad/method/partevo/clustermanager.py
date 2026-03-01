# cluster_manager.py

from __future__ import annotations
import numpy as np
import random
import traceback
import os
from typing import List, Dict, Tuple, Any, Optional
from threading import RLock

from .clusterunit import ClusterUnit
from llm4ad.base import Function
from .base import Evoind
from codebleu import calc_codebleu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt


# 使用BERT模型生成文本嵌入
def get_bert_embeddings(texts: List[str], model_path: str = None) -> np.ndarray:
    """使用BERT模型生成文本嵌入"""
    try:
        from transformers import BertTokenizer, BertModel
        import torch
    except ImportError as e:
        print(f"⚠️ [Warning in clustermanager.py] Optional dependencies missing: {e}. Feature extraction might fail.")

    # 默认路径处理，增强代码移植性
    target_path = model_path if model_path and os.path.exists(model_path) else 'bert-base-uncased'

    try:
        tokenizer = BertTokenizer.from_pretrained(target_path)
        model = BertModel.from_pretrained(target_path)
    except Exception as e:
        print(f"❌ [Error] Failed to load BERT from {target_path}: {e}")
        # 返回随机向量防止程序崩溃 (仅用于调试流程)
        return np.random.rand(len(texts), 768)

    embeddings = []
    # 批量处理建议：如果数据量大，建议分batch处理，这里简化为逐个处理
    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
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
    计算种群特征：AST, random, language, objective
    """
    if not population:
        return

    print(f'[Feature Extraction] Processing feature types: {feature_type}')
    population_size = len(population)
    features = [[] for _ in range(population_size)]

    # 1. AST 特征 (CodeBLEU)
    if 'ast' in feature_type:
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
        texts = [ind.function.to_code_without_docstring() for ind in population]
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
            ind.set_feature(all_features_reduced[i])  # 确保 Evoind 有 .feature 属性或 .set_feature 方法
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
                 intra_operators_frequency: Optional[Dict] = None,
                 use_resource_tilt: bool = False,  # 是否开启资源倾斜
                 resource_tilt_alpha: float = 2.0,  # 倾斜强度 (仅在 True 时生效)
                 feature_type: Tuple[str, ...] = ('ast',),

                 bert_model_path: str = None,  # [新增] 指定 BERT 路径
                 debug_flag: bool = False,
                 ):

        self.debug_flag = debug_flag
        self.pop_size = pop_size
        self.generation = 0
        self.n_clusters = n_clusters
        self.bert_model_path = bert_model_path

        self.feature_type = feature_type

        # [Fix] 初始化状态标志
        self.is_initialized = False

        # --- Operators Config ---
        # 1. 设置父代数量需求 (Default)
        default_parent_num = {'re': 1, 'se': 1, 'cc': 1, 'lge': 1}
        self.intra_cluster_operators_parent_num = intra_operators_parent_num or default_parent_num

        # 2. 设置算子频率 (Frequency)
        # 如果未提供，默认所有算子频率为 1
        freq_config = intra_operators_frequency or {op: 1 for op in intra_operators}

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
        self.global_best_ind: Optional[Function] = None

        self.population: List[Function] = []
        self.next_pop: List[Function] = []

        self._lock = RLock()

    def initial_population_clustering(self):
        """
        基于 self.population (Function) 进行聚类初始化。
        步骤：Function -> Evoind -> Feature -> KMeans -> ClusterUnit
        """
        # 1. 临时转换 Function -> Evoind 以便计算特征
        # 只取有分数的
        valid_funcs = [f for f in self.population if f.score is not None]

        if len(valid_funcs) < self.n_clusters:
            print(f"❌ [Manager] Not enough individuals to cluster ({len(valid_funcs)}). Waiting...")
            return

        print(f"⏳ [Manager] Initializing Clustering with {len(valid_funcs)} individuals...")

        # 创建临时 Evoind 列表用于聚类计算
        temp_evo_pop = [Evoind(function=f) for f in valid_funcs]

        # 2. 计算特征 (In-place 修改 temp_evo_pop 中 Evoind 的 feature)
        save_path = "init_debug" if self.debug_flag else ""
        individual_feature(temp_evo_pop, feature_type=self.feature_type,
                           save_path=save_path, bert_model_path=self.bert_model_path)

        # 3. 准备聚类数据
        features = []
        for ind in temp_evo_pop:
            if hasattr(ind, 'feature') and ind.feature is not None and len(ind.feature) > 0:
                features.append(ind.feature)
            else:
                features.append(np.zeros(10))
        features = np.array(features)

        # 4. K-Means
        try:
            if len(features) >= self.n_clusters:
                kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features)
            else:
                labels = np.random.randint(0, self.n_clusters, size=len(temp_evo_pop))
        except Exception as e:
            print(f"⚠️ [Manager] KMeans failed: {e}. Using random assignment.----")
            labels = np.random.randint(0, self.n_clusters, size=len(temp_evo_pop))

        # 5. 分发并创建 ClusterUnit
        self.cluster_units.clear()
        grouped_pop = {i: [] for i in range(self.n_clusters)}

        for ind, label in zip(temp_evo_pop, labels):
            ind.cluster_id = label  # 标记
            grouped_pop[label].append(ind)  # 注意：这里存入 Cluster 的是 Evoind

        for c_id in range(self.n_clusters):
            unit_pop = grouped_pop[c_id]
            new_unit = ClusterUnit(
                cluster_id=c_id,
                max_pop_size=self.pop_size,  # Cluster 内部容量
                intra_operators=self.intra_cluster_operators,
                intra_operators_parent_num=self.intra_cluster_operators_parent_num,
                pop=unit_pop  # 传入 List[Evoind]
            )
            self.cluster_units[c_id] = new_unit

        # 6. 更新全局最优 (基于 Evoind)
        self._update_global_best(temp_evo_pop)

        self.is_initialized = True
        print("🎉 [Manager] Clustering Finished. System Online. 👍")

    def _calculate_selection_probs(self) -> Tuple[List[int], List[float]]:
        """计算 Cluster 被选概率"""
        cluster_ids = []
        scores = []
        for c_id, unit in self.cluster_units.items():
            cluster_ids.append(c_id)
            if self.use_resource_tilt:
                best_ind = unit.get_best_individual()  # 返回 Evoind
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
        选择父代 (返回 Function 列表)。
        """
        with self._lock:
            # === Case A: 冷启动 (未聚类) ===
            if not self.is_initialized:
                # 从全局 population (Function) 中选
                pool = self.population + self.next_pop
                valid_pool = [f for f in pool if f.score is not None]

                if not valid_pool:
                    return [], 'error', -1  # 无种子，返回空让外部做初始化

                # 随机选一个做 mutation
                parent = random.choice(valid_pool)
                return [parent], 're', -1

            # === Case B: 正常进化 (已聚类) ===
            cluster_ids, probs = self._calculate_selection_probs()
            if not cluster_ids:
                return [], 'error', -1  # 异常兜底

            chosen_c_id = np.random.choice(cluster_ids, p=probs)
            target_unit = self.cluster_units[chosen_c_id]

            # Unit 返回 List[Function]
            parents, operator, need_external = target_unit.selection(help_inter=False)

            # === 外部协作处理 ===
            if need_external:
                # 1. Crossover (cn)
                if operator == 'cn':
                    # 找外援 Cluster
                    other_units = [u for uid, u in self.cluster_units.items() if uid != chosen_c_id and len(u) > 0]
                    helper_func = None

                    if other_units:
                        helper_unit = random.choice(other_units)
                        # Helper 模式选 Top 1 Function
                        h_parents, _, _ = helper_unit.selection(help_inter=True, mode='top', help_number=1)
                        if h_parents:
                            helper_func = h_parents[0]

                    # 兜底：用 Global Best
                    if not helper_func and self.global_best_ind:
                        helper_func = self.global_best_ind

                    if helper_func:
                        parents.append(helper_func)

                # 2. God View (lge)
                elif operator == 'lge':
                    # 注入 Global Best Function
                    if self.global_best_ind:
                        if not any(f.body == self.global_best_ind.body for f in parents):
                            parents.append(self.global_best_ind)

                    # 注入 Cluster Best Function
                    # 注意：get_best_individual 返回 Evoind
                    cluster_best_evo = target_unit.get_best_individual()  # Evoind
                    if cluster_best_evo:
                        if not any(f.body == cluster_best_evo.function.body for f in parents):
                            parents.append(cluster_best_evo.function)

            return parents, operator, chosen_c_id

    def _should_reject_duplicate(self, offspring: Function) -> bool:
        """
        检查是否是重复个体。
        Returns:
            True  -> 拒绝（是无意义的低分重复）
            False -> 接受（是新代码，或者是分数更高的旧代码）
        """
        target_code = offspring.body
        for existing_func in self.population:
            if existing_func.body == target_code:
                # 发现代码重复
                if offspring.score > existing_func.score:
                    # 新的分数更高！这是一个"有价值的重复"
                    print(
                        f"✨ [Manager] Found better score for existing code: {existing_func.score:.4f} -> {offspring.score:.4f}")
                    return False  # 允许进入
                else:
                    # 分数没变或更低，拒绝
                    return True
        return False

    def register_function(self, offspring: Function, from_which_cluster: int = None):
        """
        注册新个体 (Function)。
        流程：
        1. 入 Manager Buffer (Function)。
        2. 若已初始化，包装成 Evoind 入 Cluster。
        3. Check Buffer Size -> Management.
        """
        with self._lock:
            # 1. 基础检查：无效分数直接丢弃
            if offspring.score is None:
                return

            try:
                if self._should_reject_duplicate(offspring):
                    print('♻️ [Info] The generated offspring is a duplicate of an existing individual. Skipping registration.')
                    return

                # === 1. Update Global Best (Fast Path) ===
                if self.global_best_ind is None or offspring.score > self.global_best_ind.score:
                    self.global_best_ind = offspring

                # === 2. Manager Buffer (Function) ===
                self.next_pop.append(offspring)

                target_id = from_which_cluster
                if self.is_initialized and target_id is not None and target_id in self.cluster_units:
                    evo_offspring = Evoind(function=offspring, cluster_id=target_id)
                    if hasattr(offspring, 'reflection'):
                        evo_offspring.set_reflection(offspring.reflection)
                    self.cluster_units[target_id].register_individual(evo_offspring)

                # 6. Trigger Global Management
                if len(self.next_pop) >= self.pop_size:
                    self._manager_pop_management()

            except Exception as e:
                print(f"🚨 [Manager] Error in register_offspring: {e}")
                traceback.print_exc()
                return

    def _manager_pop_management(self):
        """
        Manager 全局种群优胜劣汰逻辑。

        触发时机：
        当 register_offspring 检测到 next_pop 缓冲区积攒了一定数量的新个体后触发。

        流程：
        1. 合并：Current Population + Offspring Buffer
        2. 清洗：去除无效分数 (None)
        3. 去重：相同代码保留高分者
        4. 排序：按分数降序
        5. 截断：保留 Top-K (pop_size)
        6. 初始化检查：如果处于冷启动阶段且凑够了人，触发聚类。
        """
        # 注意：此方法通常在 register_offspring 的锁内被调用，
        # 但为了安全起见，再次确认锁也没问题 (RLock 支持重入)
        with self._lock:
            # 1. 合并当前种群和新产生的种群
            candidates = self.population + self.next_pop

            # 2. 去重与清洗 (Deduplication & Cleaning)
            unique_map: Dict[str, Function] = {}

            for func in candidates:
                # 过滤掉无效分数的个体
                if func.score is None:
                    continue

                # 使用 algorithm (代码字符串) 作为去重键
                # 确保同一份代码只保留分数最高的那个版本 (以防随机性导致的波动)
                code_key = func.body

                if code_key not in unique_map:
                    unique_map[code_key] = func
                else:
                    if func.score > unique_map[code_key].score:
                        unique_map[code_key] = func

            # 转回列表
            valid_funcs = list(unique_map.values())

            # 3. 排序 (Sorting) - 分数从高到低
            valid_funcs.sort(key=lambda x: x.score, reverse=True)

            # 4. 截断 (Truncation) - 保持种群规模恒定
            # 如果是冷启动初期，可能还不够 pop_size，那就全保留
            self.population = valid_funcs[:self.pop_size]
            self.next_pop = []
            self.generation += 1

            # Log
            best_score = self.population[0].score if self.population else None
            print(f"[Manager] Global Population Updated. Current Size: {len(self.population)}\n"
                  f"(Best: {self.population[0].score:.4f} if {len(self.population)}>0 else None)")

            # 6. [关键] 尝试触发初始化 (Cold Start -> Clustering)
            # 如果还没初始化，且当前有效个体数已经超过了聚类所需的最小簇数
            if not self.is_initialized and len(self.population) >= self.n_clusters:
                print(f"[Manager] 🚀 Sufficient individuals collected ({len(self.population)} >= {self.n_clusters}). "
                      f"Triggering Initial Clustering...")
                self.initial_population_clustering()

    def _update_global_best(self, population: List[Evoind]):
        """入参是 List[Evoind]"""
        for ind in population:
            if self.global_best_ind is None or ind.function.score > self.global_best_ind.score:
                self.global_best_ind = ind.function

    def debug_status(self):
        """打印状态"""
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
