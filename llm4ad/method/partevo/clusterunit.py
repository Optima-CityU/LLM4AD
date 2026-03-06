# cluster_unit.py

from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple, Literal, Optional, Dict, Any
from llm4ad.base import Function
from .base import Evoind
from threading import RLock
import itertools
import random


class ClusterUnit:
    """
    一个算法池单元，管理一个特定算法分支的算法子种群。

    职责：
    - 为本簇选择父代个体。
    - 管理本簇的种群（注册、优胜劣汰）。
    """

    # === [新增] 算子对应的默认选择策略 ===
    OPERATOR_SELECTION_MAP = {
        're': 'exp',  # 改良：锦标赛，选比较好的但保留随机性
        'se': 'tournament',  # 粒子群：同上
        'cc': 'tournament',  # 交叉：同上
        'lge': 'random'  # 上帝视角/全局探索：均匀随机，给所有个体机会
    }

    def __init__(self, cluster_id: int,
                 max_pop_size: int,
                 intra_operators: Tuple[str, ...],  # ('re', 'se', 'cc', 'lge')
                 intra_operators_parent_num: Dict[str, int],
                 pop: List[Evoind] | ClusterUnit | None = None,
                 ):

        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop.copy()
        elif hasattr(pop, '_population'):
            self._population = pop._population.copy()
        else:
            self._population = []

        self.cluster_id = cluster_id
        self._max_pop_size = max_pop_size

        self.cumulative_Non_improvement_count = 0  # 累计停滞次数
        self._history_best_score = float('-inf')

        self._lock = RLock()

        self.intra_operators = intra_operators
        # 默认配置：cc 只需本簇出 1 人（另 1 人由外部提供），lge 通常不需要父代(视具体prompt而定，这里设为0或1均可)
        # 如果 lge 需要 1 个父代作为种子，这里设为 1
        self.intra_operators_parent_num = intra_operators_parent_num or {'re': 1, 'se': 1, 'cc': 1, 'lge': 1}
        self._intra_operators_iterator = itertools.cycle(self.intra_operators)

    def __len__(self):
        with self._lock: return len(self._population)

    def __getitem__(self, item) -> Evoind:
        with self._lock: return self._population[item]

    def __setitem__(self, key, value):
        with self._lock: self._population[key] = value

    @property
    def population(self):
        with self._lock: return self._population.copy()

    def _calculate_dynamic_scores(self, population: List[Evoind],
                                  help_inter: bool,
                                  target_instances: Optional[List[Any]]) -> List[Tuple[Evoind, float]]:
        """
        [核心算分逻辑]
        策略：
        1. 默认情况：使用 ind.function.score (综合均值/总分)。
        2. 特殊情况 (扩展功能)：仅当 (我是Helper) AND (外界指定了target_instances) 时，
           才去计算针对性分数。
        """
        candidates_with_score = []
        for ind in population:
            # === 1. 默认基准：综合分 ===
            eff_score = ind.function.score

            # === 2. 扩展逻辑：仅在明确指定 Target 时覆盖 ===
            if help_inter and target_instances:
                score_sum = 0
                valid_cnt = 0
                if ind.function.all_ins_performance:
                    for ins_id in target_instances:
                        perf = ind.function.all_ins_performance.get(ins_id)
                        if perf and perf.get('score') is not None:
                            s = perf['score']
                            if s != float('-inf'):
                                score_sum += s
                                valid_cnt += 1

                # 如果指定了 Target，但该个体完全没跑过这些 Target，
                # 视情况给 -inf 或 0 (取决于是否允许未测试个体当专家)
                eff_score = score_sum if valid_cnt > 0 else float('-inf')

            # === 3. 过滤无效分数 ===
            if eff_score is not None and not math.isinf(eff_score):
                candidates_with_score.append((ind, eff_score))

        return candidates_with_score

    def selection(self, existing_functions: Optional[List[Function]] = None,
                  best_must: bool = False,
                  mode: str = None,  # [修改] 默认为 None，表示自动推断
                  help_inter: bool = False,
                  help_number: int = 1,
                  tournament_k: int = 3,
                  target_instances: Optional[List[Any]] = None
                  ) -> Tuple[List[Function], str | None, bool]:
        """
        统一选择接口。

        Returns:
            (选出的父代列表, 当前轮到的算子名称, 是否需要外部协作)
        """
        number = 0
        current_operator = None
        need_external_help = False

        with self._lock:
            # === 步骤 0: 确定角色与算子 ===
            if help_inter:
                # [Helper 模式]: 只出人，不进化，不轮询算子
                number = help_number
                current_operator = None
                need_external_help = False
                # Helper 模式下，如果没有指定 mode，默认使用 'tournament'
                if mode is None:
                    mode = 'tournament'
            else:
                # [Requester 模式]: 自身进化，轮询算子
                current_operator = next(self._intra_operators_iterator)
                number = self.intra_operators_parent_num.get(current_operator, 1)

                # 如果算子是 cc (Crossover)，需要外部提供第 2 个父代
                if current_operator == 'cc':
                    need_external_help = True
                elif current_operator == 'lge':
                    need_external_help = True

                # === [关键] 动态策略选择 ===
                # 如果外部没有强制指定 mode，则根据算子类型自动选择
                if mode is None:
                    mode = self.OPERATOR_SELECTION_MAP.get(current_operator, 'tournament')

            # === 步骤 1: 全局过滤 ===
            full_population = self._population.copy()
            valid_pop = full_population

            if existing_functions:
                existing_ids = {id(func) for func in existing_functions}
                valid_pop = [ind for ind in full_population if id(ind.function) not in existing_ids]

            # 鲁棒性回退：如果过滤后没人了，用回全集
            if not valid_pop and full_population:
                valid_pop = full_population

            if not valid_pop:
                return [], current_operator, False

            # === 步骤 2: 计算动态分数 ===
            candidates_with_score = self._calculate_dynamic_scores(valid_pop, help_inter, target_instances)

            # === 步骤 3: 数量不足时的回退机制 ===
            if len(candidates_with_score) < number:
                # 尝试从原始种群（忽略 existing 过滤）里捞人
                candidates_with_score = self._calculate_dynamic_scores(full_population, help_inter, target_instances)

                if len(candidates_with_score) < number:
                    return [], current_operator, False

            # === 步骤 4: 核心选择策略 ===
            selected_individuals = []

            # A. Top-N (贪婪)
            if mode == 'top':
                candidates_with_score.sort(key=lambda x: x[1], reverse=True)
                selected_individuals = [x[0] for x in candidates_with_score[:number]]

            # B. 锦标赛 (Tournament)
            elif mode == 'tournament':
                pool_indices = list(range(len(candidates_with_score)))
                for _ in range(number):
                    if not pool_indices: break
                    k = min(len(pool_indices), tournament_k)
                    chosen_indices = np.random.choice(pool_indices, size=k, replace=False)
                    # 选动态分数最高的
                    best_idx = max(chosen_indices, key=lambda i: candidates_with_score[i][1])
                    selected_individuals.append(candidates_with_score[best_idx][0])
                    pool_indices.remove(best_idx)

            # C. [新增] 均匀随机 (Random) - 适用于 lge 等需要广泛探索的算子
            elif mode == 'random':
                if len(candidates_with_score) >= number:
                    chosen_tuples = random.sample(candidates_with_score, number)
                    selected_individuals = [t[0] for t in chosen_tuples]
                else:
                    selected_individuals = [t[0] for t in candidates_with_score]

            # D. 概率采样 (Roulette/Linear/Exp)
            elif mode in ['roulette', 'linear', 'exp']:
                candidates_with_score.sort(key=lambda x: x[1], reverse=True)
                sorted_inds = [x[0] for x in candidates_with_score]

                if mode == 'exp':
                    p = np.exp(-np.arange(len(sorted_inds)) / 2.0)
                elif mode == 'linear':
                    weights = np.arange(len(sorted_inds), 0, -1)
                    p = weights
                else:
                    # 简单 Softmax 防止负分报错
                    raw_scores = np.array([x[1] for x in candidates_with_score])
                    p = np.exp(raw_scores - np.max(raw_scores))

                probabilities = p / np.sum(p)
                try:
                    selected_individuals = list(
                        np.random.choice(sorted_inds, size=number, p=probabilities, replace=False))
                except ValueError:
                    selected_individuals = sorted_inds[:number]

            else:
                # 默认 Top-N
                candidates_with_score.sort(key=lambda x: x[1], reverse=True)
                selected_individuals = [x[0] for x in candidates_with_score[:number]]

            # === 步骤 5: Best Must (精英保留) ===
            if best_must and len(selected_individuals) > 0 and len(valid_pop) > 0:
                global_best = max(valid_pop, key=lambda ind: ind.function.score)
                if not any(ind is global_best for ind in selected_individuals):
                    selected_individuals[-1] = global_best
        return [ind.function for ind in selected_individuals], current_operator, need_external_help

    def has_duplicate_function(self, func: str | Evoind) -> bool:
        """检查是否有代码完全相同的个体"""
        target_code = func.function.body if isinstance(func, Evoind) else str(func)
        for ind in self._population:
            if ind.function.body == target_code:
                return True
        return False

    def register_individual(self, new_individual: Evoind):
        """
        注册新个体：增加“等分替换”逻辑，确保种群多样性。
        """
        with self._lock:
            new_score = new_individual.function.score
            if new_score is None or math.isinf(new_score):
                return False

            # === 1. 等分或同代码替换逻辑 ===
            replaced = False
            for i, ind in enumerate(self._population):
                # 检查代码是否相同 或 分数是否几乎一致 (使用 epsilon 容差)
                is_same_code = (ind.function.body == new_individual.function.body)
                is_same_score = (abs(ind.function.score - new_score) < 1e-9)

                if is_same_code or is_same_score:
                    # 只有当新个体分数更高，或者分数一样但它是“新鲜血液”时才替换
                    if new_score >= ind.function.score:
                        # 原地替换
                        self._population[i] = new_individual
                        replaced = True
                        # 如果分数有提升，重置停滞计数
                        if new_score > self._history_best_score + 1e-6:
                            self._history_best_score = new_score
                            self.cumulative_Non_improvement_count = 0
                        break
                    else:
                        # 如果新个体分数更低，直接拒绝
                        return False

            # === 2. 如果不是替换旧个体，则作为新成员加入 ===
            if not replaced:
                if new_score > self._history_best_score + 1e-6:
                    self._history_best_score = new_score
                    self.cumulative_Non_improvement_count = 0
                else:
                    self.cumulative_Non_improvement_count += 1

                self._population.append(new_individual)

            # === 3. 规模维护 ===
            if len(self._population) > self._max_pop_size:
                self.do_pop_management()

            return True

    def do_pop_management(self):
        """
        对种群进行管理，包括去重和淘汰。
        """
        if not self._population:
            return

        # 使用 dict 以代码 body 为键进行最终去重，确保万无一失
        unique_map: Dict[str, Evoind] = {}
        for ind in self._population:
            code_key = ind.function.body
            if code_key not in unique_map or ind.function.score >= unique_map[code_key].function.score:
                unique_map[code_key] = ind

        # 转换回列表并按分数降序排列
        sorted_pop = list(unique_map.values())
        sorted_pop.sort(key=lambda ind: ind.function.score, reverse=True)

        # 截断到最大容量
        self._population = sorted_pop[:self._max_pop_size]

    def get_best_individual(self) -> Evoind | None:
        """返回种群中得分最高的个体。"""
        with self._lock:
            if not self._population: return None
            return max(self._population, key=lambda ind: ind.function.score)