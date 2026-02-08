from __future__ import annotations

import math
from threading import Lock
from typing import List
import numpy as np

from ...base import *


class Population:
    def __init__(self, pop_size, generation=0, pop: List[Function] | Population | None = None):
        if pop is None:
            self._population = []
        elif isinstance(pop, list):
            self._population = pop
        else:
            self._population = pop._population

        self._pop_size = pop_size
        self._lock = Lock()
        self._next_gen_pop = []
        self._generation = generation
        self._pop_register_number = 1

    def __len__(self):
        return len(self._population)

    def __getitem__(self, item) -> Function:
        return self._population[item]

    def __setitem__(self, key, value):
        self._population[key] = value

    @property
    def population(self):
        return self._population

    @property
    def generation(self):
        return self._generation

    def register_function(self, func: Function):
        # in population initialization, we only accept valid functions
        if self._generation == 0 and func.score is None:
            return
        # if the score is None, we still put it into the population,
        # we set the score to '-inf'
        if func.score is None:
            func.score = float('-inf')
        try:
            self._lock.acquire()
            if self.has_duplicate_function(func):
                func.score = float('-inf')
            func.pop_register_number = self._pop_register_number
            self._pop_register_number += 1
            # register to next_gen
            self._next_gen_pop.append(func)
            # update: perform survival if reach the pop size
            if len(self._next_gen_pop) >= self._pop_size:
                pop = self._population + self._next_gen_pop
                pop = sorted(pop, key=lambda f: f.score, reverse=True)
                self._population = pop[:self._pop_size]
                self._next_gen_pop = []
                self._generation += 1
        except Exception as e:
            return
        finally:
            self._lock.release()

    def has_duplicate_function(self, func: str | Function) -> bool:
        for f in self._population:
            if str(f) == str(func) or func.score == f.score:
                return True
        for f in self._next_gen_pop:
            if str(f) == str(func) or func.score == f.score:
                return True
        return False

    # def selection(self) -> Function:
    #     funcs = [f for f in self._population if not math.isinf(f.score)]
    #     func = sorted(funcs, key=lambda f: f.score, reverse=True)
    #     p = [1 / (r + len(func)) for r in range(len(func))]
    #     p = np.array(p)
    #     p = p / np.sum(p)
    #     return np.random.choice(func, p=p, replace=False)

    def selection(self, number=1, best_must=False, mode='exp', pressure=0.5) -> List[Function]:
        """
        :param number: 需要选择的个体数量
        :param best_must: 是否必须包含最优解
        :param mode: 'exp' (指数) 或 'linear' (线性)
        :param pressure: 选择压力系数 (仅用于 exp 模式). 值越大，越倾向于只选头部个体。建议 0.1 - 1.0
        :return: 选择出的个体列表
        """
        valid_funcs = [f for f in self._population if not math.isinf(f.score)]
        if not valid_funcs:
            return []
        sorted_funcs = sorted(valid_funcs, key=lambda f: f.score, reverse=True)
        n = len(sorted_funcs)

        ranks = np.arange(n)

        if mode == 'exp':
            p = np.exp(-pressure * ranks)
        elif mode == 'linear':
            p = 1.0 / (ranks + n)
        else:
            # 默认均匀分布
            p = np.ones(n)

        p = p / np.sum(p)

        # 4. 执行选择
        # 注意：如果 number > n 且 replace=False，会报错。
        # 这里做一个简单的容错处理
        use_replace = False
        if number > n:
            use_replace = True  # 个体不够时，必须允许重复抽样

        # 关键修复：这里必须从 sorted_funcs 中抽取，因为 p 是对应 rank 的
        selected = list(np.random.choice(sorted_funcs, size=number, p=p, replace=use_replace))

        # 5. 精英策略 (Elitism)
        if best_must:
            best_ind = sorted_funcs[0]  # 关键修复：从排序后的列表中取第一个

            # 检查最优解是否已经被选中
            # 注意：这里对比的是对象引用。如果对象重写了 __eq__，逻辑可能不同
            if best_ind not in selected:
                # 替换掉最后一个被选中的（通常概率最低），或者是随机替换一个
                selected[-1] = best_ind

        return selected
