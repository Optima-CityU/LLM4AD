import random
from threading import RLock
from typing import List, Tuple

from ...base import Function

class ExternalArchive:
    """
    ExternalArchive (全局外部记忆库)

    作用于 ClusterManager 层级，负责记录全局进化历史。
    采用“瀑布流 (Waterfall)”和“难负样本 (Hard Negatives)”机制：
    1. 确保 Elite 库纯洁性，仅保留 Global Top-K。
    2. 被 Elite 淘汰的高分个体，降级进入 Failure (次优) 库。
    3. 为 SE (Summary/Semantic Exploration) 算子提供高对比度样本。
    """

    def __init__(self, max_elites: int = 5, max_failures: int = 30):
        self.max_elites = max_elites
        self.max_failures = max_failures

        # 降序排列的 Top Tier (全局最强的前 N 个)
        self.elites: List[Function] = []

        # 降序排列的 Second Tier (被挤出的前精英，或非常有潜力的高分失败者)
        self.failures: List[Function] = []

        self._lock = RLock()

    def register(self, func: Function):
        """主入口：审查并注册新个体"""
        if func.score is None:
            return

        with self._lock:
            target_code = func.body

            # === 1. 全局去重 & 动态升降级逻辑 ===

            # 1.1 检查是否已在 Elites 中
            for i, e in enumerate(self.elites):
                if e.body == target_code:
                    if func.score > e.score:
                        # 评估存在随机性时，同代码跑出更高分，直接更新 (不作为 failure 避免 LLM 幻觉)
                        self.elites[i] = func
                        self.elites.sort(key=lambda x: x.score, reverse=True)
                    return  # 同代码已处理完毕，直接退出

            # 1.2 检查是否已在 Failures 中
            for i, f in enumerate(self.failures):
                if f.body == target_code:
                    if func.score > f.score:
                        # 更新分数值
                        self.failures[i] = func

                        # [动态提拔] 分数提高后，尝试让它重新冲击精英库
                        upgraded_func = self.failures.pop(i)
                        self._try_add_to_elites(upgraded_func)
                    return  # 同代码已处理完毕，直接退出

            # === 2. 全新代码，执行瀑布流冲刺 ===
            self._try_add_to_elites(func)

    def _try_add_to_elites(self, func: Function):
        """尝试加入精英库，进不去或被挤出的，扔给次优库"""
        if len(self.elites) < self.max_elites:
            self.elites.append(func)
            self.elites.sort(key=lambda x: x.score, reverse=True)
        else:
            if func.score > self.elites[-1].score:
                # 实力足够，挤掉当前最弱的精英
                self.elites.append(func)
                self.elites.sort(key=lambda x: x.score, reverse=True)
                dropped_elite = self.elites.pop()

                # [瀑布流] 被挤掉的前精英，降级去次优库继续发光发热
                self._try_add_to_failures(dropped_elite)
            else:
                # 连最弱的精英都打不过，直接去次优库报到
                self._try_add_to_failures(func)

    def _try_add_to_failures(self, func: Function):
        """尝试加入次优库"""
        if len(self.failures) < self.max_failures:
            self.failures.append(func)
            self.failures.sort(key=lambda x: x.score, reverse=True)
        else:
            if func.score > self.failures[-1].score:
                # 挤掉次优库里最差的 (彻底淘汰)
                self.failures.append(func)
                self.failures.sort(key=lambda x: x.score, reverse=True)
                self.failures.pop()

    def sample_for_summary(self, k_success: int = 2, k_failure: int = 2) -> Tuple[List[Function], List[Function]]:
        """为 SE 算子提供高对比度的样本对"""
        with self._lock:
            # Elites: 优先给头部最强解
            k_s = min(len(self.elites), k_success)
            sampled_elites = self.elites[:k_s]

            # Failures: 随机采样，提供多样化的高质量错题本
            k_f = min(len(self.failures), k_failure)
            sampled_failures = random.sample(self.failures, k_f) if k_f > 0 else []

            return sampled_elites, sampled_failures

    def debug_status(self):
        """打印档案库状态"""
        with self._lock:
            print("\n=== External Archive Status ===")
            print(f"Elites ({len(self.elites)}/{self.max_elites}): " +
                  ", ".join([f"{f.score:.4f}" for f in self.elites]))
            print(f"Failures ({len(self.failures)}/{self.max_failures}): " +
                  ", ".join([f"{f.score:.4f}" for f in self.failures]))