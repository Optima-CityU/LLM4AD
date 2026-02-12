from llm4ad.base.code import Function
from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class Evoind:
    """
    给带算法分类的框架使用的
    进化个体，封装了 Function 对象并存储其在特定实例簇中的元数据。

    一个 Evoind 是算法池单元中进行选择、淘汰等操作的基本单位。

    使用的时候:
    I1 = Evoind(function=Function, cluster_id=)
    """
    function: Function
    cluster_id: Any

    # --- 演化元数据 (选填，带默认值) ---
    reflection: str = ""
    feature: List[Any] = field(default_factory=list)  # dataclass 中 list 需这样定义

    def __str__(self) -> str:
        return str(self.function)

    def __hash__(self):
        """
        基于底层 Function 对象实现哈希，以便在种群中进行去重。
        """
        return hash(self.function)

    def __eq__(self, other):
        """
        比较两个 Individual 实例的底层算法是否相同。
        """
        if not isinstance(other, Evoind):
            return NotImplemented
        return self.function == other.function

    def set_feature(self, feature_given):
        self.feature = feature_given

    def set_reflection(self, reflection_given):
        self.reflection = reflection_given