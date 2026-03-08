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

from llm4ad.base.code import Function
from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class Evoind:
    """
    Evolutionary Individual used within the PartEvo framework.
    Wraps a Function object and stores its metadata within a specific algorithm cluster.

    An Evoind is the basic unit for selection, elimination, and other operations
    within the algorithm pool.
    """
    function: Function
    cluster_id: Optional[int] = None

    # --- Evolutionary Metadata ---
    reflection: str = ""
    feature: List[Any] = field(default_factory=list)

    def __str__(self) -> str:
        return str(self.function)

    def __hash__(self):
        """Implement hashing based on the underlying Function object for deduplication."""
        return hash(self.function)

    def __eq__(self, other):
        """Compare if two Individuals have the same underlying Function object."""
        if not isinstance(other, Evoind):
            return NotImplemented
        return self.function == other.function

    def set_feature(self, feature_given):
        self.feature = feature_given

    def set_reflection(self, reflection_given):
        self.reflection = reflection_given