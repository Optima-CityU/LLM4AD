# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
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

"""
This file implements 2 classes representing unities of code:

- Function, containing all the information we need about functions: name, args,
  body and optionally a return type and a docstring.

- Program, which contains a code preface (which could be imports, global
  variables and classes, ...) and a list of Functions.

- For example, a function is shown below,
which is an un-executable program because 'np' is not defined, and 'WEIGHT' is not defined.
--------------------------------------------------------------------------------------------
def func(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = b + WEIGHT
    return a + b
--------------------------------------------------------------------------------------------

- A program is an executable object as shown below.
--------------------------------------------------------------------------------------------
import numpy as np
WEIGHT = 10

def func(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    b = b + WEIGHT
    return a + b
--------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from typing import Any, List, Callable


@dataclasses.dataclass
class Program:
    """A parsed Python function."""

    algorithm = ''
    name: str
    args: str
    body: str

    return_type: str | None = None
    docstring: str | None = None
    score: Any | None = None
    evaluate_time: float | None = None
    sample_time: float | None = None

    def __str__(self) -> str:
        return_type = f' -> {self.return_type}' if self.return_type else ''

        function = f'def {self.name}({self.args}){return_type}:\n'
        if self.docstring:
            # self.docstring is already indented on every line except the first one.
            # Here, we assume the indentation is always four spaces.
            new_line = '\n' if self.body else ''
            function += f'    """{self.docstring}"""{new_line}'
        # self.body is already indented.
        function += self.body + '\n\n'
        return function

    def __setattr__(self, name: str, value: str) -> None:
        # Ensure there aren't leading & trailing new lines in `body`.
        if name == 'body':
            value = value.strip('\n')
        # Ensure there aren't leading & trailing quotes in `docstring``.
        if name == 'docstring' and value is not None:
            if '"""' in value:
                value = value.strip()
                value = value.replace('"""', '')
        super().__setattr__(name, value)

    def __eq__(self, other: Program):
        assert isinstance(other, Program)
        return (self.name == other.name and
                self.args == other.args and
                self.return_type == other.return_type and
                self.body == other.body)
