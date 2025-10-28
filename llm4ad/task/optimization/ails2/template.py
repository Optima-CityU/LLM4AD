
template_program = '''
import math
import numpy as np

def priority(el: tuple[int, ...], n: int = 15, w: int = 10) -> float:
    """Returns the priority with which we want to add `el` to the set.
    Args:
        el: the unique vector has the same number w of non-zero elements.
        n : length of the vector.
        w : number of non-zero elements.
    """
    return 0.
'''

task_description = """\
Help me design a novel algorithm to evaluate vectors for potential inclusion in a set. 
This involves iteratively scoring the priority of adding a vector 'el' to the set based on analysis (like bitwise), 
with the objective of maximizing the set's size.
"""

aim_java_relative_path = r"Method/src/Perturbation/Perturbation.java"

java_dir = "CVRPLIB-2025-AILSII"

# java_dir = CVRPLIB-2025-AILSII 是为了多进程并行被复制的源目录。在项目执行前该目录会被复制”进程数量“份。
 # aim_java_relative_path 是被修改的java文件相对于java_dir的相对路径 比如"./././xxx.java"
