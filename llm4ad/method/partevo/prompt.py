from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class PartEvoPrompt:
    """
    Template factory for Multimodal LLM prompts.
    Converts evolutionary search operations (Initialization, Crossover, Mutation, Modification)
    into structured instructions for the LLM.
    """

    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function):
        """
        [INIT] Zero-shot Initialization Prompt.
        Requests the LLM to design a baseline algorithm from scratch based
        solely on the task description.
        """
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content
        prompt_content = f'''You are assigned as an expert to solve the following task: {task_prompt} \n
Please design a novel algorithm to address this task by following the steps below:
1. First, describe your algorithm and its main steps in a single sentence. The description must be enclosed within double curly braces {{}}.
2. Then, implement the following Python function:
```python
{str(temp_func)}
```
'''
        return prompt_content

    @classmethod
    def get_prompt_batch_init(cls, task_prompt: str, template_function: 'Function',
                              current_population: List['Function'],
                              branch_novelty=30):
        """Generate a structured prompt to initialize a batch of novel algorithms."""
        for indi in current_population:
            assert hasattr(indi, 'algorithm'), "Each individual must have an 'algorithm' attribute."

        # 1. 安全处理模板
        temp_func = copy.deepcopy(template_function)
        temp_func.body = '    # TODO: Implement your algorithm logic here\n    pass'

        messages = []

        # 2. 独立 System Prompt
        system_prompt = (
            "You are an elite algorithm design expert. Your task is to design a novel and efficient algorithm "
            "to solve a given problem, ensuring it is highly distinct from any previously proposed solutions.\n"
            "You must strictly follow formatting instructions and output valid Python code."
        )
        messages.append({"role": "system", "content": system_prompt})

        # 3. 构建 User Content
        content = []

        # 任务背景
        content.append({
            "type": "text",
            "text": f"### Task Assignment\n{task_prompt}\n"
        })

        # 现有算法与目标 (动态生成)
        current_size = len(current_population)
        if current_size > 0:
            init_cue = f"### Existing Algorithms\nSo far, experts have proposed {current_size} algorithms. Their high-level concepts are summarized below:\n"
            content.append({"type": "text", "text": init_cue})

            for i, indi in enumerate(current_population):
                content.append({
                    "type": "text",
                    "text": f"- **Algorithm #{i + 1} Concept**: {indi.algorithm}\n"
                })

            init_task = f"\n### Goal\nYour goal is to design a new algorithm whose core concept differs from *all existing ones* by at least {branch_novelty}%."
        else:
            init_task = f"### Goal\nBased on your expertise, please create a novel and efficient algorithm to solve this problem."

        content.append({"type": "text", "text": init_task})

        # 4. 操作指令 (统一使用 <concept> 标签与严格的代码输出约束)
        operator_prompt = f"""
### Instructions
Please design your new algorithm by following these exact steps:

1. **Propose a Novel Concept**
   Describe your concept for the new algorithm and its main steps in as few words as possible while ensuring clarity.
   Wrap your core conceptual description EXACTLY inside `<concept>` and `</concept>` tags.

2. **Implement the New Algorithm**
   Implement your proposed algorithm using the exact Python function template provided below:
```python
{str(temp_func)}
```

   **STRICT RULES FOR CODE:**
   - Do NOT change the function signature (name, arguments, type hints).
   - Include all necessary `import` statements at the beginning of the function body.
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages.append({
            'role': 'user',
            'content': content
        })

        return messages

#     @classmethod
#     def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function):
#         """
#         [CROSSOVER] Diversity-driven Crossover.
#         Provides multiple parent algorithms and explicitly asks the LLM
#         to create something 'totally different' to avoid local optima.
#         """
#         for indi in indivs:
#             assert hasattr(indi, 'algorithm')
#         # template
#         temp_func = copy.deepcopy(template_function)
#         temp_func.body = ''
#         # create prompt content for all individuals
#         indivs_prompt = ''
#         for i, indi in enumerate(indivs):
#             indi.docstring = ''
#             indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{indi.to_code_without_docstring()}'
#         # create prmpt content
#         prompt_content = f'''{task_prompt}
# I have {len(indivs)} existing algorithms with their codes as follows:
# {indivs_prompt}
# Please help me create a new algorithm that has a totally different form from the given ones.
# 1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
# 2. Next, implement the following Python function:
# {str(temp_func)}
# '''
#         return prompt_content
#
#     @classmethod
#     def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function):
#         for indi in indivs:
#             assert hasattr(indi, 'algorithm')
#
#         # template
#         temp_func = copy.deepcopy(template_function)
#         temp_func.body = ''
#         # create prompt content for all individuals
#         indivs_prompt = ''
#         for i, indi in enumerate(indivs):
#             indi.docstring = ''
#             indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{indi.to_code_without_docstring()}'
#         # create prmpt content
#         prompt_content = f'''{task_prompt}
# I have {len(indivs)} existing algorithms with their codes as follows:
# {indivs_prompt}
# Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
# 1. Firstly, identify the common backbone idea in the provided algorithms.
# 2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
# 3. Thirdly, implement the following Python function:
# {str(temp_func)}'''
#         return prompt_content

    @classmethod
    def get_prompt_reflection(cls, task_prompt: str, func: 'Function', template_function: 'Function'):
        """Generate a structured prompt to ask the LLM for targeted suggestions (reflection) on a given algorithm."""
        assert hasattr(func, 'algorithm'), "func must have an 'algorithm' attribute."

        messages = []

        # 1. System Prompt
        system_prompt = (
            "You are an elite algorithm design expert. Your task is to critically analyze "
            "an algorithmic attempt and provide targeted, highly actionable suggestions for improving its task performance. "
            "You must strictly follow formatting instructions."
        )
        messages.append({"role": "system", "content": system_prompt})

        # 2. 构建 User Content
        content = []

        # 任务背景
        content.append({
            "type": "text",
            "text": f"### Task Assignment\nAn intelligent agent is currently executing the following design task:\n{task_prompt}\n"
        })

        # 当前进度与历史代码
        content.append({
            "type": "text",
            "text": f"### Current Attempt\nThe agent has designed an algorithm with the following concept and code:\n"
                    f"- **Concept**: {func.algorithm}\n"
                    f"- **Implementation**:\n```python\n{str(func)}\n```\n"
        })

        temp_func = copy.deepcopy(template_function)
        temp_func.body = '    # Your algorithm logic must be implemented here\n    pass'

        # 3. 操作指令与格式约束 (使用 XML 标签并提供示例)
        instruction_prompt = f"""### Instructions
Based on your understanding and knowledge of this design task, please provide targeted suggestions for the current algorithm to guide the agent in improving it. Identify any mathematical weaknesses, logical flaws, or opportunities for algorithmic optimization.

**STRICT RULES FOR OUTPUT:**
- **Limit your response:** Provide a maximum of the 3 MOST CRITICAL suggestions. Do not list exhaustive minor details; focus only on what will yield the highest performance gain.
- **Interface Constraint:** All your suggestions MUST be implementable within the exact Python function template provided below:
```python
{str(temp_func)}
```
- Enclose your suggestions EXACTLY within `<reflection>` and `</reflection>` tags.

Example format:
<reflection>
1. Your point 1.
2. Your point 2.
...
</reflection>
"""
        content.append({
            "type": "text",
            "text": instruction_prompt
        })

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    @classmethod
    def get_prompt_summary(cls, task_prompt: str, template_function: 'Function',
                           summary_context_samples: Dict[str, List['Function']],
                           current_summary: str = "") -> List[Dict]:
        messages = []

        # 1. System Prompt (设定为客观的首席研究员视角)
        system_prompt = (
            "You are an elite Lead Algorithm Researcher. Your task is to oversee the progress of multiple intelligent agents "
            "solving an algorithm design problem. You must critically analyze the concepts and corresponding performance scores "
            "of the algorithms they have designed, and synthesize an objective summary of what algorithmic techniques "
            "are effective and which are not, to guide the agents' future design directions. You must strictly follow formatting instructions."
        )
        messages.append({"role": "system", "content": system_prompt})

        # 2. 构建 User Content
        content = []

        # 任务背景
        content.append({
            "type": "text",
            "text": f"### Task Assignment\nThe intelligent agents are currently executing the following algorithm design problem:\n{task_prompt}\n"
        })

        # 历史总结 (增量更新机制)
        if current_summary:
            content.append({
                "type": "text",
                "text": f"### Previous Summary\nBased on earlier iterations, we have the following established summary for your reference:\n"
                        f"<previous_summary>\n{current_summary}\n</previous_summary>\n"
            })

        # 核心：客观呈现所有算法与分数 (摒弃精英/失败的主观标签)
        context_text = "### Explored Algorithms & Performance\n"
        context_text += "Below is a sample of algorithms explored by the agents. You can analyze them based on their scores (a higher score indicates better performance).\n\n"

        # 合并列表，让 LLM 自己根据 Score 去判断优劣
        elites = summary_context_samples.get('elites', [])
        hard_negatives = summary_context_samples.get('hard_negatives', [])
        all_samples = elites + hard_negatives

        # 尝试按照分数从高到低排序，方便 LLM 观察趋势
        try:
            all_samples.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        except Exception:
            pass

        for i, func in enumerate(all_samples):
            concept = getattr(func, 'algorithm', 'No concept description provided.')
            score = getattr(func, 'score', 'N/A')
            context_text += f"- **Algorithm #{i + 1}** (Score: {score}):\n  Concept: {concept}\n\n"

        content.append({
            "type": "text",
            "text": context_text
        })

        # 3. 操作指令与格式约束 (强调客观推断与未来方向)
        instruction_prompt = """### Instructions
Please review these algorithms, their performance scores, and the previous summary (if provided) to summarize the findings and deduce how to better solve this algorithm design task.

**STRICT RULES FOR OUTPUT:**
- If a previous summary was provided, update and evolve it with these new insights rather than just repeating it.
- Enclose your ENTIRE summary EXACTLY within `<summary>` and `</summary>` tags.

Example format:
<summary>
[A concise yet clear summary.]
</summary>
"""
        content.append({
            "type": "text",
            "text": instruction_prompt
        })

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    @classmethod
    def get_prompt_re(cls, task_prompt: str, parent_func: Function, template_function: Function, reflection: str):
        assert hasattr(parent_func, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = '    # TODO: Implement your algorithm logic here\n    pass'

        # Construct prompt content
        messages = []

        system_prompt = (
            "You are an elite algorithm design expert. Your task is to analyze previous algorithmic attempts, "
            "incorporate expert feedback if provided, and design a novel, superior algorithm.\n"
            "You must strictly follow formatting instructions and output valid Python code."
        )

        messages.append({"role": "system", "content": system_prompt})

        content = []

        content.append({
            "type": "text",
            "text": f"### Task Assignment\nAn intelligent agent is executing the following design task:\n{task_prompt}\n"
        })

        content.append({
            "type": "text",
            "text": f"### Previous Attempt\nThe agent designed the following algorithm:\n"
                    f"- **Concept**: {parent_func.algorithm}\n"
                    f"- **Implementation**:\n```python\n{str(parent_func)}\n```\n"
        })

        if reflection:
            content.append({
                "type": "text",
                "text": f"### Expert Reflection\nAn expert reviewed the previous attempt and provided the following feedback:\n"
                        f"<feedback>{reflection}</feedback>\n"
                        f"Please consider this feedback to create a new, improved algorithm."
            })
        else:
            content.append({
                "type": "text",
                "text": "### Expert Reflection\nNo specific expert feedback was provided. Please independently identify weaknesses in the previous attempt and create a new, improved algorithm."
            })

        operator_prompt = f"""### Instructions
Please design your new algorithm by following these exact steps:

1. **Propose a New Concept**
   First, describe your concept for the new algorithm and its main steps in as few words as possible while ensuring clarity.First, describe your concept for the new algorithm and its main steps in as few words as possible while ensuring clarity.
   Wrap your core conceptual description inside `<concept>` and `</concept>` tags.

2. **Implement the New Algorithm**
   Next, implement your concept using the exact Python function template provided below:
```python
{str(temp_func)}
```

   **STRICT RULES FOR CODE:**
   - Do NOT change the function signature (name, arguments, type hints).
   - Include all necessary `import` statements at the beginning of the function body.
"""
        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    @classmethod
    def get_prompt_se(cls, task_prompt: str, parent_func: 'Function', template_function: 'Function', global_summary: str) -> List[Dict]:
        """
                Generate a structured prompt for the Semantic Exploration (SE) operator.
                Uses a 'God View' global summary to guide the modification of the current algorithm.
                """
        assert hasattr(parent_func, 'algorithm'), "parent_func must have an 'algorithm' attribute."

        # Prepare the function template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = '    # TODO: Implement your algorithm logic here\n    pass'

        messages = []

        system_prompt = (
            "You are a Senior Algorithm Research Scientist specializing in "
            "iterative optimization and automated algorithm design.\n\n"
            "Your objective is to evolve a specific algorithm instance by "
            "considering:\n"
            "1) The current local implementation\n"
            "2) High-level global insights extracted from prior experiments\n\n"
            "You may reason and organize your thoughts freely.\n"
            "However, your FINAL OUTPUT must strictly contain:\n"
            "1) A new algorithm concept wrapped inside <concept> and </concept>\n"
            "2) A valid Python implementation of the upgraded algorithm\n"
        )
        messages.append({"role": "system", "content": system_prompt})

        content = []

        # 2. 任务背景
        content.append({
            "type": "text",
            "text": f"### Task Assignment\nThe algorithm design task is:\n{task_prompt}\n"
        })

        # 3. 当前待进化的个体 (局部视角)
        content.append({
            "type": "text",
            "text": f"### Current Algorithm\n"
                    f"- **Concept**: {parent_func.algorithm}\n"
                    f"- **Implementation**:\n```python\n{str(parent_func)}\n```\n"
        })

        # 4. 引入全局经验 (历史尝试的总结)
        if global_summary:
            content.append({
                "type": "text",
                "text": f"### Global Insights from Past Attempts\n"
                        f"Based on the validation of various algorithms across multiple expert clusters, the following insights have been summarized:\n"
                        f"<global_summary>\n{global_summary}\n</global_summary>\n\n"
                        f"Please analyze these insights carefully and apply them to modify and improve the current algorithm to create a more promising solution."
            })
        else:
            # Fallback 容错：如果系统早期还没有收集到足够的样本生成总结
            content.append({
                "type": "text",
                "text": "### Global Insights from Past Attempts\n"
                        "No global summary is currently available. Please independently analyze the current algorithm, identify its potential weaknesses, and modify it to create a superior solution."
            })

        # 5. 操作指令与格式约束
        # 注意：这里修正了 re 原有代码里 "First, describe your concept..." 句子重复两遍的小笔误
        operator_prompt = f"""### Instructions
Design an improved algorithm by strictly following the steps below:

1. **Propose an Upgraded Concept**
   First, describe your concept for the new algorithm and its main steps as succinctly as possible while ensuring clarity.
   Wrap your core conceptual description EXACTLY inside `<concept>` and `</concept>` tags.

2. **Implement the New Algorithm**
   Next, implement your upgraded concept using the exact Python function template provided below:
```python
{str(temp_func)}
```

**STRICT RULES FOR CODE:**
- Do NOT change the function signature (name, arguments, type hints).
- Include all necessary `import` statements at the beginning of the function body.
"""
        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    @classmethod
    def get_prompt_cn(cls, task_prompt: str, parents: List[Function], template_function: Function):
        """Use Figures to instruct the design progress"""
        for indi in parents:
            assert hasattr(indi, 'algorithm')

        messages = []

        # 1. System Prompt (设定为精通算法融合的高级开发者)
        system_prompt = (
            "You are an elite Algorithm Developer specializing in algorithmic synthesis and hybridization. "
            "Your task is to review multiple existing algorithms, use one as the primary framework, "
            "and intelligently incorporate the advantageous characteristics of the others to synthesize a superior algorithm."
        )
        messages.append({"role": "system", "content": system_prompt})

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''

        # Construct prompt content
        content = []

        # 任务背景
        content.append({
            "type": "text",
            "text": f"### Task Assignment\nThe core problem you need to solve is:\n{task_prompt}\n"
        })

        # 核心：展示父代个体 (区分主干和辅助)
        context_text = "### Parent Algorithms for Crossover\n"

        # 提取主干算法 (Main Framework)
        main_parent = parents[0]
        main_concept = getattr(main_parent, 'algorithm', 'No concept description provided.')
        main_score = getattr(main_parent, 'score', 'N/A')

        context_text += f"#### Main Framework (Algorithm #1)\n"
        context_text += f"**Score:** {main_score}\n"
        context_text += f"**Concept:** {main_concept}\n"
        context_text += f"**Code:**\n```python\n{main_parent.to_code_without_docstring()}\n```\n\n"

        # 提取辅助算法 (Auxiliary Algorithms)
        if len(parents) > 1:
            context_text += "#### Auxiliary Algorithms\n"
            for i, aux_parent in enumerate(parents[1:]):
                aux_concept = getattr(aux_parent, 'algorithm', 'No concept description provided.')
                aux_score = getattr(aux_parent, 'score', 'N/A')

                context_text += f"--- Auxiliary Algorithm #{i + 2} ---\n"
                context_text += f"**Score:** {aux_score}\n"
                context_text += f"**Concept:** {aux_concept}\n"
                context_text += f"**Code:**\n```python\n{aux_parent.to_code_without_docstring()}\n```\n\n"

        content.append({
            "type": "text",
            "text": context_text
        })

        # Expert instructions
        operator_prompt = f"""### Instructions
Please take **Algorithm #1 as the main framework** and critically analyze the concepts and code of the Auxiliary Algorithm(s). 
Identify their strengths and creatively incorporate those advantageous characteristics into the main framework to design a new, better algorithm.

Please follow these steps to output your newly designed algorithm:

1. **Propose an Upgraded Concept**
   First, describe your concept for the new algorithm and its main steps as succinctly as possible while ensuring clarity.
   Wrap your core conceptual description EXACTLY inside `<concept>` and `</concept>` tags.

2. **Implement the New Algorithm**
   Next, implement your upgraded concept using the exact Python function template provided below:
```python
{str(temp_func)}
```

**STRICT RULES FOR CODE:**
- Do NOT change the function signature (name, arguments, type hints).
- Include all necessary `import` statements at the beginning of the function body.
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages.append({
            "role": "user",
            "content": content
        })

        return messages

    @classmethod
    def get_prompt_e2_advanced(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        """Return Messages"""
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})
        description_situation = (
            f"There are {len(indivs)} existing algorithms that are functional, but we aim to develop more effective ones (A higher score indicates better performance). "
            f"Each algorithm's concept and implementation is provided below:")
        content.append({"type": "text", "text": description_situation})
        for i, indi in enumerate(indivs):
            content.extend([
                {
                    "type": "text",
                    "text": f'Algorithm #{i + 1} (Score: {indi.score:.3f}):\n'
                            f'Concept: {indi.algorithm}\n'
                            f'Implementation:\n{indi.to_code_without_docstring()}\n'
                }
            ])

        # Expert instructions
        operator_prompt = f"""
Please analyze the above algorithms and design a new one that differs structurally but draws inspiration from them. Follow these steps:

1. Identify the strengths and weaknesses of each algorithm. Write your analysis inside square brackets: [your analysis].
2. Design a novel algorithm that integrates the strengths while addressing the limitations of the current ones. Clearly explain the core idea and enclose it in curly braces: {{your core idea}}.
3. Implement your proposed algorithm using the following Python function template:
```python
{str(temp_func)}
```
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_e2_M(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        """Use Figures to instruct the design progress"""
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
            assert hasattr(indi, 'image64')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})
        description_situation = (
            f"There are {len(indivs)} existing algorithms that are functional, but we aim to develop more effective ones (A higher score indicates better performance). "
            f"Each algorithm's concept, implementation, and execution results are provided below:")
        content.append({"type": "text", "text": description_situation})
        for i, indi in enumerate(indivs):
            content.extend([
                {
                    "type": "text",
                    "text": f'Algorithm #{i + 1} (Score: {indi.score:.3f}):\n'
                            f'Concept: {indi.algorithm}\n'
                            f'Implementation:\n{indi.to_code_without_docstring()}\n'
                },
                {
                    "type": "text",
                    "text": f"Execution results visualization for Algorithm #{i + 1}:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{indi.image64}",
                    }
                }
            ])

        # Expert instructions
        operator_prompt = f"""
Please begin by carefully analyzing the execution results shown in the images. Provide your interpretation enclosed in single quotes: 'your description here'.
Then, following your analysis, improve upon the algorithms using the steps below:

1. Analyze the strengths and weaknesses of each algorithm. Write your analysis inside square brackets: [your analysis].
2. Propose a new algorithm concept that integrates the strengths while addressing the weaknesses. Enclose your core idea in curly braces: {{your core idea}}.
3. Implement the improved algorithm using the following Python function template:
```python
{str(temp_func)}
```
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept and implementation:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{indi.to_code_without_docstring()}\n'
            }
        ])

        operator_prompt = f"""Please assist in designing an improved version of the algorithm. Your task involves the following steps:
1. **Analyze** the current algorithm to identify its potential weaknesses and areas for improvement. Enclose your analysis in square brackets [ ].
2. **Propose** an enhanced algorithm. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.
3. **Implement** the enhanced algorithm using the following Python function template:
```python
{str(temp_func)}
```\n"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept and implementation:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{indi.to_code_without_docstring()}\n'
            }
        ])

        operator_prompt = f"""Please help optimize the algorithm by modifying its parameter settings. Follow these steps:
1. Parameter Analysis
   - Identify all key parameters and their functions.  
   - Determine which parameters are candidates for modification to improve performance.  
   - Explain why these specific changes would help.
   - All content related to Parameter Analysis must be enclosed within brackets {[]}.

2. Create a new algorithm that has a different parameter settings of the algorithm provided. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

3. Implement the improved version using this Python function template:
```python
{str(temp_func)}
```
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m1_M(cls, task_prompt: str, indi: Function, template_function: Function):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'image64'), "Individual must have 'image64' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept, implementation, and execution results:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{indi.to_code_without_docstring()}\n'
            },
            {
                "type": "text",
                "text": f"Execution results visualization for the algorithm:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{indi.image64}",
                }
            }
        ])

        operator_prompt = f"""Please start by providing a detailed description and analysis of the execution result, enclosed within single quotes (' '). 
Next, based on your analysis, optimize the algorithm by following these steps:

1. Analyze why the results were produced in relation to the algorithm. Identify its weaknesses and areas for improvement, and enclose your analysis within square brackets [ ].

2. Propose an enhanced algorithm. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

3. Implement the enhanced algorithm using the following Python function template:
```python
{str(temp_func)}
```
\n"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m1_M_only_image(cls, task_prompt: str, indi: Function, template_function: Function):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'image64'), "Individual must have 'image64' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept, implementation, and execution results:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{indi.to_code_without_docstring()}\n'
            },
            {
                "type": "text",
                "text": f"Execution results visualization for the algorithm:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{indi.image64}",
                }
            }
        ])

        operator_prompt = f"""Please optimize this algorithm by following these steps:

1. Analyze why the results were produced in relation to the algorithm. Identify its weaknesses and areas for improvement, and enclose your analysis within square brackets [ ].

2. Propose an enhanced algorithm. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

3. Implement the enhanced algorithm using the following Python function template:
```python
{str(temp_func)}
```
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m1_M_text_info(cls, task_prompt: str, indi: Function, template_function: Function,
                                  information_explanation):
        """Use text information [no-image] to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'observation'), "Individual must have 'observation' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept, implementation, and a execution result:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{indi.to_code_without_docstring()}\n'
            },
            {
                "type": "text",
                "text": f"Information contained in the execution result: {information_explanation}\n"
            },
            {
                "type": "text",
                "text": f'The specific execution result is as follows: {indi.observation}\n'
            }
        ])

        operator_prompt = f"""Please start by providing a detailed description and analysis of the execution result, enclosed within single quotes (' '). 
Next, based on your analysis, optimize the algorithm by following these steps:

1. Analyze why the results were produced in relation to the algorithm. Identify its weaknesses and areas for improvement, and enclose your analysis within square brackets [ ].

2. Propose an enhanced algorithm. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

3. Implement the enhanced algorithm using the following Python function template:
```python
{str(temp_func)}
```
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_image_description(cls, task_prompt: str, indi: Function, template_function: Function):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'image64'), "Individual must have 'image64' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept, implementation, and execution results:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{str(indi)}\n'
            },
            {
                "type": "text",
                "text": f"Execution results visualization for the algorithm:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{indi.image64}",
                }
            }
        ])

        operator_prompt = (
            "Please provide a detailed description and analysis of the execution results. "
            "You may analyze why the results appear as they do, in relation to the algorithm's structure and behavior. "
            "Identify any limitations, inefficiencies, or opportunities for improvement. "
            "Feel free to extend beyond these suggestions. "
            "Make sure your response is clear, logically organized, and enclosed within double curly braces {{ }}."
        )

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m1_M_image_description(cls, task_prompt: str, indi: Function, template_function: Function,
                                          description: str):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'image64'), "Individual must have 'image64' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept, implementation, and an expert evaluation of its execution results:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{indi.to_code_without_docstring()}\n'
            },
            {
                "type": "text",
                "text": f"Expert Evaluation of Execution Results:\n{description}\n"
            },
        ])

        operator_prompt = f"""Based on the algorithm and the expert evaluation, please optimize the algorithm by following these steps:

        1. Analyze the evaluation results in conjunction with the given algorithm to identify the root causes of any issues. Clearly specify the algorithm’s weaknesses and potential areas for improvement. Present your analysis within square brackets [ ].

        2. Propose an enhanced algorithm. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

        3. Implement the enhanced algorithm using the following Python function template:
        ```python
        {str(temp_func)}
        ```
        \n"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m2_M(cls, task_prompt: str, indi: Function, template_function: Function):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'image64'), "Individual must have 'image64' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept, implementation, and execution results:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{indi.to_code_without_docstring()}\n'
            },
            {
                "type": "text",
                "text": f"Execution results visualization for the algorithm:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{indi.image64}",
                }
            }
        ])

        operator_prompt = f"""Please start by providing a detailed description and analysis of the execution result, enclosed within single quotes (' '). 
Next, based on your analysis, optimize the algorithm by following these steps:

1. Parameter Analysis:
   - Identify all key parameters and their functions.
   - Determine which parameters should be modified to improve results.
   - Explain why these specific changes would help.
   - All content related to Parameter Analysis must be enclosed within brackets {[]}.

2. Create a new algorithm that has a different parameter settings of the algorithm provided. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

3. Implement the improved version using this Python function template:
```python
{str(temp_func)}
```
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m2_M_image_description(cls, task_prompt: str, indi: Function, template_function: Function,
                                          description: str):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'image64'), "Individual must have 'image64' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its concept, implementation, and an expert evaluation of its execution results:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Concept: {indi.algorithm}\n'
                        f'Implementation:\n{indi.to_code_without_docstring()}\n'
            },
            {
                "type": "text",
                "text": f"Expert Evaluation of Execution Results:\n{description}\n"
            },
        ])

        operator_prompt = f"""Based on the algorithm and the expert evaluation, please optimize the algorithm by following these steps:
1. Parameter Analysis:
    - Identify all key parameters and their functions.
    - Determine which parameters should be modified to improve results.
    - Explain why these specific changes would help.
    - All content related to Parameter Analysis must be enclosed within brackets {[ ]}.

2. Create a new algorithm that has a different parameter settings of the algorithm provided. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

3. Implement the improved version using this Python function template:
```python
{str(temp_func)}
```
"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_e1_nothought(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f"No. {i + 1} algorithm\'s code is:{indi.to_code_without_docstring()}"
        # create prmpt content
        prompt_content = f'''{task_prompt}
    I have {len(indivs)} existing algorithms with their codes as follows:
    {indivs_prompt}
    Please help me create a new algorithm that has a totally different form from the given ones. 
    1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
    2. Next, implement the following Python function:
    {str(temp_func)}
    '''
        return prompt_content

    @classmethod
    def get_prompt_e2_nothought(cls, task_prompt: str, indivs: List[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f"No. {i + 1} algorithm\'s code is:{indi.to_code_without_docstring()}"
        # create prmpt content
        prompt_content = f'''{task_prompt}
    I have {len(indivs)} existing algorithms with their codes as follows:
    {indivs_prompt}
    Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
    1. Firstly, identify the common backbone idea in the provided algorithms. 
    2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside within boxed {{}}.
    3. Thirdly, implement the following Python function:
    {str(temp_func)}'''
        return prompt_content

    @classmethod
    def get_prompt_m1_M_nothought(cls, task_prompt: str, indi: Function, template_function: Function):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'image64'), "Individual must have 'image64' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its implementation and execution results:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Implementation:\n{indi.to_code_without_docstring()}\n'
            },
            {
                "type": "text",
                "text": f"Execution results visualization for the algorithm:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{indi.image64}",
                }
            }
        ])

        operator_prompt = f"""Please start by providing a detailed description and analysis of the execution result, enclosed within single quotes (' '). 
    Next, based on your analysis, optimize the algorithm by following these steps:

    1. Analyze why the results were produced in relation to the algorithm. Identify its weaknesses and areas for improvement, and enclose your analysis within square brackets [ ].

    2. Propose an enhanced algorithm. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

    3. Implement the enhanced algorithm using the following Python function template:
    ```python
    {str(temp_func)}
    ```
    \n"""

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages

    @classmethod
    def get_prompt_m2_M_nothought(cls, task_prompt: str, indi: Function, template_function: Function):
        """Use Figures to instruct the design progress"""
        assert hasattr(indi, 'algorithm'), "Individual must have 'algorithm' attribute"
        assert hasattr(indi, 'image64'), "Individual must have 'image64' attribute"

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # Construct prompt content
        content = []
        # Task assignment
        assigment_prompt = f"You are assigned as an expert to participate in the following task:\n{task_prompt}\n"
        content.append({"type": "text", "text": assigment_prompt})

        description_situation = ("We have a working algorithm that needs optimization."
                                 "Below are its implementation and execution results:")
        content.append({"type": "text", "text": description_situation})

        content.extend([
            {
                "type": "text",
                "text": f'Implementation:\n{indi.to_code_without_docstring()}\n'
            },
            {
                "type": "text",
                "text": f"Execution results visualization for the algorithm:"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{indi.image64}",
                }
            }
        ])

        operator_prompt = f"""Please start by providing a detailed description and analysis of the execution result, enclosed within single quotes (' '). 
    Next, based on your analysis, optimize the algorithm by following these steps:

    1. Parameter Analysis:
       - Identify all key parameters and their functions.
       - Determine which parameters should be modified to improve results.
       - Explain why these specific changes would help.
       - All content related to Parameter Analysis must be enclosed within brackets {[]}.

    2. Create a new algorithm that has a different parameter settings of the algorithm provided. Use concise language to describe the core idea of your algorithm, and enclose the core idea within curly braces {{ }}.

    3. Implement the improved version using this Python function template:
    ```python
    {str(temp_func)}
    ```
    """

        content.append({
            "type": "text",
            "text": operator_prompt
        })

        messages = [{
            'role': 'user',
            'content': content
        }]

        return messages