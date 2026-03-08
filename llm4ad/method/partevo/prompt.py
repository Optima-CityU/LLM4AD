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
            "You are an elite algorithm design expert and a creative computer scientist. "
            "Your task is to design a novel and executable algorithm to solve a given problem. "
            "It is crucial that your core methodology and algorithmic logic are highly distinct from any previously proposed solutions (if provided).\n\n"
            "You may reason and organize your thoughts freely, but your FINAL OUTPUT must strictly adhere to the following format:\n"
            "1) A concise summary of your new algorithm concept wrapped EXACTLY inside <concept> and </concept> tags.\n"
            "2) A valid Python implementation of the algorithm."
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

            init_task = (f"\n### Goal\nDesign a new algorithm whose core mechanism is substantially different "
                         f"from ALL existing algorithms listed above.\n"
                         f"The conceptual difference must be at least {branch_novelty}% "
                         f"in terms of underlying strategy or logic.\n"
                         f"Simple modifications, recombinations, or parameter tweaks are NOT sufficient."
                         )

        else:
            init_task = f"### Goal\nBased on your expertise, please create a novel and efficient algorithm to solve this problem."

        content.append({"type": "text", "text": init_task})

        operator_prompt = f"""
### Instructions
Please design your new algorithm by following these exact steps:

Step 1 -- Propose a Novel Algorithmic Concept
- Clearly describe the core mechanism of your new algorithm.
- Focus on the fundamental reasoning principle.
- Keep it concise but precise.
- Wrap the description EXACTLY inside:
<concept>
...
</concept>

Step 2 — Implement the Algorithm  
- Implement your proposed algorithm using the exact Python function template provided below:
```python
{str(temp_func)}
```
- Do NOT modify the function name, arguments, or return type.
- Place all necessary `import` statements inside the function body.
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

    @classmethod
    def get_prompt_reflection(cls, task_prompt: str, func: 'Function', template_function: 'Function'):
        """Generate a structured prompt to ask the LLM for targeted suggestions (reflection) on a given algorithm."""
        assert hasattr(func, 'algorithm'), "func must have an 'algorithm' attribute."

        messages = []

        # 1. System Prompt
        system_prompt = (
            "You are an elite algorithm design expert.\n"
            "Your task is to critically analyze an existing algorithmic attempt and propose "
            "actionable, high-impact reflections that may enhance its task performance.\n\n"
            "Focus on identifying fundamental reasoning flaws, structural inefficiencies, "
            "mathematical weaknesses, suboptimal search dynamics, or poorly chosen parameter strategies.\n"
            "DO NOT provide generic advice.\n"
            "You must strictly follow the required output format."
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
            "text": (
                f"### Current Attempt\n"
                f"The agent has designed an algorithm with the following concept and implementation:\n"
                f"- **Concept**: {func.algorithm}\n"
                f"- **Implementation**:\n"
                f"```python\n{str(func)}\n```\n"
            )
        })

        temp_func = copy.deepcopy(template_function)
        temp_func.body = '    # Your algorithm logic must be implemented here\n    pass'

        # 3. 操作指令与格式约束 (使用 XML 标签并提供示例)
        instruction_prompt = f"""### Instructions
Based on your expert knowledge of the design task, critically evaluate the current algorithm and provide targeted suggestions to guide the agent in improving it.

**STRICT RULES FOR OUTPUT:**
- Provide a maximum of the 3 MOST CRITICAL suggestions. Focus on what will yield the performance gain.
- All your suggestions MUST be implementable within the exact Python function template provided below:
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

        # 1. System Prompt
        system_prompt = (
            "You are an elite Lead Algorithm Researcher. Your task is to oversee the progress of intelligent agents "
            "solving an algorithm design task. You must critically analyze the concepts and corresponding performance scores "
            "of the algorithms they have designed, and synthesize an objective and comprehensive summary of what algorithmic techniques "
            "are effective and which are not for the task, to guide the agents' future design directions. You must strictly follow formatting instructions."
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
                "text": f"### Previous Summary\nBased on earlier attempts, we have the following established summary:\n"
                        f"<previous_summary>\n{current_summary}\n</previous_summary>\n"
            })

        # 核心：客观呈现所有算法与分数 (摒弃精英/失败的主观标签)
        context_text = "### Explored Algorithms & Performance\n"
        context_text += "Below is a set of algorithms explored by the agents. You can analyze them based on their scores (a higher score indicates better performance).\n\n"

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
            "You are an elite algorithm design expert. Your task is to analyze a previous algorithmic attempt "
            "and expert feedback (if provided), and then design a new, superior algorithm.\n"
            "You may reason and organize your thoughts freely, but your FINAL OUTPUT must strictly adhere to the following format:\n"
            "1) A concise summary of your new algorithm concept wrapped EXACTLY inside <concept> and </concept> tags.\n"
            "2) A valid Python implementation of the algorithm."
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
Importantly, you must modify and upgrade the previous algorithmic attempt; superficial refactoring, or simple reweighting is NOT allowed.
Please design your new algorithm by following these exact steps:

Step 1 -- Propose a New Algorithmic Concept
- Clearly describe the core mechanism of your algorithm.
- Focus on the fundamental reasoning principle.
- Keep it concise but precise.
- Wrap the description EXACTLY inside:
<concept>
...
</concept>

Step 2 — Implement the Algorithm  
- Implement your proposed algorithm using the exact Python function template provided below:
```python
{str(temp_func)}
```
- Do NOT modify the function name, arguments, or return type.
- Place all necessary `import` statements inside the function body.
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
    def get_prompt_se(cls, task_prompt: str, parent_func: 'Function', template_function: 'Function',
                      global_summary: str) -> List[Dict]:
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
            "You are a Senior Algorithm Research Scientist.\n"
            "Your task is to evolve and improve a specific algorithm by simultaneously considering:\n"
            "1) The current algorithmic attempt and its implementation.\n"
            "2) High-level global insights extracted from prior experiments.\n\n"
            "You may reason and organize your thoughts freely, but your FINAL OUTPUT must strictly adhere to the following format:\n"
            "1) A concise summary of your new algorithm concept wrapped EXACTLY inside <concept> and </concept> tags.\n"
            "2) A valid Python implementation of the algorithm."
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
                        f"Based on the validation of various algorithms across multiple experts, the following insights have been summarized:\n"
                        f"<global_summary>\n{global_summary}\n</global_summary>\n\n"
                        f"Please analyze these insights carefully and apply them to modify and improve the current algorithm to create a more promising one."
            })
        else:
            # Fallback 容错：如果系统早期还没有收集到足够的样本生成总结
            content.append({
                "type": "text",
                "text": "### Global Insights from Past Attempts\n"
                        "No global summary is currently available. Please independently analyze the current algorithm, identify its potential weaknesses, and modify it to create a superior solution."
            })

        # 5. 操作指令与格式约束
        operator_prompt = f"""### Instructions
Importantly, you must modify and upgrade the Current Algorithm; superficial refactoring, or simple reweighting is NOT allowed.
Please design your new algorithm by following these exact steps:

Step 1 -- Propose a New Algorithmic Concept
- Clearly describe the core mechanism of your algorithm.
- Focus on the fundamental reasoning principle.
- Keep it concise but precise.
- Wrap the description EXACTLY inside:
<concept>
...
</concept>

Step 2 — Implement the Algorithm  
- Implement your proposed algorithm using the exact Python function template provided below:
```python
{str(temp_func)}
```
- Do NOT modify the function name, arguments, or return type.
- Place all necessary `import` statements inside the function body.
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
        for indi in parents:
            assert hasattr(indi, 'algorithm')

        messages = []

        # 1. System Prompt
        system_prompt = (
            "You are an elite Algorithm Developer specializing in algorithmic synthesis and hybridization. "
            "Your task is to review multiple existing algorithms, use the primary one as your foundation, "
            "and intelligently graft or integrate the advantageous characteristics of the others to synthesize a new, superior algorithm."
            "You may reason and organize your thoughts freely, but your FINAL OUTPUT must strictly adhere to the following format:\n"
            "1) A concise summary of your hybridized algorithm concept wrapped EXACTLY inside <concept> and </concept> tags.\n"
            "2) A valid Python implementation of the upgraded algorithm."
        )
        messages.append({"role": "system", "content": system_prompt})

        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = '    # TODO: Implement your algorithm logic here\n    pass'

        # Construct prompt content
        content = []

        # 任务背景
        content.append({
            "type": "text",
            "text": f"### Task Assignment\nThe core problem you need to solve is:\n{task_prompt}\n"
        })

        # 核心：展示父代个体 (区分主干和辅助)
        context_text = "### Parent Algorithms\n"

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
Please take **Algorithm #1 as the primary algorithm** and critically analyze the concepts and code of the Auxiliary Algorithm(s). 
Identify their strengths and creatively incorporate those advantageous characteristics into the primary one to design a new, superior algorithm.
Importantly, you must modify and upgrade the primary algorithm; superficial refactoring, or simple reweighting is NOT allowed.

Please design your new algorithm by following these exact steps:

Step 1 -- Propose a New Algorithmic Concept
- Clearly describe the core mechanism of your algorithm.
- Focus on the fundamental reasoning principle.
- Keep it concise but precise.
- Wrap the description EXACTLY inside:
<concept>
...
</concept>

Step 2 — Implement the Algorithm  
- Implement your proposed algorithm using the exact Python function template provided below:
```python
{str(temp_func)}
```
- Do NOT modify the function name, arguments, or return type.
- Place all necessary `import` statements inside the function body.
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
    def get_prompt_lge(cls, task_prompt: str, parents: List['Function'], template_function: 'Function') -> List[Dict]:
        """
        Generates the prompt for the Local-Global Exploration (LGE) operator.
        Inspired by Particle Swarm Optimization (PSO), it evolves parents[0] (the current algorithm)
        by drawing insights from superior references (Local Best and/or Global Best) in parents[1:].
        """
        import copy

        for indi in parents:
            assert hasattr(indi, 'algorithm'), "Each parent must have an 'algorithm' description."

        messages = []

        # 1. System Prompt (设定为利用群体智能的高级开发者)
        system_prompt = (
            "You are a Senior Algorithm Research Scientist. "
            "Your task is to modify a baseline algorithm by identifying its weaknesses and drawing strategic "
            "inspiration from the best-performing algorithms discovered by the agent swarm. "
            "You must synthesize these inspirations to push the baseline algorithm past its current performance ceiling.\n\n"
            "You may reason and organize your thoughts freely, but your FINAL OUTPUT must strictly adhere to the following format:\n"
            "1) A concise summary of your upgraded algorithm concept wrapped EXACTLY inside <concept> and </concept> tags.\n"
            "2) A valid Python implementation of the algorithm."
        )
        messages.append({"role": "system", "content": system_prompt})

        # 2. Prepare the Template Function Skeleton
        temp_func = copy.deepcopy(template_function)
        temp_func.body = '    # TODO: Implement your algorithm logic here\n    pass'

        # 3. Construct User Content
        content = []

        # 任务背景
        content.append({
            "type": "text",
            "text": f"### Task Assignment\nThe algorithm design task you need to solve is:\n{task_prompt}\n"
        })

        # 核心：展示当前算法和优秀的参考系
        context_text = "Evolutionary Context\n"
        context_text += "You are currently focusing on the following baseline algorithm. Your goal is to evolve and improve it.\n\n"

        # 提取当前基础算法 (Current Evolving Individual)
        current_parent = parents[0]
        curr_concept = getattr(current_parent, 'algorithm', 'No concept description provided.')
        curr_score = getattr(current_parent, 'score', 'N/A')

        context_text += f"#### Current Algorithm Baseline\n"
        context_text += f"**Score:** {curr_score}\n"
        context_text += f"**Concept:** {curr_concept}\n"
        context_text += f"**Code:**\n```python\n{current_parent.to_code_without_docstring()}\n```\n\n"

        # 提取全局/局部最佳作为参考 (Global/Local Optima References)
        if len(parents) > 1:
            context_text += "#### Superior References\n"
            context_text += "Other agents in the swarm have discovered the following better-performing algorithms. "
            context_text += "Use them as gravitational references to guide the evolution of your Current Algorithm.\n\n"

            for i, ref_parent in enumerate(parents[1:]):
                ref_concept = getattr(ref_parent, 'algorithm', 'No concept description provided.')
                ref_score = getattr(ref_parent, 'score', 'N/A')

                context_text += f"--- Reference Optimum #{i + 1} ---\n"
                context_text += f"**Score:** {ref_score}\n"
                context_text += f"**Concept:** {ref_concept}\n"
                context_text += f"**Code:**\n```python\n{ref_parent.to_code_without_docstring()}\n```\n\n"
        else:
            context_text += "#### Note\n"
            context_text += "The current algorithm is already the global best known in the swarm. Please perform a deep self-reflection to mutate and push its analytical boundaries further.\n\n"

        content.append({
            "type": "text",
            "text": context_text
        })

        operator_prompt = f"""### Instructions
Carefully analyze the Superior References to understand why they achieved higher scores. 
Extract the key principles behind their success and integrate those insights to improve the Current Algorithm. 
Importantly, you must modify and upgrade the current algorithm by incorporating insights;  
superficial refactoring, or simple reweighting is NOT allowed.

Please design a new algorithm by following these exact steps:

Step 1 -- Propose a New Algorithmic Concept
- Clearly describe the core mechanism of your algorithm.
- Focus on the fundamental reasoning principle.
- Keep it concise but precise.
- Wrap the description EXACTLY inside:
<concept>
...
</concept>

Step 2 — Implement the Algorithm  
- Implement your proposed algorithm using the exact Python function template provided below:
```python
{str(temp_func)}
```
- Do NOT modify the function name, arguments, or return type.
- Place all necessary `import` statements inside the function body.
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
