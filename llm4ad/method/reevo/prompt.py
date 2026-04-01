from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *
from ...prompts import load_prompt_text, render_prompt


class ReEvoPrompt:

    @classmethod
    def get_pop_init_prompt(cls, task_prompt: str, template_function: Function) -> str:
        template_function = copy.deepcopy(template_function)
        func_name = template_function.name
        template_function.name = f'{template_function.name}_v1'
        return '\n'.join([
            load_prompt_text('reevo', 'pop_init_system.txt'),
            render_prompt(
                'reevo',
                'pop_init_user.txt',
                task_prompt=task_prompt,
                template_function=template_function,
                func_name_v1=f'{func_name}_v1',
                func_name_v2=f'{func_name}_v2',
            ),
        ])

    @classmethod
    def get_short_term_reflection_prompt(cls, task_prompt: str, indivs: List[Function]) -> str:
        assert len(indivs) == 2
        indivs = copy.deepcopy(indivs)
        indivs.sort(key=lambda function: function.score)
        return '\n'.join([
            load_prompt_text('reevo', 'short_term_reflection_system.txt'),
            render_prompt(
                'reevo',
                'short_term_reflection_user.txt',
                function_name=indivs[0].name,
                task_prompt=task_prompt,
                worse_code=indivs[0],
                better_code=indivs[1],
            ),
        ])

    @classmethod
    def get_crossover_prompt(cls, task_prompt: str, short_term_reflection_prompt: str, indivs: List[Function], ) -> str:
        indivs = copy.deepcopy(indivs)
        indivs.sort(key=lambda function: function.score)
        func_name = indivs[0].name
        indivs[0].name = f'{indivs[0].name}_v0'
        indivs[1].name = f'{indivs[1].name}_v1'
        return '\n'.join([
            load_prompt_text('reevo', 'crossover_system.txt'),
            render_prompt(
                'reevo',
                'crossover_user.txt',
                task_prompt=task_prompt,
                worse_code=indivs[0],
                better_code=indivs[1],
                short_term_reflection=short_term_reflection_prompt,
                func_name_v2=f'{func_name}_v2',
            ),
        ])

    @classmethod
    def get_long_term_reflection_prompt(cls, task_prompt: str, prior_long_term_reflection: str, new_short_term_reflections: List[str]) -> str:
        new_short_term_reflections = '\n'.join(new_short_term_reflections)
        return '\n'.join([
            load_prompt_text('reevo', 'long_term_reflection_system.txt'),
            render_prompt(
                'reevo',
                'long_term_reflection_user.txt',
                task_prompt=task_prompt,
                prior_long_term_reflection=prior_long_term_reflection,
                new_short_term_reflections=new_short_term_reflections,
            ),
        ])

    @classmethod
    def get_elist_mutation_prompt(cls, task_prompt: str, long_term_reflection_prompt: str, elite_function: Function) -> str:
        elite_function = copy.deepcopy(elite_function)
        func_name = elite_function.name
        elite_function.name = f'{elite_function.name}_v1'
        return '\n'.join([
            load_prompt_text('reevo', 'elite_mutation_system.txt'),
            render_prompt(
                'reevo',
                'elite_mutation_user.txt',
                task_prompt=task_prompt,
                long_term_reflection=long_term_reflection_prompt,
                elite_function=elite_function,
                func_name_v2=f'{func_name}_v2',
            ),
        ])
