from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *
from ...prompts import load_prompt_text, render_prompt
from .func_ruin import LHNSFunction, LHNSFunctionRuin

class LHNSPrompt:
    @classmethod
    def create_instruct_prompt(cls, prompt: str) -> List[Dict]:
        content = [
            {'role': 'system', 'message': cls.get_system_prompt()},
            {'role': 'user', 'message': prompt}
        ]
        return content

    @classmethod
    def get_system_prompt(cls) -> str:
        return load_prompt_text('lhns', 'system.txt')

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: LHNSFunction):
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content
        return render_prompt(
            'lhns',
            'prompt_i1.txt',
            task_prompt=task_prompt,
            template_function=temp_func,
        )

    @classmethod
    def get_prompt_merge(cls, task_prompt: str, indi: LHNSFunction, prev_indi: LHNSFunction, template_function: LHNSFunction):
        assert hasattr(indi, 'algorithm')
        
        indi = LHNSFunctionRuin.merge_features(indi, prev_indi.features)
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prompt content for all individuals
        indivs_prompt = ''
        indi.docstring = ''
        indivs_prompt += f'No. A algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}'
        indivs_prompt += f'No. B algorithm is:\n{indi.algorithm}'
        # create prmpt content
        return render_prompt(
            'lhns',
            'prompt_merge.txt',
            task_prompt=task_prompt,
            indivs_prompt=indivs_prompt,
            template_function=temp_func,
        )

    @classmethod
    def get_prompt_rr(cls, task_prompt: str, indi: LHNSFunction, cooling_rate: float, template_function: LHNSFunction):
        assert hasattr(indi, 'algorithm')
        indi, number_of_delete = LHNSFunctionRuin.delete_function_snips(indi, cooling_rate)
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''

        # create prmpt content
        return render_prompt(
            'lhns',
            'prompt_rr.txt',
            task_prompt=task_prompt,
            algorithm=indi.algorithm,
            code=indi,
            number_of_delete=number_of_delete,
            template_function=temp_func,
        )

    @classmethod
    def get_prompt_m(cls, task_prompt: str, indi: LHNSFunction, template_function: LHNSFunction):
        assert hasattr(indi, 'algorithm')
        # template
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        # create prmpt content
        return render_prompt(
            'lhns',
            'prompt_m.txt',
            task_prompt=task_prompt,
            algorithm=indi.algorithm,
            code=indi,
            template_function=temp_func,
        )
