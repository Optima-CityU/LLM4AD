from __future__ import annotations

import copy
from typing import List, Dict

from ...base import *


class EoHPrompt:
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
    def get_prompt_i1(cls, task_prompt: str, template_function: str):
        # create prompt content
        prompt_content = f'''\
{task_prompt}

Your task is to write a new "ruin" operator in Java for solving the CVRP. You must use the following provided Java implementation template and adhere strictly to the class summaries.
-------------------------
{template_function}
-------------------------

1. Briefly describe your idea in **one concise sentence**, 
   enclosed within << >>.
2. Then, design the Java implementation. The implementation must be enclosed within double square brackets [[ ]] as follows:
   [[JAVA_CODE_START
   // Your complete improved Java code here
   JAVA_CODE_END]]

'''
        return prompt_content

    # ！！ TODO: e1, e2, m1, m2 prompt构造

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[JavaScripts], template_function: str | None=None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indivs_prompt += f'No. {i + 1} operator and the corresponding java script are:\n{indi.algorithm}\n{str(indi)}'
            indivs_prompt += '\n'
            indivs_prompt += '-' * 20
            indivs_prompt += '\n'
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing operators with their java scripts as follows:
{indivs_prompt}
Please help me create a new operator that has a totally different form from the given ones. You must use the provided Java implementation template and adhere strictly to the class summaries.
-------------------------
{template_function}
-------------------------
1. Briefly describe your idea in **one concise sentence**, 
   enclosed within << >>.
2. Then, design the Java implementation. The implementation must be enclosed within double square brackets [[ ]] as follows:
   [[JAVA_CODE_START
   // Your complete improved Java code here
   JAVA_CODE_END]]

'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[JavaScripts], template_function: str | None=None):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        # create prompt content for all individuals
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indivs_prompt += f'No. {i + 1} operator and the corresponding java script are:\n{indi.algorithm}\n{str(indi)}'
            indivs_prompt += '\n'
            indivs_prompt += '-' * 20
            indivs_prompt += '\n'
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have {len(indivs)} existing operators with their java scripts as follows:
{indivs_prompt}
Please help me create a new operator that has a totally different form from the given ones but can be motivated from them. You must use the provided Java implementation template and adhere strictly to the class summaries.
-------------------------
{template_function}
------------------------- 
1. Firstly, identify the common backbone idea in the provided operators. 
2. Secondly, based on the backbone idea describe your new operator in one sentence. The description must be inside within boxed <<Your backbone idea here>>.
3. Thirdly, design the Java implementation. The implementation must be enclosed within double square brackets [[ ]] as follows:
   [[JAVA_CODE_START
   // Your complete improved Java code here
   JAVA_CODE_END]]

'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: JavaScripts, template_function: str | None=None):
        assert hasattr(indi, 'algorithm')
        # template
        # temp_func = copy.deepcopy(template_function)
        # temp_func.body = ''

        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one operator with its java script as follows. Description:
{indi.algorithm}
-------------------
Java script:
{str(indi)}
-------------------
Please assist me in creating a new operator that has a different form but can be a modified version of the algorithm provided. You must use the provided Java implementation template and adhere strictly to the class summaries.
-------------------------
{template_function}
-------------------------
1. Briefly describe your idea in **one concise sentence**, 
   enclosed within << >>.
2. Then, design the Java implementation. The implementation must be enclosed within double square brackets [[ ]] as follows:
   [[JAVA_CODE_START
   // Your complete improved Java code here
   JAVA_CODE_END]]

'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: JavaScripts, template_function: str | None=None):
        assert hasattr(indi, 'algorithm')
        # template
        # temp_func = copy.deepcopy(template_function)
        # temp_func.body = ''
        # create prmpt content
        prompt_content = f'''{task_prompt}
I have one operator with its java script as follows. Description:
{indi.algorithm}
-------------------
Java script:
{str(indi)}
-------------------
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided. You must use the provided Java implementation template and adhere strictly to the class summaries.
-------------------------
{template_function}
-------------------------
1. Briefly describe your idea in **one concise sentence**, 
   enclosed within << >>.
2. Then, design the Java implementation. The implementation must be enclosed within double square brackets [[ ]] as follows:
   [[JAVA_CODE_START
   // Your complete improved Java code here
   JAVA_CODE_END]]

'''
        return prompt_content
