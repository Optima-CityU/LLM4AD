from __future__ import annotations

import re
from typing import Tuple, List, Dict

from .prompt import MLESPrompt
from ...base import LLM, SampleTrimmer, Function, Program


class MLESSampler:
    def __init__(self, sampler: LLM, template_program: str | Program):
        self._sampler = sampler
        self._template_program = template_program

    def get_thought_and_function(self, prompt: str, image64s: List = None, messages: List = None) -> tuple[
        str | None, Function | None, str]:
        response = self._sampler.draw_sample(prompt, image64s=image64s, messages=messages)
        thought = self.__class__.trim_thought_from_response(response)
        code = SampleTrimmer.trim_preface_of_function(response)
        function = SampleTrimmer.sample_to_function(code, self._template_program)
        return thought, function, response

    def get_image_description(self, prompt: str, image64s: List = None, messages: List = None):
        response = self._sampler.draw_sample(prompt, image64s=image64s, messages=messages)
        description = self.__class__.trim_description_from_response(response)
        return description, response

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'\{.*?\}'  # Compared with r'\{(.*)\}'
            bracketed_texts = re.findall(pattern, response)
            return bracketed_texts[0]
        except:
            return None

    @classmethod
    def trim_description_from_response(cls, response: str) -> str | None:
        try:
            # Adjust the pattern to match multi-line text within braces
            pattern = r'\{\{.*?\}\}'  # Non-greedy match for double braces
            bracketed_texts = re.findall(pattern, response, re.DOTALL)  # Use re.DOTALL to match newlines
            return bracketed_texts[0] if bracketed_texts else None
        except Exception as e:
            print(f"Error: {e}")
            return None
