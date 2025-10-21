import re
from typing import Tuple
from llm4ad.base import SampleTrimmer, Function, Program
from llamea import LLM

class LLaMEASampler:
    def __init__(self, llm: LLM):
        self.llm = llm
        self.draw_samples = self.llm.query
