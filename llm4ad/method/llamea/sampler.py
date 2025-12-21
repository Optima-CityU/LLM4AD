import re
from typing import Tuple
from llm4ad.base import SampleTrimmer, Function, Program

class LLaMEASampler:
    def __init__(self, llm):
        self.llm = llm
        self.draw_samples = self.llm.query
