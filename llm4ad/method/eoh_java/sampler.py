from __future__ import annotations

import re
from typing import Tuple

from ...base import LLM


class EoH_Java_Sampler:
    def __init__(self, llm: LLM):
        self.llm = llm

    def get_thought_and_function(self, prompt: str) -> Tuple[str, str]:
        response = self.llm.draw_sample(prompt)
        thought = self.__class__.trim_thought_from_response(response)
        java_code = self.__class__.trim_java_from_response(response)
        return thought, java_code

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        """
        Extract the idea/thought from LLM response.
        Expected format: <<your idea here>>
        """
        try:
            pattern = r"<<(.*?)>>"
            match = re.search(pattern, response, re.DOTALL)
            return match.group(1).strip() if match else None
        except Exception:
            return None

    @classmethod
    def trim_java_from_response(cls, response: str) -> str | None:
        """
                Extract the Java implementation from LLM response.
                Expected format: [[JAVA_CODE_START ... JAVA_CODE_END]]
                """
        try:
            pattern = r"\[\[\s*JAVA_CODE_START\s*(.*?)\s*JAVA_CODE_END\s*\]\]"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            return match.group(1).strip() if match else None
        except Exception:
            return None

    # def trim_java_from_response(cls, response: str) -> str | None:
    #     """
    #     Extract the Java implementation from LLM response and clean it.
    #
    #     This method performs two main steps:
    #     1. Extracts the code block from the primary delimiters: [[JAVA_CODE_START ... JAVA_CODE_END]]
    #     2. Removes optional Markdown-style code fences (e.g., ```java ... ```) from the extracted block.
    #     """
    #     try:
    #         # Step 1: Extract the main code block using the primary delimiters.
    #         primary_pattern = r"$$$$\s*JAVA_CODE_START\s*(.*?)\s*JAVA_CODE_END\s*$$$$"
    #         primary_match = re.search(primary_pattern, response, re.DOTALL | re.IGNORECASE)
    #
    #         if not primary_match:
    #             return None
    #
    #         java_code = primary_match.group(1).strip()
    #
    #         # Step 2: Clean up potential Markdown fences from the extracted code.
    #         # This pattern looks for an optional ```java at the start and ``` at the end.
    #         # It's non-greedy (.*?) to handle cases where ``` might appear inside comments.
    #         cleanup_pattern = r"^\s*```(?:java)?\s*(.*?)\s*```\s*$"
    #         cleanup_match = re.search(cleanup_pattern, java_code, re.DOTALL | re.IGNORECASE)
    #
    #         if cleanup_match:
    #             # If Markdown fences are found, return the content inside them.
    #             return cleanup_match.group(1).strip()
    #         else:
    #             # Otherwise, return the originally extracted code.
    #             return java_code
    #
    #     except Exception:
    #         return None

if __name__ == '__main__':
    response = """
    <<This version introduces adaptive cosine decay for smoother distance adjustment.>>
    [[JAVA_CODE_START
    public class DistAdjustment {
        // Improved implementation
    }
    JAVA_CODE_END]]
    """

    sampler = EoH_Java_Sampler(llm=None)  # llm 为你的 LLM 实例
    thought = sampler.trim_thought_from_response(response)
    java_code = sampler.trim_java_from_response(response)

    print("Thought:", thought)
    print("Java Code:", java_code)