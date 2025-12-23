from __future__ import annotations

from ...tools.llm.llm_api_https import HttpsApi

import logging
import re
from abc import abstractmethod
from llamea.solution import Solution
from llamea.utils import NoCodeException, apply_code_delta

try:
    from ConfigSpace import ConfigurationSpace
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    ConfigurationSpace = None

class LlameaLLM(HttpsApi):
    def __init__(self,
                 host,
                 key,
                 model,
                 timeout=60,
                 code_pattern=None,
                 name_pattern=None,
                 desc_pattern=None,
                 cs_pattern=None,
                 logger=None,
                 **kwargs):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
        """
        super().__init__(host, key, model, timeout, **kwargs)
        self.model = model
        self.query = self.draw_sample

        self.base_url = host
        self.api_key = key
        self.model = model
        self.logger = logger
        self.log = self.logger is not None
        self.code_pattern = (
            code_pattern
            if code_pattern is not None
            else r"```(?:python|diff)?\n(.*?)\n```"
        )
        self.name_pattern = (
            name_pattern
            if name_pattern != None
            else "class\\s*(\\w*)(?:\\(\\w*\\))?\\:"
        )
        self.desc_pattern = (
            desc_pattern if desc_pattern != None else r"#\s*Description\s*:\s*(.*)"
        )
        self.cs_pattern = (
            cs_pattern
            if cs_pattern != None
            else r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        )

    # @abstractmethod
    # def query(self, session: list):
    #     """
    #     Sends a conversation history to the configured model and returns the response text.
    #
    #     Args:
    #         session (list of dict): A list of message dictionaries with keys
    #             "role" (e.g. "user", "assistant") and "content" (the message text).
    #
    #     Returns:
    #         str: The text content of the LLM's response.
    #     """
    #     pass

    def set_logger(self, logger):
        """
        Sets the logger object to log the conversation.

        Args:
            logger (Logger): A logger object to log the conversation.
        """
        self.logger = logger
        self.log = True

    def sample_solution(
        self,
        session_messages: list,
        parent_ids: list | None = None,
        HPO: bool = False,
        base_code: str | None = None,
        diff_mode: bool = False,
    ):
        """Generate or mutate a solution using the language model.

        Args:
            session_messages: Conversation history for the LLM.
            parent_ids: Identifier(s) of parent solutions.
            HPO: If ``True``, attempt to extract a configuration space.
            base_code: Existing code to patch when ``diff_mode`` is ``True``.
            diff_mode: When ``True``, interpret the LLM response as a unified
                diff patch to apply to ``base_code`` rather than full source
                code.

        Returns:
            tuple: A tuple containing the new algorithm code, its class name, its full descriptive name and an optional configuration space object.

        Raises:
            NoCodeException: If the language model fails to return any code.
            Exception: Captures and logs any other exceptions that occur during the interaction.
        """
        if parent_ids is None:
            parent_ids = []

        if self.log:
            self.logger.log_conversation(
                "client", "\n".join([d["content"] for d in session_messages])
            )

        message = self.query(session_messages)

        if self.log:
            self.logger.log_conversation(self.model, message)

        code_block = self.extract_algorithm_code(message)
        code = ""
        success = False  # <- Flag to Implement fall back to code block update, when LLM fails to adhere to diff mode.
        if diff_mode:
            if base_code is None:
                base_code = ""
            else:
                code, success, similarity = apply_code_delta(code_block, base_code)
                print(
                    f"\t Diff application {'un' if not success else ''}successful, Similarity {similarity * 100:.2f}%."
                )
        else:
            code = code_block

        if diff_mode and not success:
            print("\t\t Falling back to code replace.")
            code = code_block

        name = re.findall(
            r"(?:def|class)\s*(\w*).*\:",
            code,
            re.IGNORECASE,
        )[0]
        desc = self.extract_algorithm_description(message)
        cs = None
        if HPO and ConfigurationSpace is not None:
            cs = self.extract_configspace(message)
        new_individual = Solution(
            name=name,
            description=desc,
            configspace=cs,
            code=code,
            parent_ids=parent_ids,
        )

        return new_individual

    def extract_configspace(self, message):
        """
        Extracts the configuration space definition in json from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            ConfigSpace: Extracted configuration space object.
        """
        if ConfigurationSpace is None:  # pragma: no cover - optional dependency
            return None
        pattern = r"space\s*:\s*\n*```\n*(?:python)?\n(.*?)\n```"
        c = None
        for m in re.finditer(pattern, message, re.DOTALL | re.IGNORECASE):
            try:
                c = ConfigurationSpace(eval(m.group(1)))
            except Exception as e:  # pragma: no cover - best effort
                logging.info(e)
                pass
        return c

    def extract_algorithm_code(self, message):
        """
        Extracts algorithm code from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm code.

        Returns:
            str: Extracted algorithm code.

        Raises:
            NoCodeException: If no code block is found within the message.
        """
        match = re.search(self.code_pattern, message, re.DOTALL | re.IGNORECASE)
        if match:
            code = match.group(1)
            main_guard_pattern = re.compile(
                r"^\s*if __name__\s*={1,2}\s*['\"]__main__['\"]\s*:\s*$",
                re.MULTILINE,
            )
            guard_match = main_guard_pattern.search(code)
            if guard_match:
                code = code[: guard_match.start()].rstrip()
            return code
        else:
            raise NoCodeException

    def extract_algorithm_description(self, message):
        """
        Extracts algorithm description from a given message string using regular expressions.

        Args:
            message (str): The message string containing the algorithm name and code.

        Returns:
            str: Extracted algorithm name or empty string.
        """
        pattern = r"#\s*Description\s*:\s*(.*)"
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
        else:
            return ""