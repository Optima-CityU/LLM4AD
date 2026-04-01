from __future__ import annotations

from pathlib import Path
from string import Template


PROMPTS_ROOT = Path(__file__).resolve().parent


def load_prompt_text(*relative_parts: str) -> str:
    prompt_path = PROMPTS_ROOT.joinpath(*relative_parts)
    with prompt_path.open('r', encoding='utf-8') as prompt_file:
        return prompt_file.read()


def render_prompt(*relative_parts: str, **kwargs) -> str:
    template = Template(load_prompt_text(*relative_parts))
    normalized_kwargs = {key: str(value) for key, value in kwargs.items()}
    return template.safe_substitute(normalized_kwargs)
