from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class IEFuncItem:
    name: str
    signature: str
    docstring: str
    source_code: str
    score_sum: float = 0.0
    score_count: int = 0
    use_count: int = 0
    created_generation: int = 0
    last_used_generation: int = -1

    @property
    def mean_score(self) -> float:
        if self.score_count <= 0:
            return 0.0
        return self.score_sum / self.score_count


class IEFuncBank:
    """Dynamic helper-function bank for EoH-IE."""

    def __init__(self) -> None:
        self._items: Dict[str, IEFuncItem] = {}

    def __len__(self) -> int:
        return len(self._items)

    def get_names(self) -> List[str]:
        return list(self._items.keys())

    def add_or_update(
        self,
        *,
        signature: str,
        docstring: str,
        source_code: str,
        generation: int,
        init_score: float = 0.0,
    ) -> bool:
        name = self._extract_name(signature, source_code)
        if not name:
            return False

        signature = (signature or "").strip()
        docstring = (docstring or "").strip()
        source_code = self._sanitize_source(source_code)
        if not signature or not source_code:
            return False

        old = self._items.get(name)
        if old is None:
            self._items[name] = IEFuncItem(
                name=name,
                signature=signature,
                docstring=docstring,
                source_code=source_code,
                score_sum=float(init_score),
                score_count=1 if init_score is not None else 0,
                created_generation=generation,
            )
        else:
            old.signature = signature
            old.docstring = docstring
            old.source_code = source_code
        return True

    def update_usage(self, used_names: List[str], sample_score: Optional[float], generation: int) -> None:
        if not used_names:
            return
        for name in set(used_names):
            item = self._items.get(name)
            if item is None:
                continue
            item.use_count += 1
            item.last_used_generation = generation
            if sample_score is not None:
                item.score_sum += float(sample_score)
                item.score_count += 1

    def keep_top(self, max_size: int) -> List[str]:
        if max_size <= 0 or len(self._items) <= max_size:
            return []

        ranked = sorted(
            self._items.values(),
            key=lambda x: (x.mean_score, x.use_count, -x.created_generation),
            reverse=True,
        )
        keep = {x.name for x in ranked[:max_size]}
        removed = [name for name in list(self._items.keys()) if name not in keep]
        for name in removed:
            self._items.pop(name, None)
        return removed

    def top_items(self, k: int) -> List[IEFuncItem]:
        ranked = sorted(
            self._items.values(),
            key=lambda x: (x.mean_score, x.use_count, -x.created_generation),
            reverse=True,
        )
        return ranked[:max(k, 0)]

    def build_prompt_context(self, k: int) -> str:
        items = self.top_items(k)
        if not items:
            return ""
        return "\n\n".join(item.source_code.strip() for item in items if item.source_code.strip())

    @staticmethod
    def extract_called_funcs_from_code(code: str, candidate_names: List[str]) -> List[str]:
        if not code or not candidate_names:
            return []

        used = set()
        candidates = set(candidate_names)

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in candidates:
                        used.add(node.func.id)
                    elif isinstance(node.func, ast.Attribute) and node.func.attr in candidates:
                        used.add(node.func.attr)
            return sorted(used)
        except Exception:
            for name in candidates:
                if re.search(rf"\b{name}\s*\(", code):
                    used.add(name)
            return sorted(used)

    @staticmethod
    def _extract_name(signature: str, source_code: str) -> Optional[str]:
        for text in (signature or "", source_code or ""):
            m = re.search(r"(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", text)
            if m:
                return m.group(1)
        return None

    @staticmethod
    def _sanitize_source(source_code: str) -> str:
        lines = []
        for ln in (source_code or "").splitlines():
            if ln.strip().startswith("from __future__ import"):
                continue
            lines.append(ln)
        return "\n".join(lines).strip()
