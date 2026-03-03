"""关系查询规格解析工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RelationQuerySpec:
    raw: str
    is_structured: bool
    subject: Optional[str]
    predicate: Optional[str]
    object: Optional[str]
    error: Optional[str] = None


def parse_relation_query_spec(relation_spec: str) -> RelationQuerySpec:
    raw = str(relation_spec or "").strip()
    if not raw:
        return RelationQuerySpec(
            raw=raw,
            is_structured=False,
            subject=None,
            predicate=None,
            object=None,
            error="empty",
        )

    if "|" in raw:
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) < 2:
            return RelationQuerySpec(
                raw=raw,
                is_structured=True,
                subject=None,
                predicate=None,
                object=None,
                error="invalid_pipe_format",
            )
        return RelationQuerySpec(
            raw=raw,
            is_structured=True,
            subject=parts[0] or None,
            predicate=parts[1] or None,
            object=parts[2] if len(parts) > 2 and parts[2] else None,
        )

    if "->" in raw:
        parts = [p.strip() for p in raw.split("->") if p.strip()]
        if len(parts) >= 3:
            return RelationQuerySpec(
                raw=raw,
                is_structured=True,
                subject=parts[0],
                predicate=parts[1],
                object=parts[2],
            )
        if len(parts) == 2:
            return RelationQuerySpec(
                raw=raw,
                is_structured=True,
                subject=parts[0],
                predicate=None,
                object=parts[1],
            )
        return RelationQuerySpec(
            raw=raw,
            is_structured=True,
            subject=None,
            predicate=None,
            object=None,
            error="invalid_arrow_format",
        )

    # legacy: "subject predicate" 或 "subject predicate object"
    # 该形式歧义较高，归类为自然语言，由上层决定是否回退语义。
    parts = raw.split()
    if len(parts) >= 3:
        return RelationQuerySpec(
            raw=raw,
            is_structured=True,
            subject=parts[0],
            predicate=parts[1],
            object=" ".join(parts[2:]),
        )
    if len(parts) == 2:
        return RelationQuerySpec(
            raw=raw,
            is_structured=True,
            subject=parts[0],
            predicate=parts[1],
            object=None,
        )

    return RelationQuerySpec(
        raw=raw,
        is_structured=False,
        subject=None,
        predicate=None,
        object=None,
    )

