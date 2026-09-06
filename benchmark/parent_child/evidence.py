"""Frozen-source evidence scoring, independent of chunk IDs and chapter-only labels."""

from __future__ import annotations

import re
from typing import Any, Sequence


def normalized(text: str) -> str:
    # MarkdownHeaderTextSplitter normalizes some whitespace. Keep punctuation/negation intact.
    return re.sub(r"\s+", "", text).casefold()


def validate_evidence(cases: Sequence[dict[str, Any]], corpus: Sequence[dict[str, Any]]) -> None:
    sources = {str(row["source"]): normalized(str(row["content"])) for row in corpus}
    for case in cases:
        if case.get("dataset") != "hard_20":
            continue
        groups = case.get("evidence_groups") or []
        if not groups:
            raise ValueError(f"{case['id']}: missing evidence_groups")
        for group in groups:
            if not group:
                raise ValueError(f"{case['id']}: empty evidence group")
            for alternative in group:
                quote = normalized(str(alternative.get("text") or ""))
                if not quote or quote not in sources.get(str(alternative.get("source")), ""):
                    raise ValueError(f"{case['id']}: evidence absent from snapshot: {alternative}")


def score_evidence(hits: Sequence[dict[str, Any]], groups: Sequence[Sequence[dict]]) -> dict[str, Any]:
    """AND between independent facts; OR between valid alternative sources of the same fact.

    Match only delivered/truncated parent bodies, never injected headings, child previews, or
    ground truth. A multi-fact question gets credit only for the facts actually present in context.
    """
    matched = []
    for index, alternatives in enumerate(groups):
        if any(
            hit.get("source") == gold.get("source")
            and normalized(str(gold["text"])) in normalized(str(hit.get("parent_text") or ""))
            for gold in alternatives
            for hit in hits
        ):
            matched.append(index)
    return {
        "recall": len(matched) / len(groups) if groups else None,
        "complete": float(len(matched) == len(groups)) if groups else None,
        "matched_groups": matched,
        "total_groups": len(groups),
    }
