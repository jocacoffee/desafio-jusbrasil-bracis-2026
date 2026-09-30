# -*- coding: utf-8 -*-
"""Pipeline 0 -- vocabulário fechado + fuzzy match (hunter/ já pronto,
nenhum treino). Mesma interface das outras duas pipelines."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hunter.fuzzy import fuzzy_find_all  # noqa: E402
from hunter import patterns as pt  # noqa: E402


def extract_spans(text: str):
    out = []
    for start, end, _phrase in fuzzy_find_all(text, pt.VAGUE_PHRASES_TEXT):
        out.append((start, end, "jurisprudencia", 1.0))
    for start, end, _phrase in fuzzy_find_all(text, pt.LEI_VAGUE_PHRASES_TEXT):
        out.append((start, end, "lei", 1.0))
    out.sort(key=lambda t: t[0])
    return out
