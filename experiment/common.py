# -*- coding: utf-8 -*-
"""Utilitários compartilhados pelas 3 pipelines do experimento -- espelha
a lógica de alinhamento por IoU do scorer.py (não importado diretamente
para manter experiment/ desacoplado do pipeline de submissão)."""
from typing import List, Tuple


def iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def match_spans(gold_spans: List[Tuple[int, int]], pred_spans: List[Tuple[int, int]],
                 threshold: float = 0.5):
    """Alinhamento guloso um-para-um por maior IoU primeiro. Retorna
    (matches, used_gold_idx, used_pred_idx) -- matches é lista de
    (gold_idx, pred_idx, iou)."""
    pairs = []
    for gi, g in enumerate(gold_spans):
        for pi, p in enumerate(pred_spans):
            v = iou(g, p)
            if v >= threshold:
                pairs.append((v, gi, pi))
    pairs.sort(key=lambda t: -t[0])
    used_g, used_p, matches = set(), set(), []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, v))
    return matches, used_g, used_p


def precision_recall_f1(n_gold: int, n_pred: int, n_matched: int):
    recall = n_matched / n_gold if n_gold else 0.0
    precision = n_matched / n_pred if n_pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def merge_overlapping_spans(spans_with_score: List[Tuple[int, int, float]]):
    """Recebe [(start, end, score)] possivelmente sobrepostos e devolve
    uma lista sem sobreposição, mantendo o de maior score primeiro
    (guloso) -- usado para colapsar janelas positivas adjacentes da
    Pipeline A num único span de citação."""
    ordered = sorted(spans_with_score, key=lambda t: -t[2])
    accepted = []
    for start, end, score in ordered:
        if any(start < e and end > s for s, e, _ in accepted):
            continue
        accepted.append((start, end, score))
    accepted.sort(key=lambda t: t[0])
    return accepted
