# -*- coding: utf-8 -*-
"""
Pipeline A -- classificador leve (TF-IDF de n-gramas de caractere +
regressão logística) para detectar menções vagas de citação.

Treina por fold, só nos exemplos de treino (documentos de outros folds).
Na inferência, varre o documento de teste inteiro em janelas de
4-10 palavras, pontua cada janela, mantém as acima do limiar e resolve
sobreposição com common.merge_overlapping_spans.
"""
import re
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SkPipeline

from common import merge_overlapping_spans

_WORD_RE = re.compile(r"\S+")
WINDOW_SIZES = (4, 5, 6, 7, 8, 9, 10)
SCORE_THRESHOLD = 0.5


def _tokenize_with_offsets(text: str):
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def make_model():
    return SkPipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


def train(examples: List[dict]):
    """examples: cada um {"window_text": ..., "label": "vago_lei"|
    "vago_jurisprudencia"|"negativo"} (ver experiment/data/training_examples.jsonl,
    gerado por augment.py). Treina UM classificador multiclasse (3 classes)
    -- mais simples e mais informativo que dois binários separados."""
    texts = [e["window_text"] for e in examples]
    labels = [e["label"] for e in examples]
    model = make_model()
    model.fit(texts, labels)
    return model


def extract_spans(model, text: str) -> List[Tuple[int, int, str, float]]:
    """Varre o documento inteiro em janelas, pontua com o modelo treinado,
    devolve spans (start, end, tipo, confianca) não sobrepostos para as
    classes vago_lei/vago_jurisprudencia."""
    tokens = _tokenize_with_offsets(text)
    if not tokens:
        return []
    windows = []
    window_texts = []
    for span_len in WINDOW_SIZES:
        for i in range(0, len(tokens) - span_len + 1):
            start = tokens[i][0]
            end = tokens[i + span_len - 1][1]
            windows.append((start, end))
            window_texts.append(text[start:end])
    if not window_texts:
        return []

    proba = model.predict_proba(window_texts)
    classes = list(model.classes_)
    candidates = []  # (start, end, score, tipo)
    for (start, end), probs in zip(windows, proba):
        for tipo_label in ("vago_lei", "vago_jurisprudencia"):
            if tipo_label not in classes:
                continue
            score = probs[classes.index(tipo_label)]
            if score >= SCORE_THRESHOLD:
                candidates.append((start, end, score, tipo_label.replace("vago_", "")))

    # resolve sobreposição globalmente (entre tipos também), maior score primeiro
    candidates.sort(key=lambda t: -t[2])
    accepted = []
    for start, end, score, tipo in candidates:
        if any(start < e and end > s for s, e, _, _ in accepted):
            continue
        accepted.append((start, end, score, tipo))
    accepted.sort(key=lambda t: t[0])
    return [(s, e, tipo, float(score)) for s, e, score, tipo in accepted]
