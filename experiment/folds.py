# -*- coding: utf-8 -*-
"""Split 5-fold estratificado por nível, no nível de DOCUMENTO ORIGINAL
(gen_n1_001..013, gen_n2_001..013) -- toda variante sintética herda o
fold do documento que a originou, então nunca há vazamento entre
treino e teste do mesmo documento-base."""
import random

N_FOLDS = 5
SEED = 42


def make_folds(doc_ids):
    """doc_ids: iterável dos 26 ids originais (gen_n1_*/gen_n2_*).
    Retorna dict doc_id -> fold_index (0..N_FOLDS-1), estratificado por
    prefixo de nível."""
    rng = random.Random(SEED)
    by_level = {}
    for d in doc_ids:
        level = d.split("_")[1]  # "n1" ou "n2"
        by_level.setdefault(level, []).append(d)

    assignment = {}
    for level, ids in by_level.items():
        ids = sorted(ids)
        rng.shuffle(ids)
        for i, doc_id in enumerate(ids):
            assignment[doc_id] = i % N_FOLDS
    return assignment


def fold_of(doc_id, assignment):
    base = doc_id
    return assignment[base]
