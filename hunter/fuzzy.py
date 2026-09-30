# -*- coding: utf-8 -*-
"""
Casamento difuso de frases fixas (menções vagas de jurisprudência/lei)
contra texto ruidoso do nível 2. As frases vagas do gerador vêm de um
vocabulário fechado e curto (~15 frases), mas o nível 2 aplica ruído OCR
letra-a-letra também em palavras comuns, não só em identificadores
("entendirnento" por "entendimento", "jurisprudêneia" por
"jurisprudência") -- por isso um regex exato falha nesses casos e a
comparação usa distância de edição tolerante (difflib, stdlib).
"""
import difflib
import re
from typing import List, Tuple

from .normalize import normalize_word

_WORD_RE = re.compile(r"\S+")


def _tokenize_with_offsets(text: str) -> List[Tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]


def fuzzy_find_all(text: str, phrases: List[str], threshold: float = 0.82):
    """Retorna lista de (start, end, phrase) para TODAS as ocorrências de
    cada frase em `phrases` que aparecerem em `text` com razão de
    similaridade >= threshold (a mesma frase vaga pode se repetir várias
    vezes no mesmo documento -- cada uma é uma citação separada), varrendo
    janelas de tamanho (em palavras) próximo ao da frase. Não sobrepõe
    matches entre si (guloso por melhor score primeiro, depois pelas mais
    à esquerda)."""
    tokens = _tokenize_with_offsets(text)
    candidates = []
    for phrase in phrases:
        phrase_norm = normalize_word(phrase)
        n_words = len(phrase.split())
        for i in range(len(tokens)):
            for span_len in (n_words - 1, n_words, n_words + 1):
                if span_len <= 0:
                    continue
                j = i + span_len
                if j > len(tokens):
                    continue
                start = tokens[i][0]
                end = tokens[j - 1][1]
                window_norm = normalize_word(text[start:end])
                ratio = difflib.SequenceMatcher(None, phrase_norm, window_norm).ratio()
                if ratio >= threshold:
                    candidates.append((ratio, start, end, phrase))
    # resolve sobreposições globalmente: maior score primeiro; entre
    # ocorrências da MESMA frase em posições diferentes (score empatado),
    # a ordem por posição não importa pois não se sobrepõem entre si.
    candidates.sort(key=lambda r: -r[0])
    accepted = []
    for ratio, start, end, phrase in candidates:
        if any(start < e and end > s for s, e, _ in accepted):
            continue
        accepted.append((start, end, phrase))
    accepted.sort(key=lambda r: r[0])
    return accepted
