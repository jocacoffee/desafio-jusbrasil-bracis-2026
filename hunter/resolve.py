# -*- coding: utf-8 -*-
"""
M3/M4 -- resolução de um Candidate contra a base canônica.

Regra (kaggle_metric.py + dataset final, 30/09/2026): cada citação real
resolve para EXATAMENTE UM registro -- id_canonico na submissão é sempre
um único doc_id, nunca um conjunto (o gabarito é quem pode aceitar mais de
um id por citação; a nossa predição só precisa acertar um deles). A
cardinalidade da consulta dá a classe diretamente:
    0 registros distintos -> inventada
    1 registro distinto   -> real (id_canonico = id desse registro)
    2+ registros distintos, sem critério de desempate -> incompleta

"Distintos" é por `id` (o doc_id Jusbrasil). Duas linhas do acervo que
compartilham número (ex.: acórdão + embargos de declaração do mesmo
feito) contam como 2 candidatos -- a organização confirmou que nenhuma
citação real do dataset aponta para um feito assim duplicado, então essa
ambiguidade só aparece quando a citação não é claramente resolvível.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .db import Base
from .extract import Candidate
from .normalize import normalize_word


@dataclass
class Resolucao:
    classificacao: str  # real | inventada | incompleta
    tipo: str
    id_canonico: Optional[str] = None  # um único doc_id, nunca um conjunto
    confianca: float = 0.5


def resolve_jurisprudencia(cand: Candidate, base: Base) -> Resolucao:
    if cand.kind == "sumula":
        return _resolve_sumula(cand, base)
    if cand.kind == "tema":
        # base nunca modela "tema de repercussão geral" como registro --
        # garantia estrutural, não achismo.
        return Resolucao("inventada", "jurisprudencia", None, confianca=0.95)
    if cand.kind == "relator_ano":
        return _resolve_relator_ano(cand, base)
    if cand.kind == "numero":
        return _resolve_numero(cand, base)
    return Resolucao("incompleta", "jurisprudencia", None, confianca=0.3)


def _resolve_numero(cand: Candidate, base: Base) -> Resolucao:
    registros = base.feitos_por_numero(cand.raw_digits)
    ids = sorted({r.id for r in registros})
    if len(ids) == 0:
        return Resolucao("inventada", "jurisprudencia", None, confianca=0.88)
    if len(ids) == 1:
        return Resolucao("real", "jurisprudencia", str(ids[0]), confianca=0.97)
    # mesmo número, 2+ registros -- acontece quando o acervo tem mais de
    # uma decisão para o mesmo feito (ex.: acórdão original + embargos de
    # declaração/divergência posteriores, mesmo número de processo). A
    # PRÓPRIA citação costuma nomear a classe da decisão referida ("AgInt
    # no Recurso Especial" vs. "AgInt nos Embargos de Divergência..."), que
    # é justamente o texto que também abre cada um desses registros --
    # usa isso como critério de desempate antes de desistir para incompleta.
    escolhido = _desambiguar_por_classe(cand.trecho, ids, base)
    if escolhido is not None:
        return Resolucao("real", "jurisprudencia", str(escolhido), confianca=0.8)
    return Resolucao("incompleta", "jurisprudencia", None, confianca=0.75)


def _desambiguar_por_classe(trecho_citado: str, ids, base: Base,
                             margem_minima: float = 0.15,
                             piso_minimo: float = 0.3) -> Optional[int]:
    """Compara a classe processual do texto citado contra a classe própria
    de cada registro candidato (ambas normalizadas, dígitos fora -- só a
    terminologia importa) e escolhe o melhor SE ele se destacar claramente
    do segundo colocado; caso contrário, None (segue ambíguo)."""
    import difflib

    alvo = _normalizar_classe(trecho_citado)
    pontuados = []
    for i in ids:
        proprio = base.own_trecho_by_id.get(i)
        if not proprio:
            continue
        score = difflib.SequenceMatcher(None, alvo, _normalizar_classe(proprio)).ratio()
        pontuados.append((score, i))
    if not pontuados:
        return None
    pontuados.sort(key=lambda t: -t[0])
    melhor_score, melhor_id = pontuados[0]
    segundo_score = pontuados[1][0] if len(pontuados) > 1 else 0.0
    if melhor_score >= piso_minimo and (melhor_score - segundo_score) >= margem_minima:
        return melhor_id
    return None


def _normalizar_classe(trecho: str) -> str:
    """Remove dígitos e pontuação de um trecho de citação/cabeçalho,
    deixando só a terminologia da classe processual para comparação."""
    import re as _re
    sem_digitos = _re.sub(r"[\d/º°\-.,]+", " ", trecho)
    return normalize_word(sem_digitos)


def _resolve_relator_ano(cand: Candidate, base: Base) -> Resolucao:
    if not cand.relator:
        return Resolucao("incompleta", "jurisprudencia", None, confianca=0.7)
    registros = base.feitos_por_relator_ano(cand.tribunal, cand.ano, cand.relator)
    ids = sorted({r.id for r in registros})
    if len(ids) == 0:
        return Resolucao("inventada", "jurisprudencia", None, confianca=0.65)
    if len(ids) == 1:
        return Resolucao("real", "jurisprudencia", str(ids[0]), confianca=0.65)
    return Resolucao("incompleta", "jurisprudencia", None, confianca=0.88)


def _resolve_sumula(cand: Candidate, base: Base) -> Resolucao:
    m_num = cand.numero
    tribunal = cand.tribunal

    candidatos = [s for s in base.sumula_records
                  if (tribunal is None or s["tribunal"] == tribunal)]
    if tribunal is None:
        # sem tribunal explícito na citação: não é buscável o suficiente
        return Resolucao("incompleta", "jurisprudencia", None, confianca=0.85)

    if m_num is None:
        # tem tribunal mas não deu pra ler o número -- ambíguo se houver
        # mais de uma súmula desse tribunal na base
        if len(candidatos) == 1:
            return Resolucao("real", "jurisprudencia",
                              str(candidatos[0]["registro"].id), confianca=0.5)
        return Resolucao("incompleta", "jurisprudencia", None, confianca=0.6)

    matches = [c for c in candidatos if c["numero"] == m_num]
    if len(matches) == 0:
        return Resolucao("inventada", "jurisprudencia", None, confianca=0.92)
    if len(matches) == 1:
        return Resolucao("real", "jurisprudencia",
                          str(matches[0]["registro"].id), confianca=0.97)
    return Resolucao("incompleta", "jurisprudencia", None, confianca=0.7)


# ---------------------------------------------------------------------
# Lei / dispositivo
# ---------------------------------------------------------------------
from . import patterns as pt  # noqa: E402


def resolve_lei(cand: Candidate, base: Base) -> Resolucao:
    if cand.artigo_num is None:
        return Resolucao("incompleta", "lei", None, confianca=0.85)

    codigo_citado = _extract_codigo_key(cand.codigo_clause)
    matches = [d for d in base.dispositivo_records if d["artigo"] == cand.artigo_num]

    if len(matches) == 0:
        return Resolucao("inventada", "lei", None, confianca=0.9)

    if codigo_citado is None:
        if len(matches) == 1:
            return Resolucao("real", "lei", str(matches[0]["registro"].id),
                              confianca=0.65)
        return Resolucao("incompleta", "lei", None, confianca=0.6)

    exact = [d for d in matches if d["codigo"] == codigo_citado]
    if len(exact) == 1:
        return Resolucao("real", "lei", str(exact[0]["registro"].id), confianca=0.97)
    if len(exact) == 0:
        # número existe na cobertura, mas para um código diferente do citado
        return Resolucao("inventada", "lei", None, confianca=0.92)
    return Resolucao("incompleta", "lei", None, confianca=0.6)


def _extract_codigo_key(clause: str) -> Optional[str]:
    """Identifica qual código foi citado a partir do trecho livre que
    segue "do"/"da" no dispositivo (ex.: "Constituição Fedcral", "CPC").
    Tolerante a ruído OCR letra-a-letra do nível 2 (comparação difusa via
    difflib, não só substring exata) -- ver patterns.ARTIGO_RE."""
    import difflib
    from .patterns import CODIGO_SYNONYMS

    norm = normalize_word(clause)
    if not norm:
        return None

    best_key, best_score = None, 0.0
    for key, synonyms in CODIGO_SYNONYMS.items():
        for syn in synonyms:
            syn_norm = normalize_word(syn)
            if syn_norm in norm:
                score = 1.0
            else:
                prefix = norm[:len(syn_norm) + 4]
                score = difflib.SequenceMatcher(None, syn_norm, prefix).ratio()
            if score > best_score:
                best_key, best_score = key, score

    if best_score < 0.72:
        return None
    if best_key == "lei_13105_2015":
        return "cpc"
    return best_key


def resolve(cand: Candidate, base: Base) -> Resolucao:
    if cand.tipo == "lei":
        return resolve_lei(cand, base)
    return resolve_jurisprudencia(cand, base)
