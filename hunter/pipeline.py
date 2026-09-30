# -*- coding: utf-8 -*-
"""Orquestra M1 (extração) + M3/M4 (resolução) por documento e monta o
JSON no formato do contrato de entrada/saída do desafio."""
from typing import Dict

from .db import Base
from .extract import find_all_candidates
from .resolve import resolve


def process_document(documento_id: str, texto: str, base: Base) -> Dict:
    citacoes = []
    for cand in find_all_candidates(texto):
        res = resolve(cand, base)
        entry = {
            "inicio": cand.inicio,
            "fim": cand.fim,
            "trecho": cand.trecho,
            "tipo": cand.tipo,
            "classificacao": res.classificacao,
            "confianca": round(res.confianca, 4),
        }
        if res.classificacao == "real" and res.id_canonico:
            entry["resolucao"] = {"id_canonico": res.id_canonico}
        else:
            entry["resolucao"] = None
        citacoes.append(entry)
    return {"documento_id": documento_id, "citacoes": citacoes}
