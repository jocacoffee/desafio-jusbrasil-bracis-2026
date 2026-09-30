# -*- coding: utf-8 -*-
"""
Pipeline B -- LLM local (qwen2.5vl:latest via Ollama) como extrator de
menções vagas. Um prompt por documento de teste: definição da tarefa
(extraída do PDF §1) + poucos exemplos do fold de treino (few-shot) +
texto do documento -> JSON com spans encontrados.
"""
import difflib
import json
import re

from ollama_client import generate, extract_json

MODEL = "qwen2.5vl:latest"

TASK_DEFINITION = """Você analisa peças jurídicas brasileiras para achar MENÇÕES VAGAS a
jurisprudência ou lei: trechos que citam um precedente ou uma norma SEM
trazer NENHUM identificador que permita localizá-lo. "Identificador" é
QUALQUER NÚMERO -- número de processo, número de artigo, número de
súmula, ano -- OU o nome de um relator/ministro específico.

REGRA PRÁTICA: se o trecho tem algum número (mesmo incompleto, tipo
"nos autos de 2020") ou nome próprio de ministro/relator, ELE NÃO é
menção vaga -- não inclua, mesmo que pareça uma citação. Só inclua o
trecho que fica se você IMAGINAR removendo qualquer número/nome dele:
se sobrar uma frase vaga fazendo sentido sozinha, é candidata; se a
frase simplesmente não faz mais sentido sem aquele número, ela NÃO é
menção vaga."""

_GENERIC_EXAMPLES = [
    ("jurisprudência pacífica desta Corte", "jurisprudencia", True,
     "não tem nenhum número nem nome -- vago"),
    ("dispositivo legal de regência", "lei", True,
     "não diz qual artigo nem qual lei -- vago"),
    ("REsp nº 1.234.567/SP", "jurisprudencia", False,
     "tem número de processo -- NÃO é vago, mesmo sendo uma citação"),
    ("art. 5º da Constituição Federal", "lei", False,
     "tem número de artigo -- NÃO é vago"),
    ("julgado do STJ de 2021, relatado pelo Ministro Fulano", "jurisprudencia", False,
     "tem ano e nome de relator -- NÃO é vago (é ambíguo, mas tem identificador)"),
    ("Súmula 231 do STF", "jurisprudencia", False,
     "tem número de súmula -- NÃO é vago"),
]


def build_prompt(doc_text: str, few_shot):
    exemplos_genericos = "\n".join(
        f'- "{trecho}" -> {"VAGO" if is_vago else "NÃO é vago"} ({motivo})'
        for trecho, _tipo, is_vago, motivo in _GENERIC_EXAMPLES
    )
    exemplos_positivos = "\n".join(
        f'- "{ex["trecho"]}" (tipo: {ex["tipo"]})' for ex in few_shot
    )
    return f"""{TASK_DEFINITION}

Exemplos genéricos (para calibrar o que É e o que NÃO É vago):
{exemplos_genericos}

Mais exemplos de menções REALMENTE vagas já identificadas em outros
documentos (todos sem número, sem nome de relator):
{exemplos_positivos}

Agora analise o documento abaixo e liste SÓ as menções REALMENTE vagas
que encontrar (revise cada uma: ela tem algum número ou nome próprio
escondido? se tiver, NÃO inclua). Para cada uma, dê o trecho EXATO
(copie literalmente, sem alterar uma letra) e o tipo ("lei" ou
"jurisprudencia").

DOCUMENTO:
\"\"\"
{doc_text}
\"\"\"

Responda SOMENTE com um JSON: {{"citacoes": [{{"trecho": "...", "tipo": "lei"|"jurisprudencia"}}, ...]}}
Se não encontrar nenhuma, responda {{"citacoes": []}}.
"""


def _find_span(doc_text: str, trecho: str):
    """Localiza `trecho` em `doc_text`: match exato primeiro, senão a
    melhor janela aproximada (LLM às vezes normaliza espaços/acentos)."""
    idx = doc_text.find(trecho)
    if idx != -1:
        return idx, idx + len(trecho)
    # fallback aproximado: desliza uma janela do tamanho do trecho
    matcher = difflib.SequenceMatcher(None, doc_text, trecho)
    match = matcher.find_longest_match(0, len(doc_text), 0, len(trecho))
    if match.size < max(6, len(trecho) * 0.4):
        return None
    # expande a partir do match mais longo até cobrir ~len(trecho)
    start = max(0, match.a - match.b)
    end = min(len(doc_text), start + len(trecho))
    return start, end


def extract_spans(doc_text: str, few_shot, model: str = MODEL):
    prompt = build_prompt(doc_text, few_shot)
    raw = generate(model, prompt, timeout=180, temperature=0.1)
    try:
        data = extract_json(raw)
    except ValueError:
        return []
    out = []
    for item in data.get("citacoes", []):
        trecho = (item.get("trecho") or "").strip()
        tipo = item.get("tipo")
        if not trecho or tipo not in ("lei", "jurisprudencia"):
            continue
        span = _find_span(doc_text, trecho)
        if span is None:
            continue
        out.append((span[0], span[1], tipo, 1.0))
    out.sort(key=lambda t: t[0])
    return out
