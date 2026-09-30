# -*- coding: utf-8 -*-
"""
Gera dados sintéticos para o experimento de detecção de "menção vaga"
(ver plano em /Users/joaocoelho/.claude/plans/swift-painting-flamingo.md).

1. Paráfrases das ~15 frases-molde de menção vaga (hunter/patterns.py),
   geradas por google/gemma-4-e4b (servidor LM Studio local na rede do
   usuário) -- modelo DIFERENTE do testado na Pipeline B (qwen2.5vl) para
   não vazar viés do modelo avaliado para dentro dos dados de treino/teste.
2. Cada paráfrase aceita substitui, in-place, uma ocorrência real de
   menção vaga em algum dos 26 documentos -- span recalculado, citações
   seguintes do mesmo documento deslocadas.
3. Ruído OCR leve aplicado com probabilidade configurável por cima da
   paráfrase.
4. Janelas de prosa comum (não-citação) amostradas como negativos.

Saída: experiment/data/augmented.jsonl
"""
import csv
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ollama_client import extract_json  # noqa: E402
from lmstudio_client import generate  # noqa: E402
import hunter.patterns as pt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TXT_DIR = ROOT / "txt"
GOLDENSET = ROOT / "goldenset.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"
INSTANCES_PATH = DATA_DIR / "instances.jsonl"
TRAINING_PATH = DATA_DIR / "training_examples.jsonl"

PARAPHRASE_MODEL = "google/gemma-4-e4b"
N_PARAPHRASES_PER_PHRASE = 8
N_NEGATIVES_PER_DOC = 8
NOISE_PROB = 0.5
RNG_SEED = 42

ALL_VAGUE_PHRASES = (
    [(p, "jurisprudencia") for p in pt.VAGUE_PHRASES_TEXT]
    + [(p, "lei") for p in pt.LEI_VAGUE_PHRASES_TEXT]
)


def load_docs():
    docs = {}
    for f in sorted(TXT_DIR.glob("*.txt")):
        docs[f.stem] = f.read_text(encoding="utf-8")
    return docs


def load_gold_rows():
    with open(GOLDENSET, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["inicio"] = int(r["inicio"])
        r["fim"] = int(r["fim"])
    return rows


def is_vago(row) -> bool:
    return row["classificacao"] == "incompleta" and not re.search(r"\d", row["trecho"])


# ---------------------------------------------------------------------
# 1) Paráfrases via Ollama
# ---------------------------------------------------------------------

def build_paraphrase_prompt(phrases):
    lista = "\n".join(f'{i+1}. "{p}"' for i, (p, _tipo) in enumerate(phrases))
    return f"""Você é um advogado brasileiro ajudando a gerar dados sintéticos para um
experimento de PLN sobre peças jurídicas em PORTUGUÊS DO BRASIL. As frases
abaixo são exemplos de MENÇÃO VAGA a jurisprudência ou lei -- citam algo
sem trazer identificador nenhum (sem número de processo, sem número de
artigo, sem número de súmula).

Para cada frase, gere {N_PARAPHRASES_PER_PHRASE} paráfrases -- TODAS 100% EM
PORTUGUÊS, jurídico formal e gramaticalmente correto -- que preservem
EXATAMENTE essa mesma vagueza (nenhum identificador, nenhum número). Cada
paráfrase deve ser ESPECÍFICA da frase original correspondente -- não
repita a mesma paráfrase genérica para frases diferentes, e não invente
palavras que não existem em português.

IMPORTANTÍSSIMO: a resposta inteira tem que estar em português do Brasil.
NENHUMA palavra em inglês, nem parcialmente (nem "law", "case", "legal",
"framework" etc.). Se não tiver certeza de uma palavra, prefira uma frase
mais simples, mas sempre em português.

Exemplo de paráfrase aceitável para "jurisprudência pacífica desta Corte":
"entendimento pacificado deste Tribunal"

Frases:
{lista}

Responda SOMENTE com um JSON no formato:
{{"1": ["paráfrase 1", "paráfrase 2", ...], "2": [...], ...}}
usando o número do item como chave. Não inclua nenhum número de processo,
artigo, súmula ou ano nas paráfrases.
"""


_ENGLISH_HINTS = {
    "the", "of", "in", "for", "law", "laws", "case", "legal", "regulation",
    "regulations", "framework", "guidelines", "and", "is", "are", "this",
    "that", "establishing", "procedure", "procedures", "provisions",
    "governing", "material", "measures", "specific", "related", "through",
    "formalized", "formalization", "instructions", "aspects", "issues",
}


def _looks_english(s: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", s.lower())
    hits = sum(1 for w in words if w in _ENGLISH_HINTS)
    return hits >= 1


def _accept_candidates(data, items, accepted_norms):
    out = []
    for i, (phrase, tipo) in enumerate(items):
        key = str(i + 1)
        for cand in data.get(key, []):
            cand = cand.strip().strip('"')
            if not cand:
                continue
            if re.search(r"\d", cand):
                continue  # menção vaga não pode ter identificador
            if len(cand) < 10 or len(cand) > 160:
                continue
            if _looks_english(cand):
                continue
            cand_norm = _normalize(cand)
            if cand_norm == _normalize(phrase):
                continue
            # quase-duplicata de algo já aceito p/ OUTRA frase-molde --
            # sinal de "resposta genérica de preenchimento", descarta
            if any(_similar(cand_norm, prev) for prev in accepted_norms):
                continue
            accepted_norms.append(cand_norm)
            out.append({"original": phrase, "tipo": tipo, "paraphrase": cand})
    return out


def fetch_paraphrases():
    """Uma chamada POR TIPO (lote menor deu qualidade melhor nos testes
    manuais que uma única chamada com as 15 frases juntas). Repete a
    chamada até 2x se a taxa de aceitação ficar baixa (o modelo às vezes
    responde em inglês apesar da instrução, sobretudo para "lei")."""
    out = []
    accepted_norms = []  # p/ filtro de quase-duplicata entre frases DIFERENTES
    for tipo, phrases in (("jurisprudencia", pt.VAGUE_PHRASES_TEXT),
                          ("lei", pt.LEI_VAGUE_PHRASES_TEXT)):
        items = [(p, tipo) for p in phrases]
        prompt = build_paraphrase_prompt(items)
        expected = N_PARAPHRASES_PER_PHRASE * len(items)
        best_batch = []
        for attempt in range(2):
            raw = generate(PARAPHRASE_MODEL, prompt, timeout=600, temperature=0.4)
            try:
                data = extract_json(raw)
            except ValueError:
                continue
            batch = _accept_candidates(data, items, list(accepted_norms))
            if len(batch) > len(best_batch):
                best_batch = batch
            if len(batch) >= expected * 0.5:
                break
            print(f"  [{tipo}] tentativa {attempt+1}: só {len(batch)}/{expected} aceitas, tentando de novo...")
        accepted_norms.extend(_normalize(b["paraphrase"]) for b in best_batch)
        out.extend(best_batch)
    return out


def _similar(a: str, b: str, threshold: float = 0.85) -> bool:
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= threshold


def _normalize(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------
# 2) Ruído OCR leve (reaproveita o modelo de ruído documentado no PDF)
# ---------------------------------------------------------------------

_ACCENT_DROP_PROB = 0.3
_LETTER_SWAPS = [("e", "c"), ("o", "0"), ("m", "rn")]  # amostra dos observados no dev set


def _strip_one_accent(s: str, pos: int) -> str:
    """Remove o acento do caractere em `pos`, se houver (á->a etc.)."""
    base = unicodedata.normalize("NFKD", s[pos])
    base = "".join(c for c in base if not unicodedata.combining(c))
    if not base:
        return s
    return s[:pos] + base + s[pos + 1:]


def apply_light_noise(text: str, rng: random.Random) -> str:
    out = text
    accented_positions = [i for i, c in enumerate(out)
                           if unicodedata.normalize("NFKD", c) != c]
    if accented_positions and rng.random() < _ACCENT_DROP_PROB:
        pos = rng.choice(accented_positions)
        out = _strip_one_accent(out, pos)
    # substituição de letra simples (ruído tipo OCR), no máximo 1 ocorrência
    for a, b in _LETTER_SWAPS:
        if rng.random() < 0.25 and a in out:
            pos = out.find(a)
            out = out[:pos] + b + out[pos + len(a):]
            break
    return out


# ---------------------------------------------------------------------
# 3) Substituição in-place com remapeamento de offsets
# ---------------------------------------------------------------------

def substitute_span(text, doc_rows, target_row, replacement):
    """Troca text[target_row.inicio:target_row.fim] por `replacement`.
    Retorna (novo_texto, novo_span, linhas_deslocadas) -- linhas com
    inicio >= fim original são deslocadas pela diferença de tamanho."""
    ini, fim = target_row["inicio"], target_row["fim"]
    new_text = text[:ini] + replacement + text[fim:]
    delta = len(replacement) - (fim - ini)
    new_span = (ini, ini + len(replacement))
    shifted = []
    for r in doc_rows:
        if r is target_row:
            continue
        if r["inicio"] >= fim:
            shifted.append({**r, "inicio": r["inicio"] + delta, "fim": r["fim"] + delta})
        else:
            shifted.append(r)
    return new_text, new_span, shifted


# ---------------------------------------------------------------------
# 4) Negativos: janelas de prosa comum fora de qualquer span rotulado
# ---------------------------------------------------------------------

def sample_negatives(doc_id, text, occupied_spans, rng, k=N_NEGATIVES_PER_DOC):
    tokens = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    negs = []
    attempts = 0
    while len(negs) < k and attempts < k * 20:
        attempts += 1
        span_len = rng.randint(4, 10)
        if len(tokens) <= span_len:
            break
        i = rng.randrange(0, len(tokens) - span_len)
        start = tokens[i][0]
        end = tokens[i + span_len - 1][1]
        if any(start < e and end > s for s, e in occupied_spans):
            continue
        negs.append((start, end, text[start:end]))
    return negs


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    rng = random.Random(RNG_SEED)
    docs = load_docs()
    gold = load_gold_rows()
    by_doc = {}
    for r in gold:
        by_doc.setdefault(r["documento_id"], []).append(r)

    print("Chamando", PARAPHRASE_MODEL, "para gerar paráfrases (pode levar alguns minutos)...")
    paraphrases = fetch_paraphrases()
    print(f"  {len(paraphrases)} paráfrases aceitas (de {N_PARAPHRASES_PER_PHRASE * len(ALL_VAGUE_PHRASES)} pedidas)")

    vago_rows_by_tipo = {"jurisprudencia": [], "lei": []}
    for r in gold:
        if is_vago(r):
            vago_rows_by_tipo[r["tipo"]].append(r)

    instances = []   # nível de documento -- para avaliação das 3 pipelines
    train_windows = []  # nível de janela -- só para treinar a Pipeline A

    # instância 1 por documento ORIGINAL (real, sem alteração)
    for doc_id, text in docs.items():
        vago_here = [{"inicio": r["inicio"], "fim": r["fim"], "tipo": r["tipo"]}
                     for r in by_doc.get(doc_id, []) if is_vago(r)]
        instances.append({
            "instance_id": doc_id,
            "base_doc_id": doc_id,
            "nivel": "1" if doc_id.startswith("gen_n1") else "2",
            "text": text,
            "vago_citations": vago_here,
            "kind": "real",
        })
        for c in vago_here:
            train_windows.append({
                "base_doc_id": doc_id,
                "window_text": text[c["inicio"]:c["fim"]],
                "label": "vago_" + c["tipo"],
            })
        occupied = [(r["inicio"], r["fim"]) for r in by_doc.get(doc_id, [])]
        for s, e, _snippet in sample_negatives(doc_id, text, occupied, rng):
            train_windows.append({
                "base_doc_id": doc_id,
                "window_text": text[s:e],
                "label": "negativo",
            })

    # 1 instância sintética por paráfrase aceita (substitui 1 menção vaga
    # real, desloca as demais citações do MESMO documento -- inclusive
    # outras menções vagas, que continuam válidas na nova posição)
    for i, p in enumerate(paraphrases):
        candidates = vago_rows_by_tipo[p["tipo"]]
        if not candidates:
            continue
        target = rng.choice(candidates)
        doc_id = target["documento_id"]
        text = docs[doc_id]
        doc_rows = by_doc[doc_id]
        replacement = p["paraphrase"]
        if rng.random() < NOISE_PROB:
            replacement = apply_light_noise(replacement, rng)
        new_text, new_span, shifted = substitute_span(text, doc_rows, target, replacement)
        assert new_text[new_span[0]:new_span[1]] == replacement

        vago_here = [{"inicio": new_span[0], "fim": new_span[1], "tipo": p["tipo"]}]
        for r in shifted:
            if is_vago(r):
                vago_here.append({"inicio": r["inicio"], "fim": r["fim"], "tipo": r["tipo"]})

        instance_id = f"synth_{doc_id}_{target['citacao_id']}_{i}"
        instances.append({
            "instance_id": instance_id,
            "base_doc_id": doc_id,
            "nivel": target["nivel"],
            "text": new_text,
            "vago_citations": vago_here,
            "kind": "synthetic",
            "original_phrase": p["original"],
        })
        train_windows.append({
            "base_doc_id": doc_id,
            "window_text": replacement,
            "label": "vago_" + p["tipo"],
        })

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(INSTANCES_PATH, "w", encoding="utf-8") as f:
        for rec in instances:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(TRAINING_PATH, "w", encoding="utf-8") as f:
        for rec in train_windows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_real_inst = sum(1 for i in instances if i["kind"] == "real")
    n_synth_inst = sum(1 for i in instances if i["kind"] == "synthetic")
    n_vago_total = sum(len(i["vago_citations"]) for i in instances)
    label_counts = {}
    for w in train_windows:
        label_counts[w["label"]] = label_counts.get(w["label"], 0) + 1
    print(f"Instâncias: {len(instances)} ({n_real_inst} reais + {n_synth_inst} sintéticas), "
          f"{n_vago_total} citações vago no total")
    print("Janelas de treino por classe:", label_counts)
    print("Salvo em", INSTANCES_PATH, "e", TRAINING_PATH)


if __name__ == "__main__":
    main()
