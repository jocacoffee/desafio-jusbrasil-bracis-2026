# -*- coding: utf-8 -*-
"""
Stress-test de generalização: aplica o MODELO DE RUÍDO DOCUMENTADO no PDF
do desafio (não paráfrase semântica, que já mostramos ter retorno baixo)
de forma mais exaustiva do que os 26 documentos de dev mostram, sobre
TODAS as citações do goldenset final (não só menção vaga) -- número,
súmula, artigo de lei. Sem LLM: cada transformação é determinística e
sabemos de antemão a resposta certa, porque nós fizemos a transformação.

Gera N variantes por citação combinando:
  - troca de abreviação de classe por um sinônimo já catalogado em
    hunter/patterns.py (inclusive os novos, vindos da auditoria de
    cobertura com LLM local)
  - reformatação do número (sem pontuação / com pontos / com espaços)
  - ruído OCR letra-a-letra (0<->O, 1<->l, 5<->S, m<->rn) em 1-2 posições
  - separador de UF (/UF, -UF, (UF))
  - quebra de linha inserida no meio do identificador

Roda o pipeline contra cada variante (substituída in-place no documento
original) e mede recall/precisão/acurácia de classe -- tudo com gabarito
conhecido (a transformação é nossa, então sabemos a resposta certa).

Uso: python experiment/stress_test_noise.py [n_variantes_por_citacao]
"""
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from hunter.db import Base
from hunter.pipeline import process_document
import hunter.patterns as pt

TXT_DIR = ROOT / "txt"
GOLDENSET = ROOT / "goldenset_offsets.csv"
DB_PATH = ROOT / "desafio1_bracis.db"

RNG_SEED = 123

# sinônimos de classe conhecidos, agrupados por família (pra trocar um
# pelo outro sem mudar o SENTIDO da citação) -- derivados de
# hunter/patterns.py, escritos por extenso (não regex) para poder inserir
# no texto.
FAMILIAS_SINONIMOS = {
    "agravo_regimental": ["AgRg", "Ag. Rg.", "AG.REG.", "Agravo Regimental"],
    "agravo_interno": ["AgInt", "Ag. Int.", "Agravo Interno"],
    "agravo_em_resp": ["AREsp", "A.REsp", "AgREsp", "Agravo em Recurso Especial"],
    "agravo_em_re": ["ARE", "AgREx", "Agravo em Recurso Extraordinário"],
    "resp": ["REsp", "R.Esp.", "Rec. Esp.", "Recurso Especial"],
    "re": ["RE", "REX", "Recurso Extraordinário"],
    "rhc": ["RHC", "RHAB", "Recurso em Habeas Corpus"],
    "rcl": ["Rcl", "Recl.", "RCL", "Reclamação"],
    "apl": ["APL", "Apelação"],
    "hc": ["HC", "H.C.", "Habeas Corpus"],
    "ar": ["AR", "AC.RES.", "Ação Rescisória"],
    "ms": ["MS", "Mandado de Segurança"],
}

_OCR_LETTER_SWAPS = [("e", "c"), ("m", "rn"), ("o", "0"), ("O", "0")]
_OCR_DIGIT_SWAPS = [("0", "O"), ("1", "l"), ("5", "S")]


def load_docs():
    return {f.stem: f.read_text(encoding="utf-8") for f in TXT_DIR.glob("*.txt")}


def load_goldenset():
    with open(GOLDENSET, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["inicio"], r["fim"] = int(r["inicio"]), int(r["fim"])
    return rows


def reformat_number(digits: str, rng: random.Random) -> str:
    style = rng.choice(["none", "dots3", "spaces3", "dots_noisy"])
    if style == "none":
        return digits
    if style == "dots3":
        return _group3(digits, ".")
    if style == "spaces3":
        return _group3(digits, " ")
    if style == "dots_noisy":
        # ponto E espaço, tipo "1.741. 784" observado no dev set
        grouped = _group3(digits, ".")
        pos = grouped.rfind(".")
        if pos != -1:
            grouped = grouped[:pos + 1] + " " + grouped[pos + 1:]
        return grouped
    return digits


def _group3(digits: str, sep: str) -> str:
    rev = digits[::-1]
    groups = [rev[i:i + 3] for i in range(0, len(rev), 3)]
    return sep.join(groups)[::-1]


def apply_ocr_noise(s: str, rng: random.Random, max_swaps: int = 1) -> str:
    out = list(s)
    swaps_done = 0
    idxs = list(range(len(out)))
    rng.shuffle(idxs)
    for i in idxs:
        if swaps_done >= max_swaps:
            break
        c = out[i]
        for a, b in _OCR_DIGIT_SWAPS + _OCR_LETTER_SWAPS:
            if c == a:
                out[i] = b
                swaps_done += 1
                break
    return "".join(out)


def insert_linebreak(s: str, rng: random.Random) -> str:
    spaces = [i for i, c in enumerate(s) if c == " "]
    if not spaces:
        return s
    pos = rng.choice(spaces)
    return s[:pos] + "\n" + s[pos + 1:]


def make_variant(trecho: str, rng: random.Random) -> str:
    """Aplica 1-3 transformações de ruído documentado sobre um trecho de
    citação, mantendo a identidade (nunca troca um dígito por outro
    valor de dígito -- só sua REPRESENTAÇÃO)."""
    out = trecho

    # 1) troca de abreviação de classe por sinônimo da mesma família
    for familia, sinonimos in FAMILIAS_SINONIMOS.items():
        for s in sinonimos:
            if re.search(r"(?<![A-Za-zÀ-ÿ])" + re.escape(s) + r"(?![A-Za-zÀ-ÿ])",
                         out, re.IGNORECASE):
                novo = rng.choice([x for x in sinonimos if x != s] or [s])
                out = re.sub(re.escape(s), novo, out, count=1, flags=re.IGNORECASE)
                break

    # 2) reformata o número (só os dígitos, preservando UF/sufixo) --
    # limiar de 5 dígitos pra nunca pegar o ANO de uma citação
    # relator+ano (sempre 4 dígitos): reagrupar ano em milhares
    # ("2 024") não é um ruído real, é só um jeito de quebrar o
    # regex \d{4} do parser por acidente do gerador de teste.
    m = re.search(r"\d[\d.\-/\s]{1,25}\d|\d", out)
    if m:
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= 5:
            novo_num = reformat_number(digits, rng)
            out = out[:m.start()] + novo_num + out[m.end():]

    # 3) separador de UF (baixo risco, sempre pode aplicar)
    uf_m = re.search(r"[/\-]\s*([A-Z]{2})\s*$", out)
    if uf_m and rng.random() < 0.5:
        uf = uf_m.group(1)
        novo_sep = rng.choice([f"/{uf}", f" - {uf}", f" ({uf})"])
        out = out[:uf_m.start()] + novo_sep

    # 4) ruído OCR letra/dígito OU quebra de linha -- no máximo UM dos
    # dois, pra não empilhar ruído demais num só trecho curto (o nível 2
    # real observado combina ruídos, mas não todos ao mesmo tempo)
    escolha = rng.choice(["ocr", "linebreak", "nenhum"])
    if escolha == "ocr":
        out = apply_ocr_noise(out, rng, max_swaps=1)
    elif escolha == "linebreak":
        out = insert_linebreak(out, rng)

    return out


def substitute_span(text, other_rows, target_row, replacement):
    ini, fim = target_row["inicio"], target_row["fim"]
    new_text = text[:ini] + replacement + text[fim:]
    delta = len(replacement) - (fim - ini)
    new_span = (ini, ini + len(replacement))
    shifted = []
    for r in other_rows:
        if r is target_row:
            continue
        if r["inicio"] >= fim:
            shifted.append({**r, "inicio": r["inicio"] + delta, "fim": r["fim"] + delta})
        else:
            shifted.append(r)
    return new_text, new_span, shifted


def main():
    n_variantes = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rng = random.Random(RNG_SEED)

    docs = load_docs()
    gold = load_goldenset()
    by_doc = defaultdict(list)
    for r in gold:
        by_doc[r["documento_id"]].append(r)

    # só citações com identificador (número/artigo/súmula) -- vago não
    # existe mais no gabarito, e relator+ano não tem "número" pra ruidar
    alvo_rows = [r for r in gold if re.search(r"\d", r["trecho"])]

    print(f"Base para stress-test: {len(alvo_rows)} citações com identificador, "
          f"{n_variantes} variantes cada = {len(alvo_rows) * n_variantes} instâncias sintéticas")

    base = Base(str(DB_PATH))

    total = matched = classe_ok = link_ok_n = link_total = 0
    falhas = []

    for row in alvo_rows:
        doc_id = row["documento_id"]
        text = docs[doc_id]
        other_rows = by_doc[doc_id]
        # usa o texto REAL do documento (row["trecho"] vem do CSV com
        # quebra de linha escapada como "\n" literal -- 2 caracteres --
        # não a quebra de linha de verdade que está no .txt)
        trecho_real = text[row["inicio"]:row["fim"]]

        for v in range(n_variantes):
            variante = make_variant(trecho_real, rng)
            new_text, new_span, _shifted = substitute_span(text, other_rows, row, variante)

            resultado = process_document(f"{doc_id}_v{v}", new_text, base)
            preds = resultado["citacoes"]

            total += 1
            found = None
            for p in preds:
                inter = max(0, min(p["fim"], new_span[1]) - max(p["inicio"], new_span[0]))
                uniao = (p["fim"] - p["inicio"]) + (new_span[1] - new_span[0]) - inter
                if uniao > 0 and inter / uniao >= 0.5:
                    found = p
                    break
            if found is None:
                falhas.append((doc_id, row["citacao_id"], "FN", repr(variante), row["classificacao"]))
                continue
            matched += 1
            if found["classificacao"] == row["classificacao"]:
                classe_ok += 1
            else:
                falhas.append((doc_id, row["citacao_id"], "CLASSE",
                               f"{row['classificacao']}->{found['classificacao']}",
                               repr(variante)))
            if row["classificacao"] == "real":
                link_total += 1
                resol = found.get("resolucao") or {}
                pred_id = str(resol.get("id_canonico", "")).strip()
                if pred_id == str(row.get("id_canonico", "")).strip():
                    link_ok_n += 1

    print(f"\nrecall (span)      = {matched}/{total} = {matched/total:.3f}")
    print(f"acurácia de classe = {classe_ok}/{matched} = {classe_ok/matched:.3f}" if matched else "n/a")
    print(f"acurácia de link   = {link_ok_n}/{link_total} = {link_ok_n/link_total:.3f}" if link_total else "n/a")

    print(f"\nFalhas ({len(falhas)}):")
    for f in falhas[:40]:
        print(" ", f)


if __name__ == "__main__":
    main()
