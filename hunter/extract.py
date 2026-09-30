# -*- coding: utf-8 -*-
"""
M1 -- extrator de spans (citações).

Encontra, num texto (petição de entrada OU acórdão da base canônica),
todas as ocorrências de:
  - citação de jurisprudência bem formada (classe processual + número)
  - súmula
  - dispositivo de lei
  - menção "incompleta" buscável mas ambígua (relator+ano+tribunal, sem
    número). Menções sem NENHUMA fonte específica ("jurisprudência
    pacífica desta Corte" etc.) não são mais extraídas: saíram do
    gabarito por comunicado da organização (30/09/2026) -- não apontam
    para nenhum registro e contá-las hoje seria falso positivo.

Cada candidato carrega span (inicio, fim em codepoints), o trecho literal,
o "tipo" do contrato (lei|jurisprudencia) e, quando aplicável, os campos
crus necessários para a resolução (M3/M4): raw_digits do número, tribunal
inferido, ano/relator (para o caso relator+ano), número do artigo etc.

A mesma função `find_jurisprudencia_candidates` é usada tanto para extrair
citações dos documentos de entrada quanto, em db.py, para localizar o
"número próprio" de cada acórdão da base (primeira ocorrência no texto).
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

from . import patterns as pt
from .normalize import raw_digits, digits_with_ocr_fix, looks_like_identifier_noise

_OCR_DIGIT_LETTERS = "OoIlSsgG"  # ver normalize.py: "g"->9 mapeado, "G" só estende o span
# letra confundível com dígito, com o mesmo separador tolerante a ruído
# usado entre dígitos normais (patterns._SEP), seguida de outro dígito --
# ou seja, "encaixada" dentro do número, não o começo de uma UF/sigla.
_OCR_MID_RE = re.compile(
    r"[\.\-/\s°ºo]{0,3}[" + _OCR_DIGIT_LETTERS + r"](?=[\.\-/\s°ºo]{0,3}\d)"
)
# continuação normal de dígitos (com separador tolerante) depois de uma
# letra confundível já consumida -- ex.: o ".779" depois do "g" em
# "1.45g.779"
_DIGIT_RUN_RE = re.compile(r"(?:[\.\-/\s°ºo]{0,3}\d)*")


def _extend_through_ocr_digit(text: str, end: int) -> int:
    """Estende o fim de um número já casado quando uma letra confundível
    com dígito por OCR (O/l/S/g -- PDF §2 + achados do dev set) aparece
    encaixada entre dois dígitos (ex.: "21737l8", "1.45g.779"). Só estende
    se depois da letra (com ou sem separador) vier outro dígito -- isso
    evita "comer" o começo de um sufixo de UF (ex. "- SP"), que nunca é
    seguido de outro dígito."""
    i = end
    while True:
        m = _OCR_MID_RE.match(text, i)
        if not m:
            break
        i = m.end()
        i = _DIGIT_RUN_RE.match(text, i).end()
    # último dígito do número também pode virar letra confundível (ex.:
    # "170076O (SP)") -- aceita se logo depois vem um sufixo de UF, já
    # que nesse caso a letra não pode ser o começo da UF (que já tem
    # suas 2 letras próprias logo em seguida).
    if i < len(text) and text[i] in _OCR_DIGIT_LETTERS and pt.UF_SUFFIX_RE.match(text[i + 1:]):
        i += 1
    return i


@dataclass
class Candidate:
    inicio: int
    fim: int
    trecho: str
    tipo: str  # "jurisprudencia" | "lei"
    kind: str  # "numero" | "sumula" | "artigo" | "relator_ano" | "tema"
    raw_digits: str = ""
    tribunal: Optional[str] = None
    ano: Optional[int] = None
    relator: Optional[str] = None
    artigo_num: Optional[int] = None
    codigo_clause: str = ""
    numero: Optional[int] = None


# palavras que, se aparecerem imediatamente antes do número, indicam que
# é o número de autos do PRÓPRIO documento (distrator), não uma citação.
_OWN_HEADER_MARK_RE = re.compile(
    r"\b(Processo|Autos|Protocolo)\s+n[ºo°.]?\s*$", re.IGNORECASE
)


def _is_own_header_number(text: str, start: int) -> bool:
    window = text[max(0, start - 25):start]
    return bool(_OWN_HEADER_MARK_RE.search(window))


def find_jurisprudencia_candidates(text: str, full: bool = True) -> List[Candidate]:
    """`full=False` pula as etapas caras (casamento difuso de menções
    vagas) e as etapas irrelevantes (tema/relator+ano) usadas só na
    extração de citações de petições -- usado por db.py para indexar
    somente o número próprio dos 1.000 acórdãos, que são textos grandes
    (até ~90 mil caracteres) onde a etapa difusa seria custosa demais."""
    out: List[Candidate] = []

    # 1) súmulas primeiro (regex mais específica, evita que o número da
    #    súmula seja depois recapturado como citação sequencial solta)
    claimed = []
    for m in pt.SUMULA_RE.finditer(text):
        vinculante = bool(m.group(1))
        numero = int(digits_with_ocr_fix(m.group(2)))
        tribunal = m.group(3) or ("STF" if vinculante else None)
        out.append(Candidate(m.start(), m.end(), m.group(), "jurisprudencia", "sumula",
                              tribunal=tribunal, numero=numero))
        claimed.append((m.start(), m.end()))

    def _overlaps_claimed(s, e):
        return any(s < ce and e > cs for cs, ce in claimed)

    # 2) citações com classe processual + número (CNJ ou sequencial)
    for num_re in (pt.CNJ_NUM_RE, pt.SEQ_NUM_RE):
        for nm in num_re.finditer(text):
            ns, ne = nm.start(), nm.end()
            if _overlaps_claimed(ns, ne):
                continue
            if _is_own_header_number(text, ns):
                continue
            ne = _extend_through_ocr_digit(text, ne)
            digits = digits_with_ocr_fix(text[ns:ne])
            if len(digits) < 3:
                continue

            # tenta casar um prefixo de classe imediatamente antes do número
            pre_window_start = max(0, ns - 80)
            pre_window = text[pre_window_start:ns]
            # remove iterativamente marcador "nº"/hífen solto colado no
            # número (ex.: "RR-1835...", "ARR-213...", "RSE-1234..."). O
            # (?<![A-Za-zÀ-ÿ]) garante que o "n"/"N" é um marcador solto,
            # não o fim de uma palavra comum -- sem essa guarda, "Agravo
            # Interno" perdia o "no" final (interpretado como "n"+"o" do
            # marcador) e virava "Agravo Inter", quebrando o átomo de classe.
            while True:
                new_pw = re.sub(r"(?<![A-Za-zÀ-ÿ])\s*[Nn][ºo°.]?\s*$", "", pre_window)
                new_pw = re.sub(r"\s*-\s*$", "", new_pw)
                if new_pw == pre_window:
                    break
                pre_window = new_pw
            # ponto de abreviação solto colado à classe ("RE.", "Recl.")
            # que o átomo em si não previa
            pre_window = re.sub(r"\.\s*$", "", pre_window)
            pre_window = pre_window.rstrip()

            best = None
            for cm in pt.CLASSE_PREFIX_RE.finditer(pre_window):
                if cm.end() == len(pre_window) and cm.group().strip():
                    if best is None or cm.start() < best.start():
                        best = cm
            if best is None:
                # sem prefixo de classe reconhecido: não é citação --
                # (evita capturar números soltos de qualquer natureza)
                continue
            span_start = pre_window_start + best.start()
            # "TST-" à esquerda da cadeia de classes já reconhecida
            # (ex.: "TST-E-RR-...", "TST- ED - E-ED-RR-...")
            tst_m = pt.TST_LEAD_RE.search(pre_window[:best.start()])
            if tst_m:
                span_start = pre_window_start + tst_m.start()

            span_end = ne
            uf_m = pt.UF_SUFFIX_RE.match(text[ne:ne + 12])
            if uf_m:
                span_end = ne + uf_m.end()

            trecho = text[span_start:span_end]
            if not looks_like_identifier_noise(nm.group()):
                continue
            if _overlaps_claimed(span_start, span_end):
                continue
            claimed.append((span_start, span_end))
            out.append(Candidate(span_start, span_end, trecho, "jurisprudencia",
                                  "numero", raw_digits=digits))

    if not full:
        out.sort(key=lambda c: c.inicio)
        return out

    # 3) "Tema/Tese N da repercussão geral" -- sempre inventada
    for m in pt.TEMA_RE.finditer(text):
        if _overlaps_claimed(m.start(), m.end()):
            continue
        claimed.append((m.start(), m.end()))
        out.append(Candidate(m.start(), m.end(), m.group(), "jurisprudencia", "tema"))

    # 4) relator+ano+tribunal (buscável, ambíguo por natureza)
    for m in pt.RELATOR_ANO_RE.finditer(text):
        if _overlaps_claimed(m.start(), m.end()):
            continue
        claimed.append((m.start(), m.end()))
        trib = m.group(1) or m.group(3)
        ano = int(m.group(2))
        relator = m.group(4).strip().rstrip(".")
        out.append(Candidate(m.start(), m.end(), m.group(), "jurisprudencia",
                              "relator_ano", tribunal=trib, ano=ano, relator=relator))

    # 5) [REMOVIDO] menções sem nenhuma fonte específica (ex.: "jurisprudência
    #    pacífica desta Corte") saíram do gabarito por comunicado da
    #    organização (30/09/2026): não apontam para nenhum registro e não
    #    contam mais como citação -- extraí-las agora seria falso positivo.
    #    "relator_ano" (passo 4) continua valendo: aponta uma fonte concreta
    #    (tribunal+relator+ano), só falta identificador para resolvê-la.

    out.sort(key=lambda c: c.inicio)
    return out


def find_lei_candidates(text: str) -> List[Candidate]:
    out: List[Candidate] = []
    claimed = []

    for m in pt.ARTIGO_RE.finditer(text):
        if any(m.start() < ce and m.end() > cs for cs, ce in claimed):
            continue
        claimed.append((m.start(), m.end()))
        artigo_num = int(m.group(1).replace(".", ""))
        out.append(Candidate(m.start(), m.end(), m.group(), "lei", "artigo",
                              artigo_num=artigo_num, codigo_clause=m.group(2)))

    # [REMOVIDO] "normas de regência da matéria" e afins -- mesma mudança
    # de gabarito do passo 5 de find_jurisprudencia_candidates, ver nota lá.

    out.sort(key=lambda c: c.inicio)
    return out


def find_all_candidates(text: str) -> List[Candidate]:
    cands = find_jurisprudencia_candidates(text) + find_lei_candidates(text)
    cands.sort(key=lambda c: c.inicio)
    return cands
