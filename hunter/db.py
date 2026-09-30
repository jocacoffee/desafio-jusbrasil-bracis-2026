# -*- coding: utf-8 -*-
"""
Carrega a base canônica (desafio1_bracis.db) e constrói, uma única vez,
os índices em memória usados pela resolução (resolve.py):

  - `numero_index`: raw_digits -> lista de registros de acórdão cujo
    "número próprio" (extraído da própria ementa/cabeçalho) contém esses
    dígitos. Construído aplicando a MESMA extração de candidatos usada
    nos documentos de entrada (extract.find_jurisprudencia_candidates)
    sobre o texto de cada um dos 1.000 acórdãos e tomando a primeira
    ocorrência -- ver extract.py e a nota em normalize.py sobre por que
    comparar dígitos crus é mais robusto que reconstruir tokens do FTS5.

  - `sumula_records`: os 5 registros natureza='sumula', com o número
    oficial da súmula resolvido (parseado do próprio texto quando
    presente -- casos STJ; ou de um mapeamento verificado para os casos
    em que o texto não embute o número -- ver SUMULA_NUMERO_CONHECIDO).

  - `dispositivo_records`: os 13 registros natureza='dispositivo', ligados
    ao código de origem via patterns.ARTIGO_TO_CODIGO.

Tudo cabe folgadamente em memória (93 MB de banco, 1.018 registros) --
nenhuma dependência além de sqlite3 da stdlib.
"""
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from . import patterns as pt
from .extract import find_jurisprudencia_candidates
from .normalize import raw_digits


@dataclass
class Registro:
    documento_id: str
    id: int
    tribunal: Optional[str]
    ano: Optional[int]
    relator: Optional[str]
    natureza: str
    tipo: str
    texto: str


# Números oficiais de súmula que NÃO aparecem no corpo do texto do
# registro (checado manualmente contra os 5 registros natureza='sumula'
# da base: STJ embute "(SÚMULA N, ...)" no próprio texto e é parseado
# via regex; STF e TST não embutem e o conteúdo foi conferido contra o
# texto oficial e público das súmulas correspondentes).
SUMULA_NUMERO_CONHECIDO = {
    "STF": 10,   # Súmula Vinculante 10 -- reserva de plenário (CF, art. 97)
    "TST": 331,  # Súmula 331 -- terceirização/responsabilidade subsidiária
}

_SUMULA_NUM_IN_TEXT_RE = re.compile(r"S[UÚ]MULA\s+(\d+)", re.IGNORECASE)

# fórmula-padrão de abertura de acórdão que identifica o próprio número
# do processo (ver _index_acordao).
_AUTOS_MARK_RE = re.compile(
    r"(?:relatados?\s+e\s+discutidos|vistos?,?\s+relatados?)\s+estes\s+autos\s+de",
    re.IGNORECASE,
)


class Base:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.registros: List[Registro] = []
        self.numero_index: Dict[str, List[Registro]] = defaultdict(list)
        self.own_digits_by_id: Dict[int, str] = {}
        self.own_trecho_by_id: Dict[int, str] = {}
        self.sumula_records: List[dict] = []  # {"tribunal":..,"numero":..,"registro":..}
        self.dispositivo_records: List[dict] = []  # {"codigo":..,"artigo":..,"registro":..}
        self._load()

    def _load(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT documento_id, id, tribunal, ano, relator, natureza, tipo, texto "
            "FROM documentos"
        )
        for row in cur:
            reg = Registro(
                documento_id=row["documento_id"],
                id=row["id"],
                tribunal=row["tribunal"],
                ano=row["ano"],
                relator=row["relator"],
                natureza=row["natureza"],
                tipo=row["tipo"],
                texto=row["texto"],
            )
            self.registros.append(reg)
        conn.close()

        for reg in self.registros:
            if reg.natureza == "acordao":
                self._index_acordao(reg)
            elif reg.natureza == "sumula":
                self._index_sumula(reg)
            elif reg.natureza == "dispositivo":
                self._index_dispositivo(reg)

    def _index_acordao(self, reg: Registro):
        cands = find_jurisprudencia_candidates(reg.texto, full=False)
        numeros = [c for c in cands if c.kind == "numero" and c.raw_digits]
        if not numeros:
            return
        # sinal preferencial: a fórmula-padrão de abertura do acórdão
        # ("Vistos, relatados e discutidos estes autos de ... nº ...")
        # -- em acórdãos longos (TST, sobretudo) o próprio texto costuma
        # citar OUTROS precedentes antes dessa fórmula, então "primeira
        # ocorrência no documento" nem sempre é o número próprio; quando
        # a fórmula aparece, ela identifica o número com mais confiança
        # do que a simples posição.
        m = _AUTOS_MARK_RE.search(reg.texto)
        if m:
            for c in numeros:
                if c.inicio >= m.end() and c.inicio - m.end() < 80:
                    self.numero_index[c.raw_digits].append(reg)
                    self.own_digits_by_id[reg.id] = c.raw_digits
                    self.own_trecho_by_id[reg.id] = c.trecho
                    return
        c = numeros[0]
        self.numero_index[c.raw_digits].append(reg)
        self.own_digits_by_id[reg.id] = c.raw_digits
        self.own_trecho_by_id[reg.id] = c.trecho

    def _index_sumula(self, reg: Registro):
        m = _SUMULA_NUM_IN_TEXT_RE.search(reg.texto)
        numero = int(m.group(1)) if m else SUMULA_NUMERO_CONHECIDO.get(reg.tribunal)
        self.sumula_records.append({
            "tribunal": reg.tribunal,
            "numero": numero,
            "registro": reg,
        })

    def _index_dispositivo(self, reg: Registro):
        m = pt.ARTIGO_NUM_RE.search(reg.texto)
        artigo = int(m.group(1)) if m else None
        codigo = pt.ARTIGO_TO_CODIGO.get(artigo)
        self.dispositivo_records.append({
            "artigo": artigo,
            "codigo": codigo,
            "registro": reg,
        })

    def feitos_por_numero(self, digits: str) -> List[Registro]:
        """Retorna os registros cujo número próprio bate com `digits`."""
        return self.numero_index.get(digits, [])

    def feitos_por_relator_ano(self, tribunal: Optional[str], ano: int,
                                relator_nome: str) -> List[Registro]:
        relator_norm = relator_nome.strip().lower()
        out = []
        for reg in self.registros:
            if reg.natureza != "acordao":
                continue
            if reg.ano != ano:
                continue
            if tribunal and reg.tribunal != tribunal:
                continue
            if not reg.relator:
                continue
            reg_relator = reg.relator.strip().lower()
            # nome do relator no texto é frequentemente parcial (sobrenome)
            if relator_norm in reg_relator or reg_relator in relator_norm:
                out.append(reg)
                continue
            # também casa por sobrenome em comum (>=1 token >=4 chars)
            tokens_a = {t for t in re.split(r"\s+", relator_norm) if len(t) >= 4}
            tokens_b = {t for t in re.split(r"\s+", reg_relator) if len(t) >= 4}
            if tokens_a & tokens_b:
                out.append(reg)
        return out
