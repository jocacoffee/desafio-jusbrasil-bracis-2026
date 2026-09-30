# -*- coding: utf-8 -*-
"""
Carrega a base canônica (desafio1_bracis.db) e constrói, uma única vez,
os índices em memória usados pela resolução (resolve.py):

  - `numero_index`: raw_digits -> lista de registros de acórdão cujo
    "número próprio" (extraído da própria ementa/cabeçalho) contém esses
    dígitos. Construído aplicando a MESMA extração de candidatos usada
    nos documentos de entrada (extract.find_jurisprudencia_candidates)
    sobre o texto de cada um dos ~1.000 acórdãos e tomando a primeira
    ocorrência -- ver extract.py e a nota em normalize.py sobre por que
    comparar dígitos crus é mais robusto que reconstruir tokens do FTS5.

  - `sumula_records`: os 5 registros natureza='sumula', com o número
    oficial da súmula resolvido (parseado do próprio texto quando
    presente -- casos STJ; ou de um mapeamento verificado para os casos
    em que o texto não embute o número -- ver SUMULA_NUMERO_CONHECIDO).

  - `dispositivo_records`: os 13 registros natureza='dispositivo', ligados
    ao código de origem via patterns.ARTIGO_TO_CODIGO.

Tudo cabe folgadamente em memória (93 MB de banco, ~1.000 registros) --
nenhuma dependência além de sqlite3 da stdlib.

ENRIQUECIMENTO (opcional): calcular numero_index para ~1.000 acórdãos
varrendo regex sobre o texto inteiro de cada um custa ~25-30s por
execução (medido). `enrich_db.py` pré-computa isso uma vez e grava em
tabelas extras (`hunter_numero_proprio`, `hunter_sumula_numero`,
`hunter_dispositivo_codigo`) numa cópia do `.db`; se essas tabelas
existirem, `Base._load()` usa o caminho rápido (leitura direta, sem
regex). Sem elas, cai no cálculo em memória de sempre -- funciona igual,
só mais devagar. As funções `compute_*` abaixo são a fonte única de
verdade usada nos dois caminhos (execução direta e `enrich_db.py`), para
nunca haver divergência entre o índice pré-computado e o calculado na
hora.
"""
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
# do processo (ver compute_own_numero).
_AUTOS_MARK_RE = re.compile(
    r"(?:relatados?\s+e\s+discutidos|vistos?,?\s+relatados?)\s+estes\s+autos\s+de",
    re.IGNORECASE,
)

ENRICHMENT_TABLES = ("hunter_numero_proprio", "hunter_sumula_numero",
                     "hunter_dispositivo_codigo")


# ---------------------------------------------------------------------
# Funções puras de cômputo -- usadas tanto pelo carregamento normal
# (Base._load, quando a base não está enriquecida) quanto por
# enrich_db.py (que persiste o resultado uma vez, para reuso rápido).
# ---------------------------------------------------------------------

def compute_own_numero(texto: str) -> Optional[Tuple[str, str]]:
    """Acha o "número próprio" de um acórdão a partir do seu texto.
    Retorna (raw_digits, trecho) ou None se não achar nenhum número
    reconhecível. Ver nota em _index_acordao (histórico) sobre a fórmula
    "Vistos, relatados e discutidos estes autos de..." como sinal
    preferencial em acórdãos longos que citam outros precedentes antes
    dela (TST, sobretudo)."""
    cands = find_jurisprudencia_candidates(texto, full=False)
    numeros = [c for c in cands if c.kind == "numero" and c.raw_digits]
    if not numeros:
        return None
    m = _AUTOS_MARK_RE.search(texto)
    if m:
        for c in numeros:
            if c.inicio >= m.end() and c.inicio - m.end() < 80:
                return c.raw_digits, c.trecho
    c = numeros[0]
    return c.raw_digits, c.trecho


def compute_sumula_numero(texto: str, tribunal: Optional[str]) -> Optional[int]:
    m = _SUMULA_NUM_IN_TEXT_RE.search(texto)
    return int(m.group(1)) if m else SUMULA_NUMERO_CONHECIDO.get(tribunal)


def compute_dispositivo_codigo(texto: str) -> Tuple[Optional[int], Optional[str]]:
    m = pt.ARTIGO_NUM_RE.search(texto)
    artigo = int(m.group(1)) if m else None
    codigo = pt.ARTIGO_TO_CODIGO.get(artigo)
    return artigo, codigo


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

        enriquecido = self._tem_enriquecimento(conn)
        enriquecimento = self._carrega_enriquecimento(conn) if enriquecido else None
        conn.close()

        for reg in self.registros:
            if reg.natureza == "acordao":
                self._index_acordao(reg, enriquecimento)
            elif reg.natureza == "sumula":
                self._index_sumula(reg, enriquecimento)
            elif reg.natureza == "dispositivo":
                self._index_dispositivo(reg, enriquecimento)

    @staticmethod
    def _tem_enriquecimento(conn: sqlite3.Connection) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN "
            "(?, ?, ?)", ENRICHMENT_TABLES,
        )
        return len(cur.fetchall()) == len(ENRICHMENT_TABLES)

    @staticmethod
    def _carrega_enriquecimento(conn: sqlite3.Connection) -> dict:
        """Lê as 3 tabelas de enriquecimento de uma vez (caminho rápido:
        sem regex, só SELECT). Ver enrich_db.py para como são geradas."""
        numero_proprio = {}
        for _id, raw_digits_, trecho in conn.execute(
                "SELECT id, raw_digits, own_trecho FROM hunter_numero_proprio"):
            numero_proprio[_id] = (raw_digits_, trecho)
        sumula_numero = {}
        for _id, numero in conn.execute("SELECT id, numero FROM hunter_sumula_numero"):
            sumula_numero[_id] = numero
        dispositivo_codigo = {}
        for _id, artigo, codigo in conn.execute(
                "SELECT id, artigo, codigo FROM hunter_dispositivo_codigo"):
            dispositivo_codigo[_id] = (artigo, codigo)
        return {
            "numero_proprio": numero_proprio,
            "sumula_numero": sumula_numero,
            "dispositivo_codigo": dispositivo_codigo,
        }

    def _index_acordao(self, reg: Registro, enriquecimento: Optional[dict]):
        if enriquecimento is not None:
            par = enriquecimento["numero_proprio"].get(reg.id)
        else:
            par = compute_own_numero(reg.texto)
        if par is None:
            return
        digits, trecho = par
        self.numero_index[digits].append(reg)
        self.own_digits_by_id[reg.id] = digits
        self.own_trecho_by_id[reg.id] = trecho

    def _index_sumula(self, reg: Registro, enriquecimento: Optional[dict]):
        if enriquecimento is not None:
            numero = enriquecimento["sumula_numero"].get(reg.id)
        else:
            numero = compute_sumula_numero(reg.texto, reg.tribunal)
        self.sumula_records.append({
            "tribunal": reg.tribunal,
            "numero": numero,
            "registro": reg,
        })

    def _index_dispositivo(self, reg: Registro, enriquecimento: Optional[dict]):
        if enriquecimento is not None:
            artigo, codigo = enriquecimento["dispositivo_codigo"].get(reg.id, (None, None))
        else:
            artigo, codigo = compute_dispositivo_codigo(reg.texto)
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
