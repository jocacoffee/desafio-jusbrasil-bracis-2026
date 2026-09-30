# -*- coding: utf-8 -*-
"""
Teste de generalização a classes processuais FORA do vocabulário fechado
(_CLASSE_ATOMS em hunter/patterns.py).

Motivação: a avaliação final roda contra um `.db` e um conjunto de
documentos NOVOS (comunicado da organização), então a extração precisa
generalizar além do que foi visto nos 26 documentos de dev e nos 996
acórdãos do `.db` atual. Uma auditoria contra a base atual (ver commit
"Amplia cobertura de classe processual via auditoria completa da base")
já achou e corrigiu 2 classes presentes no acervo mas nunca citadas no
dev set -- mas isso só cobre o que JÁ existe no `.db` atual, não uma
classe genuinamente nova no `.db` da avaliação.

Este teste simula esse cenário com classes que sabidamente NÃO estão em
_CLASSE_ATOMS (ADI, ADPF, ADC, Mandado de Injunção, Habeas Data, Conflito
de Competência, Revisão Criminal, Recurso de Reexame Necessário, Agravo
de Execução -- lista curada manualmente; uma tentativa inicial de pedir
essa lista para uma LLM local (qwen3:8b) devolveu majoritariamente
duplicatas triviais do vocabulário já existente, mesmo com instrução
explícita para não repetir -- então a lista final aqui vem de
conhecimento de domínio verificado, não do LLM) e mede o efeito do
fallback de número em formato CNJ padrão adicionado em extract.py:

  - ANTES do fallback (`git stash` neste arquivo alvo): 0/9 em todas as
    colunas -- nenhuma classe fora do vocabulário era reconhecida, nem
    para extrair a citação nem para indexar o "número próprio" do
    acórdão na base.
  - DEPOIS: 9/9 quando o número da citação está no formato CNJ padrão
    (13-20 dígitos, bem mais distintivo que um número curto solto) --
    tanto na extração quanto na indexação. Continua 0/9 para números no
    formato sequencial antigo (poucos dígitos) associados a uma classe
    nunca vista: não dá pra aceitar um número curto sem NENHUMA âncora
    (nome de classe OU formato distintivo) sem disparar falsos positivos
    em qualquer número solto do texto (datas, protocolos, etc.).

Esse fallback foi testado contra falso positivo com CNPJ/CPF/OAB (também
têm 13+ dígitos com separador) antes de aceitar a mudança -- ver a
exclusão _NAO_CITACAO_MARK_RE em extract.py.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hunter.db import compute_own_numero
from hunter.extract import find_jurisprudencia_candidates

# classes deliberadamente ausentes de _CLASSE_ATOMS -- ver nota acima
# sobre a curadoria manual (não a sugestão crua do LLM local).
CASOS = [
    ("Ação Direta de Inconstitucionalidade", "ADI"),
    ("Arguição de Descumprimento de Preceito Fundamental", "ADPF"),
    ("Ação Declaratória de Constitucionalidade", "ADC"),
    ("Mandado de Injunção", "MI"),
    ("Habeas Data", "HD"),
    ("Conflito de Competência", "CC"),
    ("Revisão Criminal", "RvCr"),
    ("Recurso de Reexame Necessário", "RReN"),
    ("Agravo de Execução", "AgExec"),
]

NUM_CNJ = "1234567-89.2021.1.00.0001"  # formato CNJ padrão (20 dígitos)
NUM_SEQ = "45.678"                      # formato sequencial antigo (poucos dígitos)


def _digits(s: str) -> str:
    return "".join(c for c in s if c.isdigit())


def _testa_extracao(abrev: str, numero: str) -> bool:
    texto = (f"Conforme entendimento consolidado no julgamento do {abrev} "
             f"nº {numero}/DF, cuja ementa reproduzimos, esta Corte fixou...")
    return any(c.kind == "numero" and c.raw_digits == _digits(numero)
               for c in find_jurisprudencia_candidates(texto))


def _testa_indexacao(nome_completo: str, numero: str) -> bool:
    texto = (f"ACÓRDÃO\n\nVistos, relatados e discutidos estes autos de "
             f"{nome_completo} nº {numero}, em que é requerente Fulano de "
             f"Tal e requerido o Estado, acordam os Ministros...")
    par = compute_own_numero(texto)
    return par is not None and par[0] == _digits(numero)


def main():
    print(f"{'classe':50s} {'extração(CNJ)':14s} {'extração(SEQ)':14s} {'indexação':10s}")
    totais = {"cnj": 0, "seq": 0, "idx": 0}
    for nome, abrev in CASOS:
        r_cnj = _testa_extracao(abrev, NUM_CNJ)
        r_seq = _testa_extracao(abrev, NUM_SEQ)
        r_idx = _testa_indexacao(nome, NUM_CNJ)
        totais["cnj"] += r_cnj
        totais["seq"] += r_seq
        totais["idx"] += r_idx
        print(f"{nome:50s} {'OK' if r_cnj else 'FALHOU':14s} "
              f"{'OK' if r_seq else 'FALHOU':14s} {'OK' if r_idx else 'FALHOU':10s}")
    n = len(CASOS)
    print(f"\ntotais: extração CNJ {totais['cnj']}/{n}, "
          f"extração SEQ {totais['seq']}/{n}, indexação {totais['idx']}/{n}")


if __name__ == "__main__":
    main()
