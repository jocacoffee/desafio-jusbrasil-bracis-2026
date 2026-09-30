# -*- coding: utf-8 -*-
"""
Pré-processamento/enriquecimento OPCIONAL da base canônica (ver e-mail da
organização, item 4: "é permitido enriquecer a base... enviem o código
que gera esse enriquecimento a partir de um .db no formato original").

Sem enriquecimento, `hunter/db.py` calcula em memória, a cada execução,
o "número próprio" de cada acórdão (~25-30s, medido, num banco de ~1.000
registros) varrendo o texto inteiro de cada um com regex. Esse cálculo
não muda entre execuções para o MESMO `.db` -- então vale a pena
persistir uma vez.

Este script lê um `.db` no formato original (intocado) e grava uma CÓPIA
com 3 tabelas extras (nunca sobrescreve o original):
  - hunter_numero_proprio(id, raw_digits, own_trecho)   -- acórdãos
  - hunter_sumula_numero(id, numero)                     -- súmulas
  - hunter_dispositivo_codigo(id, artigo, codigo)        -- dispositivos

`hunter/db.py` detecta essas tabelas automaticamente e usa o caminho
rápido quando presentes; sem elas, o pipeline funciona exatamente igual
(mesmo resultado), só mais devagar -- o enriquecimento é puramente uma
otimização de tempo, nunca uma dependência de correção.

Uso:
    python enrich_db.py <db_original> <db_enriquecido>
"""
import shutil
import sqlite3
import sys
from pathlib import Path

from hunter.db import compute_own_numero, compute_sumula_numero, compute_dispositivo_codigo


def enrich(db_original: str, db_saida: str) -> None:
    origem = Path(db_original)
    destino = Path(db_saida)
    if not origem.is_file():
        sys.exit(f"ERRO: '{db_original}' não encontrado.")
    if destino.resolve() == origem.resolve():
        sys.exit("ERRO: o destino não pode ser o mesmo arquivo do original "
                  "(o enriquecimento nunca sobrescreve o .db de entrada).")

    shutil.copy2(origem, destino)
    conn = sqlite3.connect(str(destino))
    try:
        conn.execute("DROP TABLE IF EXISTS hunter_numero_proprio")
        conn.execute("DROP TABLE IF EXISTS hunter_sumula_numero")
        conn.execute("DROP TABLE IF EXISTS hunter_dispositivo_codigo")
        conn.execute("CREATE TABLE hunter_numero_proprio "
                     "(id INTEGER PRIMARY KEY, raw_digits TEXT NOT NULL, "
                     "own_trecho TEXT NOT NULL)")
        conn.execute("CREATE TABLE hunter_sumula_numero "
                     "(id INTEGER PRIMARY KEY, numero INTEGER)")
        conn.execute("CREATE TABLE hunter_dispositivo_codigo "
                     "(id INTEGER PRIMARY KEY, artigo INTEGER, codigo TEXT)")

        n_acordao = n_sumula = n_dispositivo = 0
        cur = conn.execute("SELECT id, tribunal, natureza, texto FROM documentos")
        linhas = cur.fetchall()
        for _id, tribunal, natureza, texto in linhas:
            if natureza == "acordao":
                par = compute_own_numero(texto)
                if par is not None:
                    digits, trecho = par
                    conn.execute(
                        "INSERT INTO hunter_numero_proprio (id, raw_digits, own_trecho) "
                        "VALUES (?, ?, ?)", (_id, digits, trecho))
                    n_acordao += 1
            elif natureza == "sumula":
                numero = compute_sumula_numero(texto, tribunal)
                conn.execute(
                    "INSERT INTO hunter_sumula_numero (id, numero) VALUES (?, ?)",
                    (_id, numero))
                n_sumula += 1
            elif natureza == "dispositivo":
                artigo, codigo = compute_dispositivo_codigo(texto)
                conn.execute(
                    "INSERT INTO hunter_dispositivo_codigo (id, artigo, codigo) "
                    "VALUES (?, ?, ?)", (_id, artigo, codigo))
                n_dispositivo += 1
        conn.commit()
    finally:
        conn.close()

    print(f"Enriquecido: {n_acordao} acórdãos, {n_sumula} súmulas, "
          f"{n_dispositivo} dispositivos -> {destino}")


def main():
    if len(sys.argv) != 3:
        sys.exit("uso: python enrich_db.py <db_original> <db_enriquecido>")
    enrich(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
