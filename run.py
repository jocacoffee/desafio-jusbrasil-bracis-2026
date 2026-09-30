# -*- coding: utf-8 -*-
"""
Entrypoint do pipeline do Caça-Alucinações (Desafio 1, BRACIS 2026).

Uso:
    python run.py <pasta_txt> <db_path> <pasta_saida_json>

Sem dependências além da stdlib (sqlite3, re, json) -- roda em
milissegundos por documento e não usa nenhum modelo de ML/LLM.
"""
import json
import sqlite3
import sys
from pathlib import Path

from hunter.db import Base
from hunter.pipeline import process_document


def fail(msg: str) -> None:
    """Mensagem de erro clara em stderr + código de saída != 0 -- para
    a organização identificar rápido o que aconteceu, sem precisar ler
    um traceback do Python."""
    print(f"ERRO: {msg}", file=sys.stderr)
    sys.exit(1)


def read_text_tolerant(path: Path) -> str:
    """Lê um .txt como UTF-8; se falhar (arquivo com encoding diferente,
    ex.: latin-1/cp1252), tenta de novo substituindo os bytes inválidos
    em vez de derrubar a execução inteira por causa de 1 documento."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        print(f"AVISO: {path.name} não é UTF-8 válido ({e}); "
              f"lendo com substituição de caracteres inválidos.", file=sys.stderr)
        return path.read_text(encoding="utf-8", errors="replace")


def main():
    if len(sys.argv) < 4:
        fail("uso: python run.py <pasta_txt> <db_path> <pasta_saida_json>")

    pasta_txt = Path(sys.argv[1])
    db_path = Path(sys.argv[2])
    pasta_saida = Path(sys.argv[3])

    if not pasta_txt.is_dir():
        fail(f"pasta de documentos não encontrada ou não é um diretório: {pasta_txt}")
    if not db_path.is_file():
        fail(f"base canônica (.db) não encontrada: {db_path}")

    try:
        base = Base(str(db_path))
    except sqlite3.DatabaseError as e:
        fail(f"não foi possível abrir '{db_path}' como banco SQLite válido: {e}")
    except sqlite3.OperationalError as e:
        fail(f"'{db_path}' não tem o esquema esperado (tabela 'documentos'): {e}")

    arquivos = sorted(pasta_txt.glob("*.txt"))
    if not arquivos:
        fail(f"nenhum arquivo .txt encontrado em {pasta_txt}")

    try:
        pasta_saida.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        fail(f"não foi possível criar a pasta de saída '{pasta_saida}': {e}")

    erros = []
    processados = 0
    for arq in arquivos:
        documento_id = arq.stem
        try:
            texto = read_text_tolerant(arq)
            doc = process_document(documento_id, texto, base)
        except Exception as e:  # noqa: BLE001 -- um documento ruim não pode
            # derrubar a execução inteira; registra e segue para os demais,
            # com saída vazia (sem citações) para esse documento em vez de
            # faltar a linha dele por completo na submissão.
            print(f"AVISO: falha processando '{arq.name}' ({type(e).__name__}: {e}); "
                  f"gravando saída vazia para esse documento.", file=sys.stderr)
            erros.append(arq.name)
            doc = {"documento_id": documento_id, "citacoes": []}

        destino = pasta_saida / f"{documento_id}.json"
        destino.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        processados += 1

    print(f"{processados} documentos processados -> {pasta_saida}")
    if erros:
        print(f"AVISO: {len(erros)} documento(s) com erro (saída vazia): "
              f"{', '.join(erros)}", file=sys.stderr)


if __name__ == "__main__":
    main()
