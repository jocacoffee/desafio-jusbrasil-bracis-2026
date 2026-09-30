# -*- coding: utf-8 -*-
"""
Entrypoint do pipeline do Caça-Alucinações (Desafio 1, BRACIS 2026).

Uso:
    python run.py <pasta_txt> <db_path> <pasta_saida_json>

Sem dependências além da stdlib (sqlite3, re, json) -- roda em
milissegundos por documento e não usa nenhum modelo de ML/LLM.
"""
import json
import sys
from pathlib import Path

from hunter.db import Base
from hunter.pipeline import process_document


def main():
    if len(sys.argv) < 4:
        sys.exit("uso: python run.py <pasta_txt> <db_path> <pasta_saida_json>")
    pasta_txt = Path(sys.argv[1])
    db_path = sys.argv[2]
    pasta_saida = Path(sys.argv[3])
    pasta_saida.mkdir(parents=True, exist_ok=True)

    base = Base(db_path)

    arquivos = sorted(pasta_txt.glob("*.txt"))
    if not arquivos:
        sys.exit(f"nenhum .txt encontrado em {pasta_txt}")

    for arq in arquivos:
        documento_id = arq.stem
        texto = arq.read_text(encoding="utf-8")
        doc = process_document(documento_id, texto, base)
        destino = pasta_saida / f"{documento_id}.json"
        destino.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"{len(arquivos)} documentos processados -> {pasta_saida}")


if __name__ == "__main__":
    main()
