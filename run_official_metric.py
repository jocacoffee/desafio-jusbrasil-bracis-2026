# -*- coding: utf-8 -*-
"""
Roda a métrica OFICIAL (kaggle_metric.py, obtido do Kaggle em 30/09/2026)
contra a submissão gerada pelo pipeline, usando o goldenset_offsets.csv
final como gabarito.

Uso: python run_official_metric.py [submission.csv] [goldenset_offsets.csv]
"""
import csv
import sys
from collections import defaultdict

import pandas as pd

from kaggle_metric import avaliar


def build_solution_df(goldenset_path: str) -> pd.DataFrame:
    by_doc = defaultdict(list)
    nivel_by_doc = {}
    with open(goldenset_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            doc = row["documento_id"]
            nivel_by_doc[doc] = int(row["nivel"])
            ini, fim = row["inicio"], row["fim"]
            classe = row["classificacao"]
            id_can = (row.get("id_canonico") or "").strip()
            doc_ids = id_can if id_can else "-"
            by_doc[doc].append(f"{ini},{fim},{classe},{doc_ids}")

    rows = []
    for doc, citacoes in by_doc.items():
        rows.append({
            "documento_id": doc,
            "nivel": nivel_by_doc[doc],
            "citacoes": "|".join(citacoes) if citacoes else "-",
        })
    return pd.DataFrame(rows)


def main():
    submission_path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    goldenset_path = sys.argv[2] if len(sys.argv) > 2 else "goldenset_offsets.csv"

    solution = build_solution_df(goldenset_path)
    submission = pd.read_csv(submission_path)

    resultado = avaliar(solution, submission, row_id="documento_id")

    for nivel, r in sorted(resultado["niveis"].items()):
        print(f"=== Nível {nivel} ===")
        print(f"  macro_f1 = {r['macro_f1']:.4f}")
        for c, f1 in r["f1_por_classe"].items():
            print(f"    F1[{c}] = {f1:.4f}")
        print(f"  tau (frac. inventada->real) = {r['tau']:.4f}")
        print(f"  s (macro_f1 com penalidade)  = {r['s']:.4f}")
        print(f"  b (bônus de calibração)      = {r['b']:.4f}")
        print(f"  score (nível, com bônus)     = {r['score']:.4f}")
        print()

    print(f"=== SCORE FINAL ===")
    print(f"{resultado['score_final']:.4f}")


if __name__ == "__main__":
    main()
