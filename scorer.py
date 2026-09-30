# -*- coding: utf-8 -*-
"""
Scorer local -- NÃO é o kaggle_metric.py oficial (não distribuído junto
com os dados neste diretório; ver PDF §5). Replica a regra descrita no
contrato para permitir iterar localmente contra o goldenset.csv de dev:

  - alinhamento predição<->gabarito por IoU de span >= 0.5 (matching
    guloso, um-para-um, maior IoU primeiro);
  - dentre os pares alinhados, acurácia de classe e de link (id_canonico,
    aceitando qualquer id do conjunto separado por espaço do gabarito);
  - recall/precisão em nível de span (entregar o span é pré-requisito
    para pontuar a classe).

Uso:
    python scorer.py <pasta_json> [goldenset.csv]
"""
import csv
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path


def iou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_gold(path):
    by_doc = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            row["inicio"] = int(row["inicio"])
            row["fim"] = int(row["fim"])
            by_doc[row["documento_id"]].append(row)
    return by_doc


def load_pred(pasta, documento_id):
    p = Path(pasta) / f"{documento_id}.json"
    if not p.exists():
        return []
    doc = json.loads(p.read_text(encoding="utf-8"))
    return doc.get("citacoes", [])


def match_doc(gold_rows, pred_rows):
    pairs = []
    for gi, g in enumerate(gold_rows):
        for pi, p in enumerate(pred_rows):
            v = iou((g["inicio"], g["fim"]), (p["inicio"], p["fim"]))
            if v >= 0.5:
                pairs.append((v, gi, pi))
    pairs.sort(key=lambda t: -t[0])
    used_g, used_p = set(), set()
    matches = []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, v))
    return matches, used_g, used_p


def main():
    if len(sys.argv) < 2:
        sys.exit("uso: python scorer.py <pasta_json> [goldenset.csv]")
    pasta_json = sys.argv[1]
    gold_csv = sys.argv[2] if len(sys.argv) > 2 else "goldenset.csv"

    gold_by_doc = load_gold(gold_csv)

    total_gold = 0
    total_pred = 0
    total_matched = 0
    class_correct = 0
    class_confusion = Counter()
    link_correct = 0
    link_total = 0
    per_nivel = defaultdict(lambda: {"gold": 0, "matched": 0, "class_ok": 0})
    fn_examples = []
    fp_examples = []
    wrong_class_examples = []

    for documento_id, gold_rows in sorted(gold_by_doc.items()):
        pred_rows = load_pred(pasta_json, documento_id)
        matches, used_g, used_p = match_doc(gold_rows, pred_rows)

        total_gold += len(gold_rows)
        total_pred += len(pred_rows)
        total_matched += len(matches)

        nivel = gold_rows[0]["nivel"] if gold_rows else "?"
        per_nivel[nivel]["gold"] += len(gold_rows)
        per_nivel[nivel]["matched"] += len(matches)

        for gi, pi, v in matches:
            g, p = gold_rows[gi], pred_rows[pi]
            gc, pc = g["classificacao"], p["classificacao"]
            class_confusion[(gc, pc)] += 1
            if gc == pc:
                class_correct += 1
                per_nivel[nivel]["class_ok"] += 1
            else:
                wrong_class_examples.append((documento_id, g["citacao_id"],
                                              gc, pc, g["trecho"][:50]))
            if gc == "real":
                link_total += 1
                gold_ids = set((g.get("id_canonico") or "").split())
                pred_res = p.get("resolucao") or {}
                pred_id = str(pred_res.get("id_canonico", "")).strip()
                if pred_id and pred_id in gold_ids:
                    link_correct += 1

        for gi, g in enumerate(gold_rows):
            if gi not in used_g:
                fn_examples.append((documento_id, g["citacao_id"], g["classificacao"],
                                     g["trecho"][:50]))
        for pi, p in enumerate(pred_rows):
            if pi not in used_p:
                fp_examples.append((documento_id, p["trecho"][:50], p["classificacao"]))

    print("=== Span alignment (IoU >= 0.5) ===")
    print(f"gold={total_gold} pred={total_pred} matched={total_matched}")
    if total_gold:
        print(f"recall (span) = {total_matched/total_gold:.3f}")
    if total_pred:
        print(f"precision (span) = {total_matched/total_pred:.3f}")

    print("\n=== Classificação (entre os pares alinhados) ===")
    if total_matched:
        print(f"acurácia de classe = {class_correct/total_matched:.3f}")
    print("confusão (gold -> pred): count")
    for (gc, pc), n in sorted(class_confusion.items(), key=lambda t: -t[1]):
        marker = "" if gc == pc else "  <-- erro"
        print(f"  {gc:12} -> {pc:12} : {n}{marker}")

    print("\n=== Link (id_canonico) entre os 'real' alinhados ===")
    if link_total:
        print(f"acurácia de link = {link_correct}/{link_total} = {link_correct/link_total:.3f}")

    print("\n=== Por nível ===")
    for nivel, d in sorted(per_nivel.items()):
        r = d["matched"] / d["gold"] if d["gold"] else 0
        c = d["class_ok"] / d["matched"] if d["matched"] else 0
        print(f"  nível {nivel}: gold={d['gold']} matched={d['matched']} "
              f"recall={r:.3f} class_acc={c:.3f}")

    print(f"\n=== Falsos negativos (gold sem span correspondente): {len(fn_examples)} ===")
    for ex in fn_examples[:20]:
        print(" ", ex)

    print(f"\n=== Falsos positivos (pred sem gold correspondente): {len(fp_examples)} ===")
    for ex in fp_examples[:20]:
        print(" ", ex)

    print(f"\n=== Erros de classe (amostra): {len(wrong_class_examples)} ===")
    for ex in wrong_class_examples[:30]:
        print(" ", ex)


if __name__ == "__main__":
    main()
