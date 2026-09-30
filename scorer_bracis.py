# -*- coding: utf-8 -*-
"""
Reconstrução da métrica OFICIAL descrita na página do desafio
("Verificação de citações jurídicas em pareceres gerados por IA -
Jusbrasil x BRACIS 2026"), já que o kaggle_metric.py real não está
disponível neste diretório:

  - F1 macro sobre as 3 classes (real, inventada, incompleta), calculado
    por nível.
  - Penalidade dupla para classificar como real uma citação inventada
    (conta 2x no FN da classe inventada e no FP da classe real).
  - Bônus de calibração de até 10% via Brier score da confiança.
  - Score final = média ponderada dos dois níveis (pesos 1x nível 1,
    2x nível 2, conforme a página de dados do desafio).

ATENÇÃO: a fórmula exata de como o bônus de calibração entra na conta
("a média ponderada das métricas") não é dada em detalhe -- isto é uma
reconstrução plausível, não uma cópia do script oficial. Serve para
estimar direção e magnitude, não para prever o número exato do
leaderboard.

Uso: python scorer_bracis.py <pasta_json> [goldenset.csv]
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

CLASSES = ["real", "inventada", "incompleta"]
NIVEL_WEIGHT = {"1": 1, "2": 2}


def iou(a, b):
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    inter = max(0, e - s)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def load_gold(path):
    by_doc = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["inicio"] = int(row["inicio"])
            row["fim"] = int(row["fim"])
            by_doc[row["documento_id"]].append(row)
    return by_doc


def load_pred(pasta, documento_id):
    p = Path(pasta) / f"{documento_id}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8")).get("citacoes", [])


def match_doc(gold_rows, pred_rows):
    pairs = []
    for gi, g in enumerate(gold_rows):
        for pi, p in enumerate(pred_rows):
            v = iou((g["inicio"], g["fim"]), (p["inicio"], p["fim"]))
            if v >= 0.5:
                pairs.append((v, gi, pi))
    pairs.sort(key=lambda t: -t[0])
    used_g, used_p, matches = set(), set(), []
    for v, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi))
    return matches, used_g, used_p


def f1_macro_for_level(gold_by_doc, pasta, doc_ids):
    tp = defaultdict(float)
    fp = defaultdict(float)
    fn = defaultdict(float)
    double_penalty_events = []
    confusion = defaultdict(int)

    for doc_id in doc_ids:
        gold_rows = gold_by_doc[doc_id]
        pred_rows = load_pred(pasta, doc_id)
        matches, used_g, used_p = match_doc(gold_rows, pred_rows)

        for gi, pi in matches:
            gc = gold_rows[gi]["classificacao"]
            pc = pred_rows[pi]["classificacao"]
            confusion[(gc, pc)] += 1
            weight = 2.0 if (gc == "inventada" and pc == "real") else 1.0
            if weight == 2.0:
                double_penalty_events.append((doc_id, gold_rows[gi]["citacao_id"]))
            if gc == pc:
                tp[gc] += 1.0
            else:
                fn[gc] += weight
                fp[pc] += weight

        for gi, g in enumerate(gold_rows):
            if gi not in used_g:
                fn[g["classificacao"]] += 1.0  # não entregou span -> erro de recall
        for pi, p in enumerate(pred_rows):
            if pi not in used_p:
                fp[p["classificacao"]] += 1.0  # span espúrio -> erro de precisão

    f1_per_class = {}
    for c in CLASSES:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1_per_class[c] = (prec, rec, f1)

    f1_macro = sum(f1_per_class[c][2] for c in CLASSES) / len(CLASSES)
    return f1_macro, f1_per_class, confusion, double_penalty_events


def brier_score(gold_by_doc, pasta, doc_ids):
    """Brier score da confiança reportada -- só sobre os pares alinhados
    (span encontrado), já que confiança não se aplica a um FN."""
    errors = []
    for doc_id in doc_ids:
        gold_rows = gold_by_doc[doc_id]
        pred_rows = load_pred(pasta, doc_id)
        matches, _, _ = match_doc(gold_rows, pred_rows)
        for gi, pi in matches:
            gc = gold_rows[gi]["classificacao"]
            pc = pred_rows[pi]["classificacao"]
            conf = pred_rows[pi].get("confianca")
            if conf is None:
                continue
            correct = 1.0 if gc == pc else 0.0
            errors.append((float(conf) - correct) ** 2)
    if not errors:
        return None, 0
    return sum(errors) / len(errors), len(errors)


def main():
    if len(sys.argv) < 2:
        sys.exit("uso: python scorer_bracis.py <pasta_json> [goldenset.csv]")
    pasta_json = sys.argv[1]
    gold_csv = sys.argv[2] if len(sys.argv) > 2 else "goldenset.csv"
    gold_by_doc = load_gold(gold_csv)

    docs_by_nivel = defaultdict(list)
    for doc_id, rows in gold_by_doc.items():
        docs_by_nivel[rows[0]["nivel"]].append(doc_id)

    nivel_scores = {}
    for nivel in sorted(docs_by_nivel):
        f1_macro, f1_per_class, confusion, dbl = f1_macro_for_level(
            gold_by_doc, pasta_json, docs_by_nivel[nivel])
        brier, n_brier = brier_score(gold_by_doc, pasta_json, docs_by_nivel[nivel])
        nivel_scores[nivel] = f1_macro

        print(f"=== Nível {nivel} ({len(docs_by_nivel[nivel])} documentos) ===")
        print(f"F1 macro (3 classes, com penalidade dupla p/ inventada->real) = {f1_macro:.4f}")
        for c in CLASSES:
            p, r, f1 = f1_per_class[c]
            print(f"  {c:12} precisão={p:.3f}  recall={r:.3f}  F1={f1:.3f}")
        if dbl:
            print(f"  ATENÇÃO: {len(dbl)} evento(s) de penalidade dupla (inventada classificada como real):")
            for x in dbl:
                print("   ", x)
        else:
            print("  Nenhum erro inventada->real (o erro mais penalizado) -- OK.")
        print("  Matriz de confusão (gold -> pred: count):")
        for (gc, pc), n in sorted(confusion.items(), key=lambda t: -t[1]):
            marker = "" if gc == pc else ("  <-- PENALIDADE DUPLA" if (gc == "inventada" and pc == "real") else "  <-- erro")
            print(f"    {gc:12} -> {pc:12} : {n}{marker}")
        if brier is not None:
            # bônus de calibração heurístico: 10% * (1 - brier/0.25), floor em 0
            # (0.25 = Brier de um "chute" constante de confiança 0.5, tomado
            # como piso de referência; Brier=0 dá bônus cheio de 10%)
            bonus = max(0.0, 1 - brier / 0.25) * 0.10
            print(f"  Brier score da confiança (n={n_brier}) = {brier:.4f}  "
                  f"-> bônus de calibração estimado ~{bonus*100:.1f}%")
        else:
            print("  Nenhuma confiança reportada nos pares alinhados.")
        print()

    if len(nivel_scores) == 2 and all(str(n) in NIVEL_WEIGHT for n in nivel_scores):
        w1, w2 = NIVEL_WEIGHT["1"], NIVEL_WEIGHT["2"]
        final = (w1 * nivel_scores["1"] + w2 * nivel_scores["2"]) / (w1 + w2)
        print(f"=== Score final estimado (peso {w1}x nível 1, {w2}x nível 2) ===")
        print(f"F1 macro combinado = {final:.4f}")


if __name__ == "__main__":
    main()
