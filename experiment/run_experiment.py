# -*- coding: utf-8 -*-
"""
Orquestra o experimento: 5-fold por documento, treina/roda as 3 pipelines
em cada fold, agrega métricas e escreve experiment/results/{results.csv,report.md}.

Uso: python experiment/run_experiment.py
"""
import csv
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline_pipeline
import ml_pipeline
import llm_pipeline
from common import match_spans, precision_recall_f1
from folds import make_folds, N_FOLDS

DATA_DIR = Path(__file__).resolve().parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
INSTANCES_PATH = DATA_DIR / "instances.jsonl"
TRAINING_PATH = DATA_DIR / "training_examples.jsonl"

N_SYNTH_PER_FOLD_TEST = 3  # limita custo da LLM local (~70s/chamada) -- ver report.md
N_FEWSHOT_PER_TIPO = 3
RNG_SEED = 7

METHODS = ["baseline", "ml_classifier", "llm_local"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def base_doc_ids(instances):
    return sorted({i["base_doc_id"] for i in instances if i["kind"] == "real"})


def select_test_instances(all_instances, fold_assignment, fold_k, rng):
    real = [i for i in all_instances
            if i["kind"] == "real" and fold_assignment[i["base_doc_id"]] == fold_k]
    synth = [i for i in all_instances
             if i["kind"] == "synthetic" and fold_assignment[i["base_doc_id"]] == fold_k]
    rng.shuffle(synth)
    synth = synth[:N_SYNTH_PER_FOLD_TEST]
    return real + synth


def pick_few_shot(train_windows, rng):
    by_label = defaultdict(list)
    for w in train_windows:
        if w["label"].startswith("vago_"):
            by_label[w["label"]].append(w)
    out = []
    for label, ws in by_label.items():
        tipo = label.replace("vago_", "")
        sample = rng.sample(ws, min(N_FEWSHOT_PER_TIPO, len(ws)))
        for w in sample:
            out.append({"trecho": w["window_text"], "tipo": tipo})
    return out


def gold_spans_and_tipo(instance):
    spans = [(c["inicio"], c["fim"]) for c in instance["vago_citations"]]
    tipos = [c["tipo"] for c in instance["vago_citations"]]
    return spans, tipos


def run_method(method, instance, model=None, few_shot=None):
    text = instance["text"]
    if method == "baseline":
        return baseline_pipeline.extract_spans(text)
    if method == "ml_classifier":
        return ml_pipeline.extract_spans(model, text)
    if method == "llm_local":
        return llm_pipeline.extract_spans(text, few_shot)
    raise ValueError(method)


def main():
    rng = random.Random(RNG_SEED)
    instances = load_jsonl(INSTANCES_PATH)
    training_examples = load_jsonl(TRAINING_PATH)

    doc_ids = base_doc_ids(instances)
    fold_assignment = make_folds(doc_ids)
    print(f"{len(doc_ids)} documentos originais em {N_FOLDS} folds:",
          {k: sum(1 for v in fold_assignment.values() if v == k) for k in range(N_FOLDS)})

    # métricas por fold por método
    per_fold_metrics = {m: [] for m in METHODS}
    per_method_time = {m: 0.0 for m in METHODS}
    per_method_calls = {m: 0 for m in METHODS}
    tipo_correct = {m: 0 for m in METHODS}
    tipo_total = {m: 0 for m in METHODS}

    for fold_k in range(N_FOLDS):
        test_instances = select_test_instances(instances, fold_assignment, fold_k, rng)
        train_docs = {d for d in doc_ids if fold_assignment[d] != fold_k}
        train_windows = [w for w in training_examples if w["base_doc_id"] in train_docs]

        print(f"\n=== Fold {fold_k}: {len(test_instances)} instâncias de teste, "
              f"{len(train_windows)} janelas de treino ===")

        t0 = time.time()
        model = ml_pipeline.train(train_windows)
        train_time = time.time() - t0
        print(f"  Pipeline A treinada em {train_time:.1f}s")

        few_shot = pick_few_shot(train_windows, rng)

        gold_n = pred_n = matched_n = 0
        counts = {m: {"gold": 0, "pred": 0, "matched": 0} for m in METHODS}

        for inst in test_instances:
            gold_spans, gold_tipos = gold_spans_and_tipo(inst)

            for method in METHODS:
                t0 = time.time()
                preds = run_method(method, inst, model=model, few_shot=few_shot)
                per_method_time[method] += time.time() - t0
                per_method_calls[method] += 1

                pred_spans = [(p[0], p[1]) for p in preds]
                matches, used_g, used_p = match_spans(gold_spans, pred_spans, threshold=0.5)
                counts[method]["gold"] += len(gold_spans)
                counts[method]["pred"] += len(pred_spans)
                counts[method]["matched"] += len(matches)
                for gi, pi, _v in matches:
                    tipo_total[method] += 1
                    if preds[pi][2] == gold_tipos[gi]:
                        tipo_correct[method] += 1

        for method in METHODS:
            c = counts[method]
            p, r, f1 = precision_recall_f1(c["gold"], c["pred"], c["matched"])
            per_fold_metrics[method].append({
                "fold": fold_k, "gold": c["gold"], "pred": c["pred"],
                "matched": c["matched"], "precision": p, "recall": r, "f1": f1,
            })
            print(f"  {method:14} gold={c['gold']:3} pred={c['pred']:3} "
                  f"matched={c['matched']:3}  P={p:.3f} R={r:.3f} F1={f1:.3f}")

    write_results(per_fold_metrics, per_method_time, per_method_calls, tipo_correct, tipo_total)


def write_results(per_fold_metrics, per_method_time, per_method_calls, tipo_correct, tipo_total):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "fold", "gold", "pred", "matched", "precision", "recall", "f1"])
        for method, folds in per_fold_metrics.items():
            for row in folds:
                w.writerow([method, row["fold"], row["gold"], row["pred"], row["matched"],
                            f"{row['precision']:.4f}", f"{row['recall']:.4f}", f"{row['f1']:.4f}"])

    summary = {}
    for method, folds in per_fold_metrics.items():
        f1s = [r["f1"] for r in folds]
        ps = [r["precision"] for r in folds]
        rs = [r["recall"] for r in folds]
        summary[method] = {
            "f1_mean": statistics.mean(f1s), "f1_std": statistics.pstdev(f1s),
            "p_mean": statistics.mean(ps), "r_mean": statistics.mean(rs),
            "tipo_acc": (tipo_correct[method] / tipo_total[method]) if tipo_total[method] else float("nan"),
            "total_time": per_method_time[method], "calls": per_method_calls[method],
            "time_per_call": per_method_time[method] / per_method_calls[method] if per_method_calls[method] else 0,
        }

    # comparação pareada (Wilcoxon) entre F1 por fold, quando scipy disponível
    wilcoxon_results = {}
    try:
        from scipy.stats import wilcoxon
        methods = list(per_fold_metrics.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                a = [r["f1"] for r in per_fold_metrics[methods[i]]]
                b = [r["f1"] for r in per_fold_metrics[methods[j]]]
                try:
                    stat, p_value = wilcoxon(a, b)
                except ValueError:
                    stat, p_value = float("nan"), float("nan")
                wilcoxon_results[(methods[i], methods[j])] = (stat, p_value)
    except ImportError:
        pass

    write_report(summary, per_fold_metrics, wilcoxon_results)
    print("\nResultados em", RESULTS_DIR / "results.csv", "e", RESULTS_DIR / "report.md")


def write_report(summary, per_fold_metrics, wilcoxon_results):
    lines = []
    lines.append("# Detecção de menção vaga: regex fechado vs. classificador leve vs. LLM local\n")
    lines.append("## Metodologia\n")
    lines.append(
        "Comparação isolada na sub-tarefa de encontrar spans de **menção vaga** "
        "(citação sem identificador -- classe `incompleta` do contrato, excluindo os "
        "casos `relator+ano+tribunal` que já resolvem por consulta direta à base). "
        "A resolução downstream (real/inventada/incompleta) permanece idêntica e "
        "determinística para os 3 métodos -- só o extrator de spans muda.\n"
    )
    lines.append(
        f"- **Dados**: 26 documentos originais (dev set do desafio) + variantes "
        "sintéticas por paráfrase (geradas por `google/gemma-4-e4b` local via LM "
        "Studio, filtradas e checadas manualmente) + ruído de superfície leve.\n"
        f"- **Split**: {N_FOLDS}-fold estratificado por nível, no nível de "
        "DOCUMENTO ORIGINAL -- variantes sintéticas herdam o fold do documento-base, "
        "sem vazamento.\n"
        "- **Pipeline 0 (baseline)**: vocabulário fechado de ~15 frases + fuzzy match "
        "(`hunter/patterns.py`, `hunter/fuzzy.py`) -- não treina.\n"
        "- **Pipeline A**: TF-IDF de n-gramas de caractere + regressão logística "
        "multiclasse, treinada por fold só nos documentos de treino.\n"
        "- **Pipeline B**: LLM local (`qwen2.5vl:latest`, Ollama) com poucos exemplos "
        "do fold de treino + texto do documento de teste inteiro.\n"
        f"- Por custo de inferência da LLM local, o conjunto de teste sintético foi "
        f"amostrado em até {N_SYNTH_PER_FOLD_TEST} instâncias por fold "
        "(mesmo subconjunto para as 3 pipelines).\n"
    )

    lines.append("## Resultados agregados (média ± desvio-padrão entre folds)\n")
    lines.append("| Método | Precisão | Recall | F1 | Acc. tipo | Tempo/chamada | Chamadas |")
    lines.append("|---|---|---|---|---|---|---|")
    name_map = {"baseline": "0 - Baseline (regex fechado)",
                "ml_classifier": "A - Classificador leve",
                "llm_local": "B - LLM local (qwen2.5vl)"}
    for method, s in summary.items():
        f1_by_fold = [r["f1"] for r in per_fold_metrics[method]]
        std = statistics.pstdev(f1_by_fold)
        lines.append(
            f"| {name_map.get(method, method)} | {s['p_mean']:.3f} | {s['r_mean']:.3f} | "
            f"{s['f1_mean']:.3f} ± {std:.3f} | {s['tipo_acc']:.3f} | "
            f"{s['time_per_call']*1000:.1f}ms | {s['calls']} |"
        )
    lines.append("")

    lines.append("## Por fold\n")
    lines.append("| Fold | Método | Gold | Pred | Matched | Precisão | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for method, folds in per_fold_metrics.items():
        for row in folds:
            lines.append(
                f"| {row['fold']} | {name_map.get(method, method)} | {row['gold']} | "
                f"{row['pred']} | {row['matched']} | {row['precision']:.3f} | "
                f"{row['recall']:.3f} | {row['f1']:.3f} |"
            )
    lines.append("")

    lines.append("## Comparação pareada (Wilcoxon signed-rank sobre F1 por fold)\n")
    if wilcoxon_results:
        lines.append("| Par | estatística | p-valor |")
        lines.append("|---|---|---|")
        for (a, b), (stat, p) in wilcoxon_results.items():
            lines.append(f"| {name_map.get(a,a)} vs. {name_map.get(b,b)} | {stat:.3f} | {p:.3f} |")
        lines.append(
            "\n**Ressalva de poder estatístico**: com apenas 5 folds, o teste de "
            "Wilcoxon tem poder muito baixo (o p-valor mínimo possível já é alto); "
            "estes números são indicativos, não conclusivos."
        )
    else:
        lines.append("scipy não disponível -- comparação pareada não computada.")
    lines.append("")

    lines.append("## Custo computacional\n")
    for method, s in summary.items():
        lines.append(f"- **{name_map.get(method, method)}**: "
                      f"{s['total_time']:.1f}s totais em {s['calls']} chamadas "
                      f"({s['time_per_call']*1000:.0f}ms/chamada em média).")
    lines.append(
        "\nA Pipeline 0 não usa RAM além do processo Python; a Pipeline A usa um "
        "modelo TF-IDF+LogReg de poucos MB; a Pipeline B mantém ~6GB do `qwen2.5vl` "
        "residente (a própria máquina de teste tem só 16GB de RAM, não os 32GB do "
        "enunciado)."
    )
    lines.append("")

    lines.append("## Limitações\n")
    lines.append(
        "- N pequeno (26 documentos originais, ~33 menções vago reais) -- os "
        "intervalos de confiança implícitos nesses números são largos.\n"
        "- O ruído sintético (paráfrase + ruído de superfície) é uma aproximação "
        "nossa do gerador real do desafio, que é desconhecido; pode não bater com "
        "a distribuição exata do conjunto cego.\n"
        "- Um único modelo local (`qwen2.5vl:latest`, multimodal, não especializado "
        "em texto jurídico) não representa LLMs em geral -- um modelo maior ou mais "
        "especializado poderia ter desempenho bem diferente.\n"
        "- As paráfrases foram geradas por outro modelo local (`google/gemma-4-e4b`, "
        "via LM Studio) para evitar viés/vazamento a favor da Pipeline B, mas isso "
        "introduz o viés inverso: a qualidade da paráfrase é limitada pelo que esse "
        "modelo consegue gerar em português (uma fração das gerações veio em inglês/"
        "quase-duplicata e foi "
        "descartada por filtro automático).\n"
        "- Por custo, a LLM local não foi avaliada em TODAS as variantes sintéticas, "
        "só numa amostra por fold (ver Metodologia) -- isso reduz ainda mais o N "
        "efetivo da Pipeline B especificamente."
    )

    with open(RESULTS_DIR / "report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
