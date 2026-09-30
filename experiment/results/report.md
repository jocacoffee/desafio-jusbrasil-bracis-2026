# Detecção de menção vaga: regex fechado vs. classificador leve vs. LLM local

## Metodologia

Comparação isolada na sub-tarefa de encontrar spans de **menção vaga** (citação sem identificador -- classe `incompleta` do contrato, excluindo os casos `relator+ano+tribunal` que já resolvem por consulta direta à base). A resolução downstream (real/inventada/incompleta) permanece idêntica e determinística para os 3 métodos -- só o extrator de spans muda.

- **Dados**: 26 documentos originais (dev set do desafio) + variantes sintéticas por paráfrase (geradas por `google/gemma-4-e4b` local via LM Studio, filtradas e checadas manualmente) + ruído de superfície leve.
- **Split**: 5-fold estratificado por nível, no nível de DOCUMENTO ORIGINAL -- variantes sintéticas herdam o fold do documento-base, sem vazamento.
- **Pipeline 0 (baseline)**: vocabulário fechado de ~15 frases + fuzzy match (`hunter/patterns.py`, `hunter/fuzzy.py`) -- não treina.
- **Pipeline A**: TF-IDF de n-gramas de caractere + regressão logística multiclasse, treinada por fold só nos documentos de treino.
- **Pipeline B**: LLM local (`qwen2.5vl:latest`, Ollama) com poucos exemplos do fold de treino + texto do documento de teste inteiro.
- Por custo de inferência da LLM local, o conjunto de teste sintético foi amostrado em até 3 instâncias por fold (mesmo subconjunto para as 3 pipelines).

## Resultados agregados (média ± desvio-padrão entre folds)

| Método | Precisão | Recall | F1 | Acc. tipo | Tempo/chamada | Chamadas |
|---|---|---|---|---|---|---|
| 0 - Baseline (regex fechado) | 1.000 | 0.746 | 0.852 ± 0.050 | 1.000 | 1511.2ms | 41 |
| A - Classificador leve | 0.243 | 0.848 | 0.373 ± 0.101 | 1.000 | 134.6ms | 41 |
| B - LLM local (qwen2.5vl) | 0.068 | 0.093 | 0.074 ± 0.077 | 1.000 | 13378.0ms | 41 |

## Por fold

| Fold | Método | Gold | Pred | Matched | Precisão | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 0 | 0 - Baseline (regex fechado) | 13 | 10 | 10 | 1.000 | 0.769 | 0.870 |
| 1 | 0 - Baseline (regex fechado) | 8 | 5 | 5 | 1.000 | 0.625 | 0.769 |
| 2 | 0 - Baseline (regex fechado) | 15 | 12 | 12 | 1.000 | 0.800 | 0.889 |
| 3 | 0 - Baseline (regex fechado) | 10 | 7 | 7 | 1.000 | 0.700 | 0.824 |
| 4 | 0 - Baseline (regex fechado) | 18 | 15 | 15 | 1.000 | 0.833 | 0.909 |
| 0 | A - Classificador leve | 13 | 52 | 13 | 0.250 | 1.000 | 0.400 |
| 1 | A - Classificador leve | 8 | 44 | 6 | 0.136 | 0.750 | 0.231 |
| 2 | A - Classificador leve | 15 | 55 | 12 | 0.218 | 0.800 | 0.343 |
| 3 | A - Classificador leve | 10 | 36 | 8 | 0.222 | 0.800 | 0.348 |
| 4 | A - Classificador leve | 18 | 41 | 16 | 0.390 | 0.889 | 0.542 |
| 0 | B - LLM local (qwen2.5vl) | 13 | 33 | 1 | 0.030 | 0.077 | 0.043 |
| 1 | B - LLM local (qwen2.5vl) | 8 | 27 | 0 | 0.000 | 0.000 | 0.000 |
| 2 | B - LLM local (qwen2.5vl) | 15 | 17 | 1 | 0.059 | 0.067 | 0.062 |
| 3 | B - LLM local (qwen2.5vl) | 10 | 37 | 1 | 0.027 | 0.100 | 0.043 |
| 4 | B - LLM local (qwen2.5vl) | 18 | 18 | 4 | 0.222 | 0.222 | 0.222 |

## Comparação pareada (Wilcoxon signed-rank sobre F1 por fold)

| Par | estatística | p-valor |
|---|---|---|
| 0 - Baseline (regex fechado) vs. A - Classificador leve | 0.000 | 0.062 |
| 0 - Baseline (regex fechado) vs. B - LLM local (qwen2.5vl) | 0.000 | 0.062 |
| A - Classificador leve vs. B - LLM local (qwen2.5vl) | 0.000 | 0.062 |

**Ressalva de poder estatístico**: com apenas 5 folds, o teste de Wilcoxon tem poder muito baixo (o p-valor mínimo possível já é alto); estes números são indicativos, não conclusivos.

## Custo computacional

- **0 - Baseline (regex fechado)**: 62.0s totais em 41 chamadas (1511ms/chamada em média).
- **A - Classificador leve**: 5.5s totais em 41 chamadas (135ms/chamada em média).
- **B - LLM local (qwen2.5vl)**: 548.5s totais em 41 chamadas (13378ms/chamada em média).

A Pipeline 0 não usa RAM além do processo Python; a Pipeline A usa um modelo TF-IDF+LogReg de poucos MB; a Pipeline B mantém ~6GB do `qwen2.5vl` residente (a própria máquina de teste tem só 16GB de RAM, não os 32GB do enunciado).

## Limitações

- N pequeno (26 documentos originais, ~33 menções vago reais) -- os intervalos de confiança implícitos nesses números são largos.
- O ruído sintético (paráfrase + ruído de superfície) é uma aproximação nossa do gerador real do desafio, que é desconhecido; pode não bater com a distribuição exata do conjunto cego.
- Um único modelo local (`qwen2.5vl:latest`, multimodal, não especializado em texto jurídico) não representa LLMs em geral -- um modelo maior ou mais especializado poderia ter desempenho bem diferente.
- As paráfrases foram geradas por outro modelo local (`google/gemma-4-e4b`, via LM Studio) para evitar viés/vazamento a favor da Pipeline B, mas isso introduz o viés inverso: a qualidade da paráfrase é limitada pelo que esse modelo consegue gerar em português (uma fração das gerações veio em inglês/quase-duplicata e foi descartada por filtro automático).
- Por custo, a LLM local não foi avaliada em TODAS as variantes sintéticas, só numa amostra por fold (ver Metodologia) -- isso reduz ainda mais o N efetivo da Pipeline B especificamente.