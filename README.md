# Caça-Alucinações · BRACIS 2026 × Jusbrasil — Desafio 1

Pipeline determinístico para encontrar citações jurídicas (jurisprudência e
lei) em peças judiciais e classificá-las contra a base canônica congelada
como `real`, `inventada` ou `incompleta`.

## Abordagem

**Sem modelo, sem LLM, sem etapa de treino.** A solução inteira é regex +
normalização de texto + consulta em memória contra o SQLite fornecido
(`sqlite3` e o resto da biblioteca padrão do Python — nenhuma dependência
externa). Isso não é uma limitação: é a escolha correta para esta tarefa
específica. Testamos formalmente essa premissa (ver `experiment/`, abaixo)
comparando o pipeline determinístico contra um classificador leve treinado
e uma LLM local na sub-tarefa mais frágil (detecção de menção sem
identificador) — o determinístico venceu com folga nos dois casos, sempre
com precisão perfeita, a um custo de milissegundos por documento contra
segundos por chamada de LLM.

A resolução real/inventada/incompleta é, por natureza, um problema de
busca exata contra uma base fechada e congelada — nenhum classificador ou
modelo de linguagem generaliza melhor do que consultar o registro
diretamente. O trabalho de engenharia está inteiro do lado de **achar e
normalizar** a citação antes da consulta: casar abreviações de classe
processual, reconstruir números com ruído de OCR (0↔O, 1↔l, 5↔S, m↔rn e
variantes observadas), tolerar quebra de linha e variação de pontuação, e
desambiguar quando a base tem mais de um registro para o mesmo número de
processo (ex.: acórdão original + embargos de declaração/divergência
posteriores).

### Estrutura

```
hunter/
  db.py         -- carrega o .db, indexa em memória (número próprio de cada
                   acórdão, súmulas, dispositivos de lei)
  patterns.py   -- biblioteca de regex (classes processuais, números,
                   súmulas, dispositivos, menções incompletas buscáveis)
  extract.py    -- M1: extrai spans de citação de um documento
  resolve.py    -- M3/M4: resolve cada span contra a base (real/
                   inventada/incompleta + confiança calibrada)
  pipeline.py   -- une extração + resolução, monta o JSON do contrato
  normalize.py, fuzzy.py -- utilitários de normalização/casamento difuso
run.py                    -- roda o pipeline sobre uma pasta de .txt
json_to_submission.py     -- converte a saída em JSON para o CSV de submissão
run.sh                     -- ponto de entrada único (ver "Como rodar")
```

Nenhum "enriquecimento" do `.db` é persistido em disco: `hunter/db.py`
constrói os índices em memória a cada execução, a partir do `.db` no
formato original. Não há etapa de pré-processamento separada a rodar
antes — o mesmo comando de execução já cobre isso.

### `experiment/` — validação da premissa "determinístico > aprendido"

Diretório à parte, **não faz parte da execução da solução** (por isso não
entra na imagem Docker). Contém o estudo comparativo de 5-fold entre o
vocabulário fechado, um classificador TF-IDF+regressão logística treinado,
e uma LLM local (`qwen2.5vl` via Ollama), na tarefa de detectar menções
sem identificador. Resultado e metodologia completos em
`experiment/results/report.md`. Os dados sintéticos usados nesse estudo
foram gerados por LLMs locais (Ollama/LM Studio) só para fins de
desenvolvimento — não participam da execução da solução final, então
ficam fora do limite de hardware/GPU da avaliação.

## Como rodar

### Docker (recomendado — ambiente isolado e reproduzível)

```bash
docker build -t bracis-solution .

docker run --rm \
  -v "/caminho/para/desafio1_bracis.db:/data/desafio1_bracis.db:ro" \
  -v "/caminho/para/txt:/data/txt:ro" \
  -v "/caminho/para/saida:/out" \
  bracis-solution /data/desafio1_bracis.db /data/txt /out/submission.csv
```

### Direto (Python 3.9+, sem dependências)

```bash
./run.sh <caminho_db> <pasta_txt> <arquivo_saida>
```

Em ambos os casos, `<arquivo_saida>` recebe um CSV no formato de
submissão (`documento_id,citacoes`), o mesmo enviado ao leaderboard do
Kaggle. Execução determinística: mesma entrada produz sempre a mesma
saída, em qualquer máquina — não há amostragem nem seed a fixar.

### Requisitos de hardware

CPU apenas, sem GPU. ~26 documentos processam em segundos; o tempo
dominante é a indexação em memória dos ~1.000 registros da base canônica
na inicialização (uma vez por execução), não o processamento por
documento.

## Validação

A saída foi conferida contra `kaggle_metric.py` (o script oficial de
avaliação, obtido da aba Data do Kaggle) usando o `goldenset_offsets.csv`
final: **F1 macro = 1,0000 nos dois níveis, score final 1,0993** (teto
teórico da fórmula é 1,10000). Script de conferência local:
`run_official_metric.py`.
