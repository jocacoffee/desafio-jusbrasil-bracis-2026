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
enrich_db.py                -- pré-processamento OPCIONAL (ver abaixo)
```

### Enriquecimento do `.db` (opcional)

`hunter/db.py` funciona direto sobre o `.db` no formato original, sem
nenhum passo antes — mas calcular o "número próprio" de cada um dos
~1.000 acórdãos varrendo regex sobre o texto inteiro custa **~26s por
execução** (medido). Como esse cálculo não muda para o mesmo `.db`,
`enrich_db.py` pré-computa uma vez e grava em 3 tabelas extras numa
**cópia** do banco (nunca sobrescreve o original):

```bash
python3 enrich_db.py <db_original> <db_enriquecido>
```

`hunter/db.py` detecta essas tabelas automaticamente e usa o caminho
rápido quando presentes (**~0,2s**, testado — mesmo resultado, índices
byte-idênticos aos calculados em memória, verificado por asserção antes
de aceitar o ganho). Sem o passo de enriquecimento, tudo funciona
exatamente igual, só ~130x mais devagar na inicialização — por isso é
opcional, não uma dependência: `run.sh`/`run.py` aceitam tanto o `.db`
original quanto um `.db` enriquecido no mesmo argumento, sem nenhuma
flag ou configuração extra.

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

### Tratamento de erros

`.db`/pasta de entrada ausentes, ou uma pasta de saída sem permissão de
escrita, param a execução com uma mensagem clara (não um traceback cru).
Um documento individual que falhe por algum motivo inesperado **não**
derruba a execução inteira: fica registrado um aviso em stderr e esse
documento entra na saída com lista de citações vazia, os demais seguem
normalmente. Um `.txt` que não seja UTF-8 válido é lido com substituição
tolerante de caracteres em vez de interromper o processamento.

### Requisitos de hardware

CPU apenas, sem GPU. ~26 documentos processam em milissegundos; o tempo
dominante é a indexação dos ~1.000 registros da base canônica na
inicialização — ~26s sem enriquecimento, ~0,2s com (ver seção acima).

## Validação

A saída foi conferida contra `kaggle_metric.py` (o script oficial de
avaliação, obtido da aba Data do Kaggle) usando o `goldenset_offsets.csv`
final: **F1 macro = 1,0000 nos dois níveis, score final 1,0993** (teto
teórico da fórmula é 1,10000). Script de conferência local:
`run_official_metric.py`.

Além do dev set, `experiment/stress_test_noise.py` gera variantes
sintéticas de ruído (troca de abreviação, reformatação de número, OCR
letra-a-dígito, separador de UF, quebra de linha) sobre as 192 citações
reais, empilhando mais ruído por citação do que o nível 2 real mostra —
um teste deliberadamente mais difícil que o esperado. Foi rodando esse
teste que achamos e corrigimos os 3 bugs descritos no commit
`b71a5a3`/seguintes (2 bugs reais de regex, 1 tolerância a ruído nova).

## Limitações conhecidas

- **Uma letra de OCR ambígua não é recuperada de propósito.** Em nível 2
  aparece ocasionalmente um "g"/"G" no lugar de um dígito (ex.:
  `"R.Esp. n° 1.45g.779-MA"`, que resolve para `1.459.779` — "g"→9). Ao
  contrário de 0↔O, 1↔l, 5↔S e m↔rn (pares com semelhança visual clara e
  mapeamento único, documentados no PDF do desafio), essa letra não tem
  correspondência visual óbvia com um único dígito, e um teste inicial
  mostrou o mesmo caractere precisando mapear para dígitos diferentes em
  citações distintas. Preferimos **não** adivinhar esse caso (deixando-o
  como falso negativo/`inventada`) a arriscar devolver um `id_canonico`
  incorreto com confiança alta — errar por omissão custa menos na métrica
  oficial do que errar afirmando uma classe errada com convicção.
- **A cobertura de abreviação de classe processual é ampla, mas não
  garantidamente exaustiva.** Foi checada contra os 26 documentos de dev,
  ampliada por auditoria de vocabulário (conhecimento de domínio + LLM
  local como fonte de candidatos, cada um verificado manualmente antes de
  entrar) e ampliada de novo por auditoria contra os 996 acórdãos + 5
  súmulas + 13 dispositivos de toda a base canônica atual (não só o que é
  citado nos 26 documentos de dev). Como o `.db` da avaliação final é
  outro, `hunter/extract.py` tem também um fallback estrutural: um número
  no formato CNJ padrão (13-20 dígitos, bem mais distintivo que um número
  curto solto) é aceito como citação mesmo sem reconhecer o nome da
  classe processual à sua frente — cobrindo uma classe genuinamente nova
  no `.db` novo, tanto para extrair a citação quanto para indexar o
  "número próprio" do acórdão correspondente na base. Testado em
  `experiment/stress_test_novel_classes.py` com 9 classes reais
  deliberadamente fora do vocabulário (ADI, ADPF, ADC, Mandado de
  Injunção, Habeas Data, Conflito de Competência, Revisão Criminal,
  Recurso de Reexame Necessário, Agravo de Execução): 9/9 de generalização
  quando o número é formato CNJ, com um teste de falso positivo
  específico para não confundir CNPJ/CPF/OAB (também têm 13+ dígitos)
  com um número de processo. **Continua 0/9** para uma classe nova citada
  no formato sequencial antigo (poucos dígitos) — não dá pra aceitar um
  número curto sem nome de classe OU formato distintivo sem abrir uma
  porta grande demais para falso positivo em qualquer número solto do
  texto. Esse risco residual (classe nova + número no formato antigo)
  fica documentado e aceito conscientemente, não escondido.
  Tentei também pedir a uma LLM local (qwen3:8b) que sugerisse classes
  processuais reais fora do vocabulário atual, para ampliar esse teste
  além do que eu já sabia de cor — o resultado veio majoritariamente com
  duplicatas triviais do que já estava coberto, mesmo com instrução
  explícita para não repetir. Descartei a sugestão automática e curei a
  lista de teste manualmente a partir de conhecimento de domínio
  verificado.
- **O stress-test sintético é uma aproximação nossa do ruído real**, não
  uma cópia do gerador oficial (desconhecido) do desafio. Ele empilha
  transformações de forma mais agressiva do que o nível 2 observado, então
  os números dele (recall ~88% no teste mais difícil vs. 100% no dev set)
  devem ser lidos como um piso pessimista de robustez, não como a
  expectativa real de desempenho no conjunto oculto. Investigando as
  falhas restantes, a maioria vem do próprio gerador de ruído aplicando
  troca de letra (principalmente "o"→"0") em palavras comuns do texto
  ("do"→"d0", "Apelação"→"Apelaçã0"), não em identificadores — o que foge
  do modelo de ruído documentado (concentrado em números/identificadores)
  e infla artificialmente a taxa de falha; tratamos isso como limitação
  do harness de teste, não do pipeline.
