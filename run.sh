#!/usr/bin/env bash
# Ponto de entrada único da solução.
#
# Uso:
#   ./run.sh <caminho_db> <pasta_txt> <arquivo_saida>
#
# <caminho_db>    -- desafio1_bracis.db (formato original, sem
#                    pré-processamento; o "enriquecimento" da base --
#                    índice de número próprio de cada acórdão, mapa
#                    artigo->código, número de súmula -- é feito em
#                    memória a cada execução por hunter/db.py, não há
#                    etapa de preparo separada nem arquivo intermediário
#                    a gerar antes)
# <pasta_txt>     -- pasta com os .txt de entrada (um por documento)
# <arquivo_saida> -- caminho do CSV de saída, no formato de submissão
#                    (documento_id,citacoes -- o mesmo enviado ao Kaggle)
#
# Determinístico: sem amostragem, sem seed, sem modelo. Mesma entrada
# sempre produz a mesma saída, em qualquer máquina.
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "uso: $0 <caminho_db> <pasta_txt> <arquivo_saida>" >&2
    exit 1
fi

DB_PATH="$1"
TXT_DIR="$2"
OUT_FILE="$3"

command -v python3 >/dev/null 2>&1 || { echo "ERRO: python3 não encontrado no PATH" >&2; exit 1; }
[ -f "$DB_PATH" ] || { echo "ERRO: base canônica (.db) não encontrada: $DB_PATH" >&2; exit 1; }
[ -d "$TXT_DIR" ] || { echo "ERRO: pasta de documentos não encontrada: $TXT_DIR" >&2; exit 1; }

OUT_DIR="$(dirname "$OUT_FILE")"
mkdir -p "$OUT_DIR" || { echo "ERRO: não foi possível criar a pasta de saída: $OUT_DIR" >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_JSON_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_JSON_DIR"' EXIT

python3 "$SCRIPT_DIR/run.py" "$TXT_DIR" "$DB_PATH" "$TMP_JSON_DIR"
python3 "$SCRIPT_DIR/json_to_submission.py" "$TMP_JSON_DIR" "$OUT_FILE"

echo "Saída escrita em $OUT_FILE"
