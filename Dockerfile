# Pipeline 100% determinístico (regex + sqlite3 + difflib -- tudo stdlib
# do Python). Nenhum modelo, nenhum peso, nenhuma GPU necessária: a
# imagem só existe para fixar a versão do Python e isolar o ambiente,
# conforme pedido pela organização.
FROM python:3.11-slim

WORKDIR /app

COPY hunter/ ./hunter/
COPY run.py json_to_submission.py run.sh ./
RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
