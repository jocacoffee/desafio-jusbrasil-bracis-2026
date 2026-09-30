# -*- coding: utf-8 -*-
"""Cliente HTTP mínimo para um servidor OpenAI-compatible local (LM Studio),
usado para gerar paráfrases com google/gemma-4-e4b -- servidor numa outra
máquina da rede local do usuário, endereço fornecido explicitamente por ele
(não usado para nada além do que ele pediu: geração de paráfrases)."""
import json
import urllib.request

BASE_URL = "http://172.20.10.3:1234"


def generate(model: str, prompt: str, timeout: int = 120, temperature: float = 0.4) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"] or ""
