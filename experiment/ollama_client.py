# -*- coding: utf-8 -*-
"""Cliente HTTP minimo para a API local do Ollama (stdlib only)."""
import json
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate(model: str, prompt: str, timeout: int = 300, temperature: float = 0.7) -> str:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("response", "")


def extract_json(text: str):
    """Extrai o primeiro objeto/array JSON de uma resposta que pode vir
    com cercas markdown, texto de raciocínio (<think>...</think>) etc."""
    # remove bloco de pensamento de modelos tipo deepseek-r1
    if "<think>" in text and "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") or part.startswith("["):
                text = part
                break
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("nenhum JSON encontrado na resposta: " + text[:200])
    # tenta achar o fechamento correspondente varrendo do fim
    for end in range(len(text), start, -1):
        candidate = text[start:end]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("JSON malformado na resposta: " + text[:200])
