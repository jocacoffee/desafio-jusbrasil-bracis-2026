# -*- coding: utf-8 -*-
"""
Normalização de identificadores numéricos e texto ruidoso (nível 2).

Regra de ouro do desafio (ver PDF §2, "Garantia do ruído"): um dígito nunca
é trocado por outro dígito. Isso significa que a sequência de dígitos crua
de uma citação real, uma vez extraídos apenas os caracteres [0-9], é sempre
idêntica à sequência de dígitos do registro correspondente na base canônica
-- não importa como o número foi pontuado, quebrado em linhas ou abreviado.

Por isso a estratégia de resolução (ver resolve.py) NÃO depende de
reconstruir a formatação exata (CNJ vs. sequencial) para bater com o
tokenizador do FTS5: em vez disso, extrai-se a string crua de dígitos de
cada candidato e compara-se diretamente contra um índice pré-computado de
"número próprio" de cada um dos 1.018 registros da base (ver db.py). Isso
é mais simples e mais robusto a ruído do que reconstruir tokens.
"""
import re
import unicodedata

# Confusões de OCR mencionadas no PDF (letras, não dígitos):
# 0<->O, 1<->l, 5<->S, m<->rn. Usadas para normalizar NOMES DE CÓDIGO e
# palavras-chave de classe processual antes de casar contra dicionários --
# nunca aplicadas a sequências que já foram identificadas como dígitos.
_OCR_LETTER_FIXES = [
    (re.compile(r"rn"), "m"),
]


def strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_word(s: str) -> str:
    """Normaliza uma palavra/frase para comparação tolerante a ruído:
    minúsculas, sem acento, espaços colapsados, common OCR letter-swaps
    corrigidos. Não mexe em dígitos."""
    s = strip_accents(s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    for pat, repl in _OCR_LETTER_FIXES:
        s = pat.sub(repl, s)
    return s


def raw_digits(s: str) -> str:
    """Extrai só os dígitos de uma string (remove pontuação, espaços,
    quebras de linha, NBSP, etc. -- tudo que não for [0-9])."""
    return re.sub(r"[^\d]", "", s)


# letras visualmente confundidas com dígitos por OCR (PDF §2: 0<->O,
# 1<->l, 5<->S -- pares com semelhança visual clara e mapeamento
# confiável). "g" (minúsculo) -> "9" confirmado contra o registro real
# ("R.Esp. n° 1.45g.779-MA" resolve a "RECURSO ESPECIAL Nº 1.459.779 -
# MA" na base canônica, id 2106305320) -- uma suspeita inicial de que
# fosse "8" não se sustentou (era só uma leitura errada da mesma
# citação, nunca um segundo caso confirmado). "G" maiúsculo não tem
# nenhuma ocorrência confirmável (só aparece numa citação `inventada`,
# sem registro-alvo para checar), então fica de fora do mapeamento de
# VALOR -- mas entra em _OCR_DIGIT_LETTERS (extract.py) para o span ao
# menos ser reconhecido e estendido corretamente; sem mapeamento, a
# letra é descartada na hora de montar os dígitos (raw_digits), o que
# ainda assim resolve certo por não casar com nenhum registro real.
# Usado só do lado da CITAÇÃO (nunca ao indexar a base canônica, que não
# tem esse ruído) e só quando a letra está encaixada entre dígitos --
# ver extract.py:_extend_through_ocr_digit.
_OCR_DIGIT_MAP = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1",
                                 "S": "5", "s": "5", "g": "9"})


def digits_with_ocr_fix(s: str) -> str:
    """Como raw_digits, mas primeiro reescreve letras confundíveis com
    dígitos (O/l/S) para o dígito correspondente antes de descartar o
    resto. Só deve ser chamada sobre um trecho já identificado como
    número (não sobre texto livre)."""
    return re.sub(r"[^\d]", "", s.translate(_OCR_DIGIT_MAP))


def looks_like_identifier_noise(s: str) -> bool:
    """True se a string é majoritariamente dígitos + separadores comuns
    (., -, /, espaço, °, º, quebra de linha, NBSP) -- sinal de que é um
    número de processo/recurso, não texto substantivo."""
    stripped = re.sub(r"[\d\.\-/°º\s ]", "", s)
    return len(stripped) == 0
