# -*- coding: utf-8 -*-
"""
Biblioteca de padrões (regex + dicionários) usada tanto para:
  (a) extrair citações dos documentos de entrada (extract.py), e
  (b) indexar o "número próprio" de cada um dos 1.018 registros da base
      canônica (db.py) -- mesma família de regex, dois usos.

Os padrões foram catalogados a partir dos 225 exemplos do goldenset.csv
(dev set) e generalizados para tolerar o ruído descrito no PDF: variantes
de abreviação, formatação de número, separador de UF, confusões de OCR e
quebras de linha no meio do identificador. Nenhum padrão assume um valor
específico de citação -- todos são famílias de forma.
"""
import re

# ---------------------------------------------------------------------
# 1) Palavras-chave de classe/recurso processual (jurisprudência)
# ---------------------------------------------------------------------
# Cada item é uma variante de forma (abreviada, ponto, espaço) já
# tolerante a ruído. A ordem importa pouco: o regex final usa alternação
# e greedy matching para capturar a cadeia mais longa de tokens.

_CLASSE_ATOMS = [
    r"Embargos de Declara[cç][aã]o",
    r"EDcl", r"ED",
    r"Embargos de Diverg[eê]ncia",
    r"Agravo Regimental", r"AgRg", r"AGRG", r"AG\.?\s*REG\.?",
    r"Agravo Interno", r"AgInt", r"AGINT", r"Ag\.?\s*Int\.?",
    r"Agravo em Recurso Especial", r"AREsp", r"ARESP", r"A\.?REsp",
    r"AgREsp", r"AGREsp",
    r"Agravo em Recurso Extraordin[aá]rio", r"ARE", r"AgREx", r"AG\.REX",
    r"Agravo de Instrumento", r"AI", r"AgR-AI", r"AG\.?\s*INT",
    r"Agravo",
    r"Recurso Especial Eleitoral", r"REspe\.?", r"RESPE",
    r"AgR-REspe", r"AREspE[Il]",  # "I"/"l" -- confusão de OCR bem documentada
    r"Recurso Especial", r"REsp", r"RESP", r"R\.?\s*Esp\.?", r"Rec\.?\s*Esp\.?",
    r"Recurso Extraordin[aá]rio", r"RE", r"REX",
    r"Recurso em Habeas Corpus", r"RHC", r"RHAB",
    r"Recurso Ordin[aá]rio", r"RO",
    r"Recurso de Revista", r"RR",
    r"Recurso em Mandado de Seguran[cç]a", r"RMS",
    r"Reclama[cç][aã]o", r"Recl\.?", r"Rcl", r"RCL",
    r"Apela[cç][aã]o", r"APL",
    r"Recurso em Sentido Estrito", r"RSE",
    r"Habeas Corpus", r"HC", r"H\.?C\.?",
    r"A[cç][aã]o Rescis[oó]ria", r"AR", r"AC\.\s*RES\.?",
    r"Mandado de Seguran[cç]a", r"MS",
    r"Suspens[aã]o de Liminar e de Senten[cç]a", r"SLS", r"SL",
    r"Ag\.?\s*Rg", r"AgARR", r"ARR",
    r"Agravo de Instrumento em Recurso de Revista", r"AIRR",
    r"AgI/RR", r"AG\.I/RR",
    r"Terceiro AG\.?\s*REG",
    r"R-Rp", r"Recurso na Representa[cç][aã]o", r"Representa[cç][aã]o", r"RRep",
]
# átomos vindos de uma auditoria de cobertura (LLM local pedindo
# abreviações oficiais/correntes por família, não usadas na extração de
# conteúdo -- só vocabulário) contra os 26 documentos de dev, que não
# necessariamente mostram todas as variantes de abreviação em uso real.
# Cada um checado manualmente por plausibilidade antes de entrar; os que
# pareciam genéricos demais (ex.: "Ap" para Apelação, "REE" p/ Especial
# Eleitoral) ficaram de fora por risco de falso positivo.
# ordena por tamanho decrescente para o regex preferir o token mais
# específico/longo antes do mais curto (evita "RE" engolir "REsp").
_CLASSE_ATOMS.sort(key=len, reverse=True)
# espaços literais dentro dos átomos viram \s+ para tolerar quebra de
# linha no meio da expressão de classe (ex.: "Recurso\nEspecial"). Os
# lookarounds evitam que uma sigla curta (AR, RE, AI...) case dentro de
# uma palavra comum (ex.: "complementAR", "REcorrente"). O qualificador
# opcional cobre variantes como "Apelação CRIMINAL", "Agravo em Recurso
# Especial ELEITORAL Nº ..." sem precisar enumerar cada combinação de
# classe x qualificador como átomo separado.
_NOT_LETTER = r"[^A-Za-zÀ-ÿ]"
_QUALIFICADOR = (
    r"(?:\s+(?:Eleitoral|Criminal|C[íi]vel|Trabalhista|Militar|"
    r"Previdenci[áa]rio|Tribut[áa]rio|Ordin[áa]ria))?"
)
CLASSE_ATOM_RE = r"(?:" + (
    r"(?:(?:(?<=" + _NOT_LETTER + r")|^)(?:"
    + "|".join(a.replace(" ", r"\s+") for a in _CLASSE_ATOMS)
    + r")" + _QUALIFICADOR + r"(?=" + _NOT_LETTER + r"|$))"
    # "E" solto só conta como classe (Embargos) em compostos com hífen
    # dos dois lados, ex. "TST-E-RR-", "ED-E-ED-RR-" -- fora desse
    # contexto quase sempre é a conjunção "e" ("...21.545/SP e 21.783/RS").
    + r"|(?:(?<=-)E(?=-))"
) + r")"

_CONNECTOR = r"(?:\s+n?[oa]s?\s+|\s*-\s*)"  # "no", "na", "nos", "-"

# prefixo = uma ou mais classes conectadas ("EDcl no AgInt no REsp")
CLASSE_PREFIX_RE = re.compile(
    r"(?:%s%s)*%s" % (CLASSE_ATOM_RE, _CONNECTOR, CLASSE_ATOM_RE),
    re.IGNORECASE,
)

# marcador "TST-" que antecede uma cadeia de classes já reconhecida por
# CLASSE_PREFIX_RE (ex.: "TST-E-RR-", "TST- ED - E-ED-RR-", com espaços
# tolerados ao redor do hífen).
TST_LEAD_RE = re.compile(r"TST\s*-\s*$", re.IGNORECASE)

# ---------------------------------------------------------------------
# 2) Números -- tolerantes a ruído
# ---------------------------------------------------------------------
# separador "livre" entre dígitos: ponto, hífen, barra, espaço, NBSP,
# quebra de linha, graus.
_SEP = r"[\.\-/\s °ºo]{0,3}"
_DIGIT_GROUP = r"\d" + _SEP  # um dígito seguido de zero ou mais separadores

# número CNJ tolerante: >=13 dígitos totais espalhados em até ~30
# caracteres (aceita ausência parcial de pontuação, quebras de linha).
CNJ_NUM_RE = re.compile(
    r"\d(?:" + _SEP + r"\d){12,19}",
)

# número sequencial (recurso "antigo"): 1 a 8 dígitos, com separadores
# tolerantes; usado quando não há 13+ dígitos seguidos. Mínimo de 1 dígito
# (não 2) para não perder números onde o SEGUNDO dígito virou letra
# confundível por OCR logo de cara (ex.: "6G.838" -- extract.py estende
# através da letra depois, mas só se houver ALGO daqui pra prosseguir).
# Números soltos de verdade não viram citação de qualquer forma: exigem
# prefixo de classe processual reconhecido (ver extract.py).
SEQ_NUM_RE = re.compile(
    r"\d(?:" + _SEP + r"\d){0,7}",
)

# sufixo de UF/tribunal depois do número: "/PR", "-PR", "(PR)", "- RJ"
UF_SUFFIX_RE = re.compile(
    r"\s*[/\-–—(]\s*([A-Z]{2})\s*\)?",
)

# ---------------------------------------------------------------------
# 3) Súmulas
# ---------------------------------------------------------------------
# tolera "m"->"rn" (par de OCR documentado no PDF) em "Súmula" e dígitos
# do número escritos como letra confundível (0<->O, 1<->l, 5<->S) --
# convertidos de volta em extract.py via normalize.digits_with_ocr_fix.
# O grupo do número fica em (?-i:...): sem essa trava, o IGNORECASE do
# resto do regex também vira as letras da classe [OoIlSs] case-insensitive
# (então "I" no padrão passa a casar "i" minúsculo, "L" maiúsculo etc.),
# o que caçava palavras comuns como "sumida"/"sumiu" (súm+i) por engano.
SUMULA_RE = re.compile(
    r"[S5]{1}[uú](?:m|rn)(?:ula)?\.?\s*(Vinculante)?\s*n?[ºo°.]?\s*"
    r"(?-i:([\dOoIlSs]+))\s*(?:d[oa]\s*)?"
    r"(STF|STJ|TST|TSE|STM)?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------
# 4) Dispositivos de lei
# ---------------------------------------------------------------------
CODIGO_SYNONYMS = {
    "cpc": ["cpc", "codigo de processo civil"],
    "cdc": ["cdc", "codigo de defesa do consumidor"],
    "clt": ["clt", "consolidacao das leis do trabalho"],
    "cf": ["cf", "cf/88", "constituicao federal", "constituicao da republica",
           "constituicao"],
    "cpm": ["cpm", "codigo penal militar"],
    "cpp": ["cpp", "codigo de processo penal"],
    "cc": ["cc", "codigo civil"],
    "codigo_eleitoral": ["codigo eleitoral"],
    "lc64_1990": ["lc 64/1990", "lc 64", "lei complementar 64/1990",
                  "lei complementar no 64/1990", "lei complementar n 64/1990",
                  "lei complementar 64", "lei no 64/1990"],
    "lei_13105_2015": ["lei 13.105/2015", "lei no 13105/2015",
                       "lei 13105/2015", "lei n 13.105/2015"],
}
# lei_13105_2015 é o número oficial da lei do CPC -- tratado como alias de "cpc".

# O "clausulado" de incisos/parágrafos é capturado de forma genérica, e o
# nome do código também (grupo final, até vírgula/ponto -- tolera quebra
# de linha no meio do nome do código, ex.: "Código\nde Processo Penal") --
# a identificação de QUAL código foi citado usa fuzzy match tolerante a
# ruído OCR letra-a-letra (ver resolve.py:_extract_codigo_key), já que o
# nível 2 do desafio troca letras dentro do próprio nome do código
# ("Constituição Fedcral", "Código de Processo Cívil" etc.) e uma
# alternação exata de sinônimos (CODIGO_NAME_ALTERNATION) não cobre isso.
ARTIGO_RE = re.compile(
    r"art(?:igo)?\.?\s*(\d{1,3}(?:\.\d{3})*)[º°o]?"
    r"(?:\s*[-,]\s*(?:§\s*\d+[º°o]?(?:-[A-Z])?|[IVXLC]+|['\"][a-zA-Z]['\"]))*"
    r"\s*(?:,\s*)?(?:d[oa]|de)\s+([^,.]{2,45})",
    re.IGNORECASE,
)

ARTIGO_NUM_RE = re.compile(r"art(?:igo)?\.?\s*(\d{1,4})", re.IGNORECASE)

# mapeamento número do artigo -> código, construído inspecionando os 13
# registros natureza='dispositivo' da base canônica (cada número é único
# nesta cobertura congelada). Ver hunter/db.py:ARTIGO_TO_CODIGO_ID para a
# ligação com o id real de cada registro.
ARTIGO_TO_CODIGO = {
    276: "codigo_eleitoral",
    290: "cpm",
    14: "cdc",
    93: "cf",
    896: "clt",
    7: "cf",
    5: "cf",
    818: "clt",
    312: "cpp",
    477: "clt",
    186: "cc",
    1: "lc64_1990",
    373: "cpc",
}

# ---------------------------------------------------------------------
# 5) Menções vagas / incompletas (sem identificador ou com identificador
#    insuficiente para individualizar um único feito)
# ---------------------------------------------------------------------
# Casamento difuso (fuzzy.fuzzy_find_all), não regex exato -- tolera tanto
# ruído OCR letra-a-letra (ex.: "entendirnento") quanto paráfrase (troca de
# palavras mantendo o sentido). A lista original (9+6 frases) cobria só o
# vocabulário literal do dev set; foi ampliada a partir de um experimento
# controlado (ver experiment/results/report.md) que gerou paráfrases
# sintéticas dessas mesmas frases com uma LLM local e mediu quais o
# fuzzy match antigo não pegava -- os agrupamentos temáticos abaixo (p.ex.
# "entendimento pacificado/consolidado/sedimentado desta Corte") vieram
# direto desse diagnóstico. Cada frase nova é só mais uma âncora dentro do
# mesmo mecanismo de match; como a precisão do vocabulário fechado nesse
# experimento foi sempre 1,000 (nunca gerou falso positivo mesmo variando
# o texto ao redor), ampliar a lista só pode aumentar recall, não custa
# precisão.
VAGUE_PHRASES_TEXT = [
    "jurisprudência pacífica desta Corte",
    "entendimento sumulado sobre a matéria",
    "verbete sumular aplicável à espécie",
    "orientação jurisprudencial da Corte Superior",
    "precedentes desta Casa em situações análogas",
    "reiterados precedentes do Superior Tribunal de Justiça",
    "precedente firmado em sede de recurso repetitivo",
    "jurisprudência consolidada dos tribunais superiores",
    "recente acórdão da Segunda Turma",
    # entendimento pacificado/consolidado/sedimentado por um tribunal
    "entendimento pacificado deste Tribunal",
    "posição consolidada desta Corte",
    "entendimento consolidado nas Cortes Superiores",
    "linha decisória consolidada da Corte",
    "orientação jurisprudencial deste Tribunal Superior",
    # súmula sem número (verbete/tese/entendimento sumulado aplicável)
    "tese fixada em súmula aplicável ao caso",
    "verbete sumular pertinente à lide",
    "entendimento cristalizado em súmula aplicável",
    # precedentes/casos análogos
    "precedentes desta Corte em casos análogos",
    "entendimento firmado em situações semelhantes",
    # recurso repetitivo
    "tese firmada em julgamento de recursos repetitivos",
    # reordenação da mesma frase (varia posição do "recente"/tribunal)
    "acórdão recente da Segunda Turma",
]

LEI_VAGUE_PHRASES_TEXT = [
    "normas de regência da matéria",
    "dispositivo constitucional invocado na origem",
    "dispositivo legal de regência",
    "legislação de regência da matéria",
    "artigo correspondente do Código de Processo Civil",
    "lei que disciplina a prescrição no caso",
    "normas aplicáveis ao tema em análise",
    "dispositivos legais pertinentes à questão",
    "preceito constitucional invocado na origem",
    "norma constitucional suscitada na petição inicial",
    "dispositivo do Código de Processo Civil pertinente ao caso",
    "ordenamento jurídico aplicável à questão",
]

# "Tema/Tese N da repercussão geral/recursos repetitivos" -- a base não
# modela "temas" como registro (só acordao/sumula/dispositivo), logo
# NUNCA resolve: é sempre inventada por construção.
TEMA_RE = re.compile(
    r"Tem[aã]\s*n?[ºo°.]?\s*\d[\d.]*\s*d[ae]\s*(?:repercuss[aã]o\s+geral|recursos?\s+repetitivos?)",
    re.IGNORECASE,
)

# referência buscável mas ambígua: "TRIBUNAL de ANO, relatoria/Rel. de NOME"
# (ver PDF §5, exemplo de "incompleta buscável"). Conectores tolerantes a
# ruído OCR letra-a-letra do nível 2 (ex.: "profcrido", "relatoria dc").
_SEP2 = r"[\s,]*"
_DE = r"d[ce]"  # "de" com possível troca de OCR e->c

# Tudo antes do nome do relator é case-insensitive (?i:...); o nome em si
# fica FORA do (?i:...) de propósito -- ele para sozinho na primeira
# palavra minúscula (ex.: "... SILVA para sustentar"), o que só funciona
# se maiúsculas ainda importarem nessa parte do padrão.
RELATOR_ANO_RE = re.compile(
    r"(?i:(?:julgado|precedente|ac[oó]rd[aã]o|Reclama[cç][aã]o|Recl\.?|Rcl|"
    r"Agravo em Recurso Especial|Recurso em Habeas Corpus|APL)" + _SEP2 +
    r"(?:d[oa]\s+(STF|STJ|TST|TSE|STM))?" + _SEP2 +
    r"(?:" + _DE + r"|prof[ec]rid[oa]\s*em|julgad[oa]\s*em)" + _SEP2 + r"(\d{4})" + _SEP2 +
    r"(?:d[oa]\s+(STF|STJ|TST|TSE|STM))?" + _SEP2 +
    r"(?:pela\s*relatoria\s*" + _DE + r"|da\s*relatoria\s*" + _DE +
    r"|sob\s*relatoria\s*" + _DE + r"|Rel\.?\s*Min\.?)" + _SEP2 + r")" +
    # nome do relator: sequência de palavras capitalizadas/em caixa alta
    r"((?:[A-ZÀ-Ú][A-Za-zÀ-ú]*\.?\s+){0,5}[A-ZÀ-Ú][A-Za-zÀ-ú]*\.?)",
)
