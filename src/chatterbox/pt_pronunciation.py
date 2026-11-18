import re

# Dicionários de substituição com base nas regras de pronúncia e fonética PT

# 1. Siglas compostas / Especiais com números ou sequências
COMPOUND_ACRONYMS = {
    r"\bGTA\s*6\b": "G. T. A. seis",
    r"\bGTA\s*5\b": "G. T. A. cinco",
    r"\bGTA\s*IV\b": "G. T. A. quatro",
    r"\bPS5\b": "P. S. cinco",
    r"\bPS4\b": "P. S. quatro",
}

# 2. Siglas avulsas que devem ser pronunciadas letra a letra
ACRONYMS_PT = {
    r"\bGTA\b": "G. T. A.",
    r"\bTVU\b": "T. V. U.",
    r"\bMJ\b": "M. J.",
    r"\bUFC\b": "U. F. C.",
    r"\bNBA\b": "N. B. A.",
    r"\bFIFA\b": "F. I. F. A.",
}

# 3. Nomes próprios adaptados para pronúncia natural em PT
NAMES_PT = {
    r"\bJason\b": "Jeison",
    r"\bMichael\b": "Maikel",
    r"\bLionida\b": "Leonida",  # Correção ortográfica solicitada
}

# 4. Palavras estrangeiras (Loanwords / Gaming / Tech) adaptadas foneticamente para o leitor PT
LOANWORDS_PT = {
    r"\bkeys\b": "keis",
    r"\bkey\b": "kei",
    r"\btrailer\b": "tréiler",
    r"\btrailers\b": "tréilers",
    r"\bvice\b": "vaise",
    r"\bultimate\b": "ultimeit",
    r"\bgaming\b": "guéiming",
    r"\bYouTube\b": "Iutuíbe",
    r"\byoutube\b": "iutuíbe",
    r"\bonline\b": "onlaini",
    r"\blike\b": "laique",
    r"\blikes\b": "laiques",
    r"\bstream\b": "striim",
    r"\blive\b": "laive",
    r"\blives\b": "laives",
    r"\bjoinha\b": "Joinha",
}

# 5. Dígitos isolados específicos frequentemente lidos em inglês por modelos mistos
ISOLATED_DIGITS = {
    r"(?<=\bGTA\s)6\b": "seis",
    r"(?<=\bGTA\s)5\b": "cinco",
}

def apply_pt_pronunciation(text: str) -> str:
    """
    Aplica regras fonéticas e ortográficas para garantir pronúncia natural em Português.
    Garante que siglas são lidas letra a letra, nomes e termos estrangeiros seguem
    fonética luso-brasileira/gaming, e números são pronunciados em português.
    """
    if not text:
        return ""

    res = text

    # Passo 1: Siglas compostas (ex: GTA 6 -> G. T. A. seis)
    for pattern, replacement in COMPOUND_ACRONYMS.items():
        res = re.sub(pattern, replacement, res, flags=re.IGNORECASE)

    # Passo 2: Nomes próprios
    for pattern, replacement in NAMES_PT.items():
        res = re.sub(pattern, replacement, res, flags=re.IGNORECASE)

    # Passo 3: Siglas individuais (letra a letra)
    for pattern, replacement in ACRONYMS_PT.items():
        res = re.sub(pattern, replacement, res)

    # Passo 4: Termos estrangeiros e gaming
    for pattern, replacement in LOANWORDS_PT.items():
        res = re.sub(pattern, replacement, res, flags=re.IGNORECASE)

    # Passo 5: Casos residuais de dígitos
    for pattern, replacement in ISOLATED_DIGITS.items():
        res = re.sub(pattern, replacement, res)

    return res
