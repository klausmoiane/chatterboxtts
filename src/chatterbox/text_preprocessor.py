import re
import textwrap

NUMBER_MAP = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
    '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
    '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
    '18': 'eighteen', '19': 'nineteen', '20': 'twenty', '30': 'thirty',
    '40': 'forty', '50': 'fifty', '60': 'sixty', '70': 'seventy',
    '80': 'eighty', '90': 'ninety'
}


def normalize_text(text: str, language_id: str = None) -> str:
    """
    Normaliza o texto para melhor síntese de voz (TTS de narração).
    - Limpa tags de áudio desnecessárias ou marcadores de transcrição.
    - Normaliza pontuações e aspas.
    - Aplica regras fonéticas específicas do idioma (ex: Português).
    - Expande algumas siglas e numerações comuns se necessário.
    """
    if not text:
        return ""

    # Remover marcações de transcrição como [Music], [ __ ], [adfern], etc.
    text = re.sub(r'\[\s*(music|__|\w+)\s*\]', '', text, flags=re.IGNORECASE)

    # Regras específicas para Português (PT)
    if language_id and language_id.lower() == 'pt':
        try:
            from .pt_pronunciation import apply_pt_pronunciation
        except (ImportError, ValueError):
            from chatterbox.pt_pronunciation import apply_pt_pronunciation
        text = apply_pt_pronunciation(text)

    # Converter horários como 8:00 p.m. / 8:00pm / 8 p.m. -> eight PM, 8:30 p.m. -> eight 30 PM (apenas se não for PT)
    if not language_id or language_id.lower() != 'pt':
        def _norm_time_match(match):
            h = match.group(1)
            mins = match.group(2)
            meridiem = match.group(3).upper().replace('.', '')
            h_word = NUMBER_MAP.get(h, h)
            if not mins or mins == '00':
                return f"{h_word} {meridiem}"
            return f"{h_word} {mins} {meridiem}"

        # 1. Matches 8:00 p.m., 8:30pm, etc.
        text = re.sub(r'\b(\d{1,2}):(\d{2})\s*(a\.?m\.?|p\.?m\.?)(?!\w)', _norm_time_match, text, flags=re.IGNORECASE)
        # 2. Matches 8 p.m., 8pm, etc.
        text = re.sub(r'\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)(?!\w)', lambda m: f"{NUMBER_MAP.get(m.group(1), m.group(1))} {m.group(2).upper().replace('.', '')}", text, flags=re.IGNORECASE)
        # 3. Normalizar a.m. / p.m. isolados restantes
        text = re.sub(r'\ba\.m\.(?!\w)', 'AM', text, flags=re.IGNORECASE)
        text = re.sub(r'\bp\.m\.(?!\w)', 'PM', text, flags=re.IGNORECASE)

    # Normalizar aspas e travessões
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("—", ", ").replace("–", ", ").replace("...", ", ")

    # Remover múltiplos espaços
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def split_into_chunks(text: str, max_chars: int = 250, language_id: str = None) -> list[str]:
    """
    Divide o texto em chunks respeitando limites de frases (Sentence-Aware Chunking).
    Estratégia B:
    1. Divide o texto por parágrafos (\n\n).
    2. Divide parágrafos em frases por pontuação (. ! ?).
    3. Agrupa frases até atingir max_chars.
    4. Se uma frase isolada for maior que max_chars, sub-divide em vírgulas.
    """
    text = normalize_text(text, language_id=language_id)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    final_chunks = []

    for para in paragraphs:
        # Regex para quebrar em frases mantendo a pontuação final
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Se a frase sozinha ultrapassar o tamanho máximo
            if len(sentence) > max_chars:
                # Se tínhamos algo acumulado no chunk atual, salva antes
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Sub-divisão por vírgulas ou pedaços menores
                sub_parts = re.split(r'(?<=,)\s+', sentence)
                sub_chunk = ""
                for part in sub_parts:
                    if len(sub_chunk) + len(part) + 1 <= max_chars:
                        sub_chunk = f"{sub_chunk} {part}".strip()
                    else:
                        if sub_chunk:
                            final_chunks.append(sub_chunk.strip())
                        # Caso mesmo a parte após a vírgula seja gigante
                        if len(part) > max_chars:
                            wrapped = textwrap.wrap(part, width=max_chars)
                            final_chunks.extend(wrapped)
                            sub_chunk = ""
                        else:
                            sub_chunk = part
                if sub_chunk:
                    final_chunks.append(sub_chunk.strip())

            else:
                # Tenta agrupar com o chunk atual
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk = f"{current_chunk} {sentence}".strip()
                else:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk:
            final_chunks.append(current_chunk.strip())

    return final_chunks


def estimate_generation(chunks: list[str], sec_per_chunk: float = 12.0) -> dict:
    """
    Retorna estimativas de tempo e métricas dos chunks.
    """
    num_chunks = len(chunks)
    total_seconds = num_chunks * sec_per_chunk
    estimated_minutes = round(total_seconds / 60.0, 1)
    total_chars = sum(len(c) for c in chunks)
    avg_chars = round(total_chars / num_chunks, 1) if num_chunks > 0 else 0

    return {
        "num_chunks": num_chunks,
        "estimated_minutes": estimated_minutes,
        "total_chars": total_chars,
        "avg_chars": avg_chars,
    }
