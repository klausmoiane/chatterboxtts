import sys
import os
import random
import numpy as np
import torch
from pathlib import Path

# Path setup
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals
from chatterbox.text_preprocessor import split_into_chunks
from voice_templates.preset_manager import load_preset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    """Define seeds para reprodutibilidade."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def crossfade_concat(wavs: list[np.ndarray], sr: int = 24000, crossfade_ms: int = 50) -> np.ndarray:
    """
    Concatena múltiplos arrays de áudio 1D aplicando crossfade linear para evitar cliques ou quebras bruscas.
    """
    if not wavs:
        return np.array([], dtype=np.float32)
    if len(wavs) == 1:
        return wavs[0]

    fade_len = int(sr * (crossfade_ms / 1000.0))
    result = list(wavs[0])

    for next_wav in wavs[1:]:
        if len(result) < fade_len or len(next_wav) < fade_len:
            result.extend(next_wav)
            continue

        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        overlap = np.array(result[-fade_len:]) * fade_out + next_wav[:fade_len] * fade_in
        result[-fade_len:] = overlap.tolist()
        result.extend(next_wav[fade_len:])

    return np.array(result, dtype=np.float32)


class NarratorPTPipeline:
    """
    Pipeline especializado para narração longa e natural em Língua Portuguesa
    usando ChatterboxMultilingualTTS com estratégias de chunking seguro (<= 280-300 chars)
    e concatenação sem pausas indesejadas.
    """
    def __init__(self):
        self.model = None
        self.sr = 24000

    def get_model(self):
        """Carregamento lazy sob demanda."""
        if self.model is None:
            print(f"[NarratorPTPipeline] Carregando Chatterbox Multilingual no dispositivo: {DEVICE}...")
            self.model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
            self.sr = self.model.sr
            print(f"[NarratorPTPipeline] Modelo pronto (sr={self.sr}).")
        return self.model

    def _load_conds_for_generation(self, preset: dict | None, audio_prompt_path: str | None, exaggeration: float = 0.35):
        """
        Carrega condicionais uma única vez:
        1. conds_pt no preset (embeddings pré-computados, 0s overhead)
        2. audio_prompt_path explícito
        3. audio_ref do preset
        """
        m = self.get_model()

        conds_pt = preset.get("conds_pt") if preset else None
        if conds_pt and os.path.exists(conds_pt):
            print(f"[NarratorPTPipeline] Carregando conds pré-computadas: {conds_pt}")
            map_loc = DEVICE if torch.cuda.is_available() else "cpu"
            m.conds = Conditionals.load(conds_pt, map_location=map_loc).to(DEVICE)
            return

        ref_path = audio_prompt_path or (preset.get("audio_ref") if preset else None)
        if ref_path and os.path.exists(ref_path):
            print(f"[NarratorPTPipeline] Calculando condicionais a partir de: {ref_path}")
            m.prepare_conditionals(ref_path, exaggeration=exaggeration)
            return

        if m.conds is None:
            raise ValueError("Nenhum áudio de referência ou conds_pt disponível para o narrador PT.")

    def generate_narration(
        self,
        text: str,
        preset: dict | None = None,
        audio_prompt_path: str | None = None,
        language_id: str = "pt",
        exaggeration: float | None = None,
        temperature: float | None = None,
        cfg_weight: float | None = None,
        base_seed: int | None = None,
        max_chars_per_chunk: int | None = None,
        progress_callback=None
    ) -> tuple[int, np.ndarray]:
        """
        Gera narração completa de texto longo dividindo em chunks semânticos com limite seguro.
        """
        params = preset.get("params", {}) if preset else {}
        exagg = exaggeration if exaggeration is not None else params.get("exaggeration", 0.35)
        temp = temperature if temperature is not None else params.get("temperature", 0.70)
        cfgw = cfg_weight if cfg_weight is not None else params.get("cfg_weight", 0.65)
        seed = base_seed if base_seed is not None else params.get("seed", 42)
        max_c = max_chars_per_chunk if max_chars_per_chunk is not None else params.get("max_chars_per_chunk", 280)
        lang = language_id or preset.get("language_id", "pt")

        m = self.get_model()

        # 1. Chunking seguro por limites de pontuação e normalização fonética
        chunks = split_into_chunks(text, max_chars=int(max_c), language_id=lang)
        if not chunks:
            raise ValueError("Texto vazio ou nenhum chunk válido gerado.")

        total_chunks = len(chunks)
        print(f"[NarratorPTPipeline] Iniciando narração PT. Total chunks: {total_chunks}")

        # 2. Carregar condicionais de voz
        self._load_conds_for_generation(preset, audio_prompt_path, exaggeration=exagg)

        generated_wavs = []

        # 3. Gerar cada chunk
        for idx, chunk_text in enumerate(chunks):
            current_seed = seed + idx if seed != 0 else 0
            if current_seed != 0:
                set_seed(current_seed)

            if progress_callback:
                progress_callback(idx + 1, total_chunks, chunk_text)

            print(f"[Chunk PT {idx + 1}/{total_chunks}] (seed={current_seed}, chars={len(chunk_text)}): '{chunk_text[:40]}...'")

            wav_tensor = m.generate(
                text=chunk_text,
                language_id=lang,
                exaggeration=exagg,
                cfg_weight=cfgw,
                temperature=temp,
            )

            wav_np = wav_tensor.squeeze(0).cpu().numpy()
            generated_wavs.append(wav_np)

        # 4. Concatenação fluida via crossfade
        full_wav = crossfade_concat(generated_wavs, sr=m.sr, crossfade_ms=50)

        return (m.sr, full_wav.astype(np.float32))
