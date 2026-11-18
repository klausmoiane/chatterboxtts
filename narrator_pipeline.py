import sys
import os
import random
import numpy as np
import torch
import pyloudnorm as ln

# Path setup so imports work regardless of working directory
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
from chatterbox.text_preprocessor import normalize_text, split_into_chunks, estimate_generation
from voice_templates.preset_manager import REFS_DIR, save_preset

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
    Concatena múltiplos arrays de áudio 1D aplicando crossfade linear para evitar cliques.
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


class NarratorPipeline:
    def __init__(self):
        self.model = None
        self.sr = 24000

    def get_model(self):
        """Carrega o modelo lazy sob demanda."""
        if self.model is None:
            print(f"[NarratorPipeline] Carregando Chatterbox Turbo no dispositivo: {DEVICE}...")
            self.model = ChatterboxTurboTTS.from_pretrained(DEVICE)
            self.sr = self.model.sr
            print("[NarratorPipeline] Modelo pronto.")
        return self.model

    def bake_voice(
        self,
        audio_path: str,
        preset_name: str,
        description: str = "",
        params: dict | None = None,
        progress_callback=None,
    ) -> dict:
        """
        Processa o áudio de referência UMA ÚNICA VEZ, guarda as conds em .pt,
        e actualiza o preset JSON para apontar ao ficheiro .pt.

        Fluxo:
          audio_path → prepare_conditionals() → Conditionals.save(.pt) → preset JSON actualizado
        """
        import json
        from pathlib import Path

        m = self.get_model()

        if progress_callback:
            progress_callback("A processar áudio de referência...")

        print(f"[bake_voice] A processar: {audio_path}")
        m.prepare_conditionals(audio_path, norm_loudness=True)

        # Guardar .pt em voice_templates/refs/
        safe_name = "".join(c for c in preset_name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
        conds_pt_path = REFS_DIR / f"{safe_name}.pt"
        m.conds.save(conds_pt_path)
        print(f"[bake_voice] Conds guardadas em: {conds_pt_path} ({conds_pt_path.stat().st_size // 1024} KB)")

        if progress_callback:
            progress_callback("A salvar preset...")

        default_params = {
            "temperature": 1.0,
            "repetition_penalty": 1.05,
            "top_p": 0.85,
            "min_p": 0.0,
            "top_k": 500,
            "seed": 42,
            "n_cfm_timesteps": 4,
            "max_chars_per_chunk": 250,
        }
        if params:
            default_params.update(params)

        preset = save_preset(
            name=preset_name,
            description=description,
            params=default_params,
            audio_ref_path=None,         # não guardar o MP3 original
            conds_pt_path=str(conds_pt_path),
        )

        if progress_callback:
            progress_callback(f"✅ Personagem '{preset_name}' treinado e salvo!")

        return preset

    def _load_conds_for_generation(self, preset: dict | None, audio_prompt_path: str | None, norm_loudness: bool):
        """
        Carrega as condicionais de voz uma única vez antes de gerar todos os chunks.

        Prioridade:
          1. conds_pt no preset (embeddings pré-computados, 0s overhead)
          2. audio_ref no preset com audio_ref_start_sec/end_sec (segmento específico)
          3. audio_ref no preset sem segmento (ficheiro completo)
          4. audio_prompt_path passado directamente
          5. voz padrão do modelo (model_weights/conds.pt)
        """
        import tempfile
        import librosa
        import soundfile as sf

        m = self.get_model()

        # 1. conds.pt pré-computado
        conds_pt = preset.get("conds_pt") if preset else None
        if conds_pt and os.path.exists(conds_pt):
            print(f"[NarratorPipeline] Carregando conds pré-baked: {conds_pt}")
            map_loc = DEVICE if torch.cuda.is_available() else "cpu"
            m.conds = Conditionals.load(conds_pt, map_location=map_loc).to(DEVICE)
            print("[NarratorPipeline] Conds carregadas instantaneamente.")
            return

        # 2+3. audio_ref do preset (com ou sem segmento)
        preset_audio_ref = preset.get("audio_ref") if preset else None
        if preset_audio_ref and os.path.exists(preset_audio_ref):
            start_sec = preset.get("audio_ref_start_sec")
            end_sec   = preset.get("audio_ref_end_sec")

            if start_sec is not None and end_sec is not None:
                # Extrair segmento específico para WAV temporário
                duration = end_sec - start_sec
                print(f"[NarratorPipeline] Extraindo segmento {start_sec}s–{end_sec}s de: {os.path.basename(preset_audio_ref)}")
                wav_seg, sr = librosa.load(preset_audio_ref, sr=None, offset=float(start_sec), duration=float(duration), mono=True)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, wav_seg, sr)
                tmp.close()
                ref_path = tmp.name
                cleanup_tmp = True
            else:
                ref_path   = preset_audio_ref
                cleanup_tmp = False

            print(f"[NarratorPipeline] prepare_conditionals: {ref_path}")
            try:
                m.prepare_conditionals(ref_path, norm_loudness=norm_loudness)
            finally:
                if cleanup_tmp:
                    try:
                        os.unlink(ref_path)
                    except Exception:
                        pass  # Windows pode ter o ficheiro bloqueado brevemente
            return

        # 4. audio_prompt_path directo (UI upload)
        if audio_prompt_path and os.path.exists(audio_prompt_path):
            print(f"[NarratorPipeline] prepare_conditionals (upload directo): {audio_prompt_path}")
            m.prepare_conditionals(audio_prompt_path, norm_loudness=norm_loudness)
            return

        # 5. Voz padrão do modelo
        if m.conds is None:
            raise ValueError("Nenhum áudio de referência, conds.pt ou voz padrão disponível.")
        print("[NarratorPipeline] Usando voz padrão do modelo.")

    def generate_narration(
        self,
        text: str,
        preset: dict | None = None,
        audio_prompt_path: str | None = None,
        temperature: float = 1.0,
        repetition_penalty: float = 1.05,
        top_p: float = 0.85,
        min_p: float = 0.0,
        top_k: int = 500,
        base_seed: int = 42,
        n_cfm_timesteps: int = 4,
        max_chars_per_chunk: int = 250,
        norm_loudness: bool = True,
        progress_callback=None
    ) -> tuple[int, np.ndarray]:
        """
        Gera a narração para um texto longo processando em chunks com consistência de voz.
        Se o preset tiver conds_pt, carrega os embeddings instantaneamente.
        """
        m = self.get_model()

        # 1. Normalização e Chunking
        chunks = split_into_chunks(text, max_chars=max_chars_per_chunk)
        if not chunks:
            raise ValueError("O texto fornecido está vazio ou não gerou chunks válidos.")

        total_chunks = len(chunks)
        print(f"[NarratorPipeline] Iniciando narração. Total de chunks: {total_chunks}")

        # 2. Carregar condicionais de voz UMA ÚNICA VEZ
        self._load_conds_for_generation(preset, audio_prompt_path, norm_loudness)

        generated_wavs = []

        # 3. Gerar chunk a chunk com Seed por Offset
        for idx, chunk_text in enumerate(chunks):
            current_seed = base_seed + idx if base_seed != 0 else 0
            if current_seed != 0:
                set_seed(current_seed)

            if progress_callback:
                progress_callback(idx + 1, total_chunks, chunk_text)

            print(f"[Chunk {idx + 1}/{total_chunks}] (seed={current_seed}): '{chunk_text[:40]}...'")

            wav_tensor = m.generate(
                text=chunk_text,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_p=top_p,
                min_p=min_p,
                top_k=int(top_k),
                audio_prompt_path=None,   # conds já carregadas!
                norm_loudness=norm_loudness,
                n_cfm_timesteps=int(n_cfm_timesteps)
            )

            wav_np = wav_tensor.squeeze(0).cpu().numpy()
            generated_wavs.append(wav_np)

        # 4. Concatenação com Crossfade
        full_wav = crossfade_concat(generated_wavs, sr=m.sr, crossfade_ms=50)

        # 5. Normalização de Loudness Global
        if norm_loudness and len(full_wav) > 0:
            try:
                meter = ln.Meter(m.sr)
                loudness = meter.integrated_loudness(full_wav)
                gain_db = -27.0 - loudness
                gain_linear = 10.0 ** (gain_db / 20.0)
                if np.isfinite(gain_linear) and gain_linear > 0:
                    full_wav = full_wav * np.float32(gain_linear)
            except Exception as e:
                print(f"Aviso na normalização global de loudness: {e}")

        # Garantir dtype float32
        return (m.sr, full_wav.astype(np.float32))
