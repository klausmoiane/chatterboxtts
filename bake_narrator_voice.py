"""
bake_narrator_voice.py
======================
Script one-shot para "treinar" o personagem Narrator EN (Documentary)
com o segmento 02:43–03:12 do ficheiro MP3 especificado.

Fluxo:
  1. Extrai segmento 2:43–3:12 do MP3 → WAV temporário
  2. Carrega o modelo Chatterbox Turbo (model_weights/)
  3. Executa prepare_conditionals() no segmento extraído
  4. Guarda os embeddings em voice_templates/refs/Narrator_EN_Documentary.pt
  5. Actualiza voice_templates/narrator_en_documentary.json com "conds_pt"
"""

import sys
import os
import json
import tempfile
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

# Path setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# ─── PARÂMETROS ──────────────────────────────────────────────────────────────
MP3_PATH    = r"C:\Users\klaus\OneDrive\Desktop\DEV\SIDE PROJECTS\YOUTUBE\tts\Why Putin Is So Hard to Kill.mp3"
START_SEC   = 2 * 60 + 43   # 02:43 = 163 segundos
END_SEC     = 3 * 60 + 12   # 03:12 = 192 segundos
PRESET_NAME = "Narrator EN (Documentary)"
PRESET_FILE = ROOT / "voice_templates" / "narrator_en_documentary.json"
REFS_DIR    = ROOT / "voice_templates" / "refs"
CONDS_OUT   = REFS_DIR / "Narrator_EN_Documentary.pt"

REFS_DIR.mkdir(parents=True, exist_ok=True)

# ─── STEP 1: Extrair segmento ────────────────────────────────────────────────
print(f"\n[1/4] A carregar MP3: {MP3_PATH}")
print(f"      Segmento: {START_SEC}s – {END_SEC}s  ({END_SEC - START_SEC}s de duração)")

# Carregar o segmento directamente (librosa suporta offset/duration)
wav_segment, sr = librosa.load(
    MP3_PATH,
    sr=None,          # manter sample rate original
    offset=START_SEC,
    duration=(END_SEC - START_SEC),
    mono=True,
)

print(f"      Sample rate original: {sr} Hz")
print(f"      Amostras carregadas: {len(wav_segment)} ({len(wav_segment)/sr:.2f}s)")

# Guardar como WAV temporário para prepare_conditionals (aceita ficheiros, não arrays)
tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
sf.write(tmp_wav.name, wav_segment, sr)
print(f"      WAV temporário: {tmp_wav.name}")

# ─── STEP 2: Carregar modelo ─────────────────────────────────────────────────
print("\n[2/4] A carregar modelo Chatterbox Turbo...")
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      Dispositivo: {DEVICE}")

from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
model = ChatterboxTurboTTS.from_pretrained(DEVICE)
print("      Modelo carregado.")

# ─── STEP 3: Processar condicionais de voz ───────────────────────────────────
print(f"\n[3/4] A executar prepare_conditionals() no segmento extraído...")
model.prepare_conditionals(tmp_wav.name, norm_loudness=True)
print("      Condicionais computadas com sucesso.")

# Limpar WAV temporário
os.unlink(tmp_wav.name)

# ─── STEP 4: Guardar .pt ─────────────────────────────────────────────────────
print(f"\n[4/4] A guardar embeddings em: {CONDS_OUT}")
model.conds.save(CONDS_OUT)
size_kb = CONDS_OUT.stat().st_size // 1024
print(f"      Ficheiro guardado: {size_kb} KB")

# ─── STEP 5: Actualizar preset JSON ──────────────────────────────────────────
print(f"\n[5/5] A actualizar preset JSON: {PRESET_FILE}")

if PRESET_FILE.exists():
    with open(PRESET_FILE, "r", encoding="utf-8") as f:
        preset_data = json.load(f)
else:
    preset_data = {
        "name": PRESET_NAME,
        "description": "Voz do narrador Why Putin Is So Hard to Kill — segmento 2:43-3:12",
        "audio_ref": None,
        "params": {
            "temperature": 1.0,
            "repetition_penalty": 1.05,
            "top_p": 0.85,
            "min_p": 0.0,
            "top_k": 500,
            "seed": 42,
            "norm_loudness": True,
            "n_cfm_timesteps": 4,
            "max_chars_per_chunk": 250,
        },
    }

# Actualizar o campo conds_pt
preset_data["conds_pt"] = str(CONDS_OUT)
preset_data["description"] = (
    "Narrador documentário — voz clonada do episódio 'Why Putin Is So Hard to Kill' "
    f"(segmento {START_SEC//60:02d}:{START_SEC%60:02d}–{END_SEC//60:02d}:{END_SEC%60:02d}). "
    "Embeddings pré-computados em conds_pt."
)

with open(PRESET_FILE, "w", encoding="utf-8") as f:
    json.dump(preset_data, f, indent=2, ensure_ascii=False)

print(f"      JSON actualizado com conds_pt={CONDS_OUT}")

# ─── RESULTADO ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ✅  TREINO CONCLUÍDO COM SUCESSO")
print("="*60)
print(f"  Personagem : {PRESET_NAME}")
print(f"  Embeddings : {CONDS_OUT} ({size_kb} KB)")
print(f"  Preset JSON: {PRESET_FILE}")
print()
print("  Da próxima vez que gerar narração com este personagem,")
print("  as conds são carregadas instantaneamente (0s de espera).")
print("  O MP3 original NÃO é necessário para gerações futuras.")
print("="*60)
