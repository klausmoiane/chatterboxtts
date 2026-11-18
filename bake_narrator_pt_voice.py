"""
bake_narrator_pt_voice.py
=========================
Script one-shot para pré-computar condicionais (bake) do personagem
'Narrator PT (Documentary)' usando o áudio de referência 'voice_preview_filipe -.mp3'
com o modelo ChatterboxMultilingualTTS.

Fluxo:
  1. Carrega o modelo Chatterbox Multilingual
  2. Executa prepare_conditionals() no áudio filipe
  3. Salva os embeddings em voice_templates/refs/Narrator_PT_Documentary.pt
  4. Atualiza voice_templates/narrator_pt_documentary.json apontando para o .pt
"""

import sys
import os
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

AUDIO_REF = ROOT / "voice_preview_filipe -.mp3"
PRESET_NAME = "Narrator PT (Documentary)"
PRESET_FILE = ROOT / "voice_templates" / "narrator_pt_documentary.json"
REFS_DIR = ROOT / "voice_templates" / "refs"
CONDS_OUT = REFS_DIR / "Narrator_PT_Documentary.pt"

REFS_DIR.mkdir(parents=True, exist_ok=True)

def bake():
    if not AUDIO_REF.exists():
        raise FileNotFoundError(f"Áudio de referência não encontrado: {AUDIO_REF}")

    print(f"\n[1/4] Áudio de referência encontrado: {AUDIO_REF}")
    size_audio = AUDIO_REF.stat().st_size // 1024
    print(f"      Tamanho: {size_audio} KB")

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[2/4] Carregando ChatterboxMultilingualTTS no dispositivo: {DEVICE}...")
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
    print("      Modelo carregado.")

    print(f"\n[3/4] Executando prepare_conditionals() com áudio filipe...")
    # exaggeration=0.35 para voz natural e estável
    model.prepare_conditionals(str(AUDIO_REF), exaggeration=0.35)
    print("      Condicionais calculadas com sucesso.")

    print(f"\n[4/4] Salvando embeddings em: {CONDS_OUT}")
    model.conds.save(CONDS_OUT)
    size_kb = CONDS_OUT.stat().st_size // 1024
    print(f"      Arquivo .pt gravado: {size_kb} KB")

    if PRESET_FILE.exists():
        with open(PRESET_FILE, "r", encoding="utf-8") as f:
            preset_data = json.load(f)
    else:
        preset_data = {
            "name": PRESET_NAME,
            "description": "Narrador documentário em português — voz clonada de 'voice_preview_filipe -.mp3'.",
            "audio_ref": str(AUDIO_REF),
            "conds_pt": None,
            "language_id": "pt",
            "params": {
                "exaggeration": 0.35,
                "temperature": 0.70,
                "cfg_weight": 0.65,
                "seed": 42,
                "max_chars_per_chunk": 280
            }
        }

    preset_data["conds_pt"] = str(CONDS_OUT)
    with open(PRESET_FILE, "w", encoding="utf-8") as f:
        json.dump(preset_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Preset JSON atualizado com conds_pt: {PRESET_FILE}")
    print("=" * 60)
    print("  ✅ BAKE NARRADOR PT CONCLUÍDO COM SUCESSO")
    print(f"  Personagem : {PRESET_NAME}")
    print(f"  Embeddings : {CONDS_OUT} ({size_kb} KB)")
    print("=" * 60)

if __name__ == "__main__":
    bake()
