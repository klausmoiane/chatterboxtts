"""
make_notebook_noticias_en.py
============================
Fase 1.2 — Gera local-gpu.ipynb com:
  - Leitura do roteiro de kaggle/input/chatterbox-voice-refs/input_text.txt
  - Divisão por pontuacao respeitando limite seguro de 220 chars
  - Geração iterativa em GPU com voz de referência montada
  - Concatenação acústica com crossfade de 50ms entre chunks
  - Progress bar: [Chunk X/N] (Y% concluído)
  - Exportação final para /kaggle/working/narracao_noticias_en_gpu.wav

NOTA: O áudio de referência NÃO é embutido em Base64 no notebook.
      Ele é lido diretamente do dataset montado:
      /kaggle/input/chatterbox-voice-refs/reference_narrator_en.wav
      O dataset deve estar declarado em kernel-metadata.json > dataset_sources.
"""

import json
from pathlib import Path

# Sem embutir áudio — o dataset já está declarado nos metadados do kernel

# ---------------------------------------------------------------------------
# Células do notebook
# ---------------------------------------------------------------------------

cell_gpu_check = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "import torch\n",
        "print('=== CUDA / GPU STATUS ===')\n",
        "print('CUDA Available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    print('GPU Device:', torch.cuda.get_device_name(0))\n",
        "    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n",
        "    print(f'VRAM Total: {vram_gb:.2f} GB')\n",
    ],
}

cell_install = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# Instalação limpa das dependências Chatterbox Turbo + áudio\n",
        "!git clone --depth 1 https://github.com/resemble-ai/perth.git /tmp/perth\n",
        "!pip install --no-deps /tmp/perth\n",
        "!git clone --depth 1 https://github.com/resemble-ai/chatterbox.git /tmp/chatterbox\n",
        "!pip install --no-deps /tmp/chatterbox\n",
        "!pip install s3tokenizer conformer diffusers==0.29.0 noisereduce transformers pyloudnorm soundfile numpy\n",
    ],
}

cell_restore_ref = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "import os\n",
        "\n",
        "# Referência de voz lida direto do dataset montado no Kaggle\n",
        "REF_AUDIO_PATH = '/kaggle/input/chatterbox-voice-refs/reference_narrator_en.wav'\n",
        "\n",
        "if not os.path.exists(REF_AUDIO_PATH):\n",
        "    raise FileNotFoundError(\n",
        "        f'Referencia de voz nao encontrada: {REF_AUDIO_PATH}\\n'\n",
        "        'Verifique se o dataset klausmultiuso/chatterbox-voice-refs '\n",
        "        'esta declarado em kernel-metadata.json > dataset_sources.'\n",
        "    )\n",
        "\n",
        "print(f'Referencia de voz OK: {REF_AUDIO_PATH}')\n",
        "print(f'Tamanho: {os.path.getsize(REF_AUDIO_PATH) / 1024:.1f} KB')\n",
    ],
}

cell_preprocess_ref = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "import soundfile as sf\n",
        "import numpy as np\n",
        "import librosa\n",
        "import pyloudnorm as ln\n",
        "import noisereduce as nr\n",
        "\n",
        "# 2.2 Carregamento do áudio de referência com librosa.load(sr=24000, mono=True)\n",
        "print('=== PRÉ-PROCESSAMENTO DO ÁUDIO DE REFERÊNCIA ===')\n",
        "ref_wav, ref_sr = librosa.load(REF_AUDIO_PATH, sr=24000, mono=True)\n",
        "orig_duration = len(ref_wav) / ref_sr\n",
        "\n",
        "# Medir loudness original\n",
        "meter = ln.Meter(ref_sr)\n",
        "orig_loudness = meter.integrated_loudness(ref_wav)\n",
        "print(f'Referência Original: {orig_duration:.2f}s | Loudness Original: {orig_loudness:.2f} LUFS')\n",
        "\n",
        "# 2.3 Redução de ruído espectral usando os primeiros 0.5s como amostra de ruído\n",
        "noise_sample = ref_wav[:int(ref_sr * 0.5)]\n",
        "ref_wav = nr.reduce_noise(y=ref_wav, sr=ref_sr, y_noise=noise_sample, prop_decrease=0.8)\n",
        "\n",
        "# 2.4 Normalização de loudness para -23 LUFS (broadcast standard)\n",
        "meter_post = ln.Meter(ref_sr)\n",
        "loudness_cur = meter_post.integrated_loudness(ref_wav)\n",
        "if np.isfinite(loudness_cur):\n",
        "    gain_linear = 10.0 ** ((-23.0 - loudness_cur) / 20.0)\n",
        "    ref_wav = ref_wav * np.float32(gain_linear)\n",
        "\n",
        "# 2.5 Soft clipping (np.clip(-0.95, 0.95)) para evitar saturação\n",
        "ref_wav = np.clip(ref_wav, -0.95, 0.95).astype(np.float32)\n",
        "\n",
        "# 2.6 Salvar resultado em /kaggle/working/ref_clean.wav e redirecionar REF_AUDIO_PATH\n",
        "CLEAN_REF_PATH = '/kaggle/working/ref_clean.wav'\n",
        "sf.write(CLEAN_REF_PATH, ref_wav, ref_sr, subtype='FLOAT')\n",
        "REF_AUDIO_PATH = CLEAN_REF_PATH\n",
        "\n",
        "# 2.7 Prints de diagnóstico (duração, loudness antes/depois)\n",
        "final_loudness = meter_post.integrated_loudness(ref_wav)\n",
        "print(f'Referência Limpa Salva: {CLEAN_REF_PATH}')\n",
        "print(f'Duração Final: {len(ref_wav)/ref_sr:.2f}s | Loudness: {orig_loudness:.2f} LUFS -> {final_loudness:.2f} LUFS')\n",
        "print('Pré-processamento de áudio concluído com sucesso!')\n",
    ],
}

cell_load_text = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# Leitura do roteiro — dataset montado tem prioridade\n",
        "DATASET_TEXT = '/kaggle/input/chatterbox-voice-refs/input_text.txt'\n",
        "FALLBACK_TEXT = '/kaggle/working/input_text_inline.txt'\n",
        "\n",
        "if os.path.exists(DATASET_TEXT):\n",
        "    with open(DATASET_TEXT, 'r', encoding='utf-8') as f:\n",
        "        full_text = f.read().strip()\n",
        "    print(f'Roteiro carregado do dataset ({len(full_text)} chars).')\n",
        "else:\n",
        "    # Texto inline como fallback (definido abaixo)\n",
        "    full_text = INLINE_TEXT.strip()\n",
        "    print(f'Roteiro carregado inline ({len(full_text)} chars).')\n",
        "\n",
        "print(f'Primeiros 200 chars:\\n{full_text[:200]}...')\n",
    ],
}

cell_inline_text = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# Texto inline (fallback caso dataset nao esteja montado)\n",
        "INLINE_TEXT = \"\"\"\\\n",
        "On the night of Monday, November 2, just hours before polls opened in the United States, \\\n",
        "Donald Trump was asked whether he would bomb Iran again after the elections. \\\n",
        "He answered with a single word: no.\\\n",
        "\"\"\"\n",
    ],
}

# A célula de inline text com o roteiro completo é gerada dinamicamente a partir do arquivo
with open("kaggle_runner/input_text.txt", "r", encoding="utf-8") as f:
    roteiro_completo = f.read().strip()

# Escapar aspas triplas para segurança no notebook
roteiro_escaped = roteiro_completo.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

cell_inline_text_real = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "import os\n",
        "\n",
        f'INLINE_TEXT = """{roteiro_escaped}"""\n',
        "\n",
        "# Roteiro final (dataset tem prioridade; inline é fallback)\n",
        "DATASET_TEXT = '/kaggle/input/chatterbox-voice-refs/input_text.txt'\n",
        "if os.path.exists(DATASET_TEXT):\n",
        "    with open(DATASET_TEXT, 'r', encoding='utf-8') as _f:\n",
        "        full_text = _f.read().strip()\n",
        "    print(f'Roteiro carregado do dataset: {len(full_text)} chars')\n",
        "else:\n",
        "    full_text = INLINE_TEXT.strip()\n",
        "    print(f'Roteiro carregado inline: {len(full_text)} chars')\n",
        "\n",
        "print(f'Preview: {full_text[:180]}...')\n",
    ],
}

cell_chunking = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "import re\n",
        "import textwrap\n",
        "\n",
        "MAX_CHARS = 250  # Limite seguro sentence-aware\n",
        "\n",
        "def normalize_text(text: str) -> str:\n",
        "    \"\"\"Normalização básica de pontuação e caracteres especiais.\"\"\"\n",
        "    if not text:\n",
        "        return ''\n",
        "    text = re.sub(r'\\[\\s*(music|__|[a-zA-Z0-9_]+)\\s*\\]', '', text, flags=re.IGNORECASE)\n",
        "    text = text.replace('“', '\"').replace('”', '\"').replace('‘', \"'\").replace('’', \"'\")\n",
        "    text = text.replace('—', ', ').replace('–', ', ').replace('...', ', ')\n",
        "    return text.strip()\n",
        "\n",
        "def split_into_chunks(text: str, max_chars: int = MAX_CHARS) -> list:\n",
        "    \"\"\"\n",
        "    Divide o texto em chunks respeitando limites de frases (Sentence-Aware Chunking - Estratégia B).\n",
        "    1. Divide o texto por parágrafos (\\n\\n).\n",
        "    2. Divide parágrafos em frases por pontuação (. ! ?).\n",
        "    3. Agrupa frases completas até atingir max_chars.\n",
        "    4. Subdivide por vírgulas apenas se uma frase isolada exceder max_chars.\n",
        "    \"\"\"\n",
        "    text = normalize_text(text)\n",
        "    if not text:\n",
        "        return []\n",
        "\n",
        "    paragraphs = [p.strip() for p in text.split('\\n\\n') if p.strip()]\n",
        "    final_chunks = []\n",
        "\n",
        "    for para in paragraphs:\n",
        "        # Divisão por frases mantendo a pontuação terminal (. ! ?)\n",
        "        sentences = re.split(r'(?<=[.!?])\\s+', para)\n",
        "        current_chunk = ''\n",
        "\n",
        "        for sentence in sentences:\n",
        "            sentence = sentence.strip()\n",
        "            if not sentence:\n",
        "                continue\n",
        "\n",
        "            if len(sentence) > max_chars:\n",
        "                if current_chunk:\n",
        "                    final_chunks.append(current_chunk.strip())\n",
        "                    current_chunk = ''\n",
        "\n",
        "                # Subdivide frase longa por vírgulas\n",
        "                sub_parts = re.split(r'(?<=,)\\s+', sentence)\n",
        "                sub_chunk = ''\n",
        "                for part in sub_parts:\n",
        "                    if len(sub_chunk) + len(part) + 1 <= max_chars:\n",
        "                        sub_chunk = f'{sub_chunk} {part}'.strip()\n",
        "                    else:\n",
        "                        if sub_chunk:\n",
        "                            final_chunks.append(sub_chunk.strip())\n",
        "                        if len(part) > max_chars:\n",
        "                            wrapped = textwrap.wrap(part, width=max_chars)\n",
        "                            final_chunks.extend(wrapped)\n",
        "                            sub_chunk = ''\n",
        "                        else:\n",
        "                            sub_chunk = part\n",
        "                if sub_chunk:\n",
        "                    final_chunks.append(sub_chunk.strip())\n",
        "            else:\n",
        "                # Agrupa frases completas no mesmo chunk respeitando max_chars\n",
        "                if len(current_chunk) + len(sentence) + 1 <= max_chars:\n",
        "                    current_chunk = f'{current_chunk} {sentence}'.strip()\n",
        "                else:\n",
        "                    final_chunks.append(current_chunk.strip())\n",
        "                    current_chunk = sentence\n",
        "\n",
        "        if current_chunk:\n",
        "            final_chunks.append(current_chunk.strip())\n",
        "\n",
        "    return [c.strip() for c in final_chunks if c.strip()]\n",
        "\n",
        "\n",
        "chunks = split_into_chunks(full_text)\n",
        "print(f'Total de chunks gerados (Estratégia B): {len(chunks)}')\n",
        "print('-' * 60)\n",
        "for i, ch in enumerate(chunks):\n",
        "    print(f'[{i+1:02d}] ({len(ch)} chars) {ch[:80]}...' if len(ch) > 80 else f'[{i+1:02d}] ({len(ch)} chars) {ch}')\n",
    ],
}

cell_load_model = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "import sys, time, torch, soundfile as sf\n",
        "import numpy as np\n",
        "import perth\n",
        "from chatterbox.tts_turbo import ChatterboxTurboTTS\n",
        "\n",
        "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "print(f'Carregando Chatterbox Turbo em: {DEVICE}...')\n",
        "load_start = time.time()\n",
        "# Patch ChatterboxTurboTTS.norm_loudness to prevent NumPy 2.x float promotion to double\n",
        "_orig_norm_loudness = ChatterboxTurboTTS.norm_loudness\n",
        "def _safe_norm_loudness(self, wav, sr, target_lufs=-27):\n",
        "    res = _orig_norm_loudness(self, wav, sr, target_lufs=target_lufs)\n",
        "    return res.astype(np.float32)\n",
        "ChatterboxTurboTTS.norm_loudness = _safe_norm_loudness\n",
        "\n",
        "model = ChatterboxTurboTTS.from_pretrained(DEVICE)\n",
        "print(f'Modelo Chatterbox Turbo carregado em {time.time() - load_start:.2f}s!')\n",
        "\n",
        "print(f'Aplicando condicionais de voz (áudio pré-processado a -23 LUFS): {REF_AUDIO_PATH}')\n",
        "model.prepare_conditionals(REF_AUDIO_PATH, norm_loudness=False)\n",
        "print('Condicionais de voz aplicados com sucesso no Chatterbox Turbo!')\n",
    ],
}

cell_generate_loop = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# GERAÇÃO SEQUENCIAL COM CHATTERBOX TURBO (CROSSFADE 100ms)\n",
        "# ============================================================\n",
        "import random\n",
        "CROSSFADE_MS = 100       # ms de crossfade entre chunks\n",
        "OUTPUT_PATH  = '/kaggle/working/narracao_noticias_en_turbo_gpu.wav'\n",
        "BASE_SEED    = 42\n",
        "\n",
        "total_chunks  = len(chunks)\n",
        "audio_segments = []\n",
        "total_gen_time = 0.0\n",
        "\n",
        "print(f'Iniciando síntese de {total_chunks} chunks no Chatterbox Turbo na GPU ({DEVICE})...')\n",
        "print('=' * 60)\n",
        "\n",
        "pipeline_start = time.time()\n",
        "\n",
        "for idx, chunk_text in enumerate(chunks):\n",
        "    chunk_num = idx + 1\n",
        "    pct = int(chunk_num / total_chunks * 100)\n",
        "    bar_filled = int(pct / 5)  # barra de 20 chars\n",
        "    bar = '#' * bar_filled + '-' * (20 - bar_filled)\n",
        "    print(f'[Chunk {chunk_num:02d}/{total_chunks}] [{bar}] {pct}% | {len(chunk_text)} chars')\n",
        "    sys.stdout.flush()\n",
        "\n",
        "    # 5.4 e 5.5 Seed determinístico por chunk (torch, random, numpy)\n",
        "    chunk_seed = BASE_SEED + idx\n",
        "    torch.manual_seed(chunk_seed)\n",
        "    if torch.cuda.is_available():\n",
        "        torch.cuda.manual_seed(chunk_seed)\n",
        "    random.seed(chunk_seed)\n",
        "    np.random.seed(chunk_seed)\n",
        "\n",
        "    t0 = time.time()\n",
        "    wav = model.generate(\n",
        "        text=chunk_text,\n",
        "        temperature=0.8,\n",
        "        top_p=0.95,\n",
        "        min_p=0.05,\n",
        "        repetition_penalty=1.2,\n",
        "        cfg_weight=0.5,\n",
        "    )\n",
        "    gen_t = time.time() - t0\n",
        "    total_gen_time += gen_t\n",
        "\n",
        "    wav_np = wav.squeeze(0).cpu().numpy()  # shape: (samples,)\n",
        "    audio_segments.append(wav_np)\n",
        "    print(f'           └─ gerado em {gen_t:.2f}s ({len(wav_np)/model.sr:.2f}s de áudio | seed={chunk_seed})')\n",
        "\n",
        "print('=' * 60)\n",
        "print(f'Todos os chunks sintetizados! Tempo total de geração: {total_gen_time:.2f}s')\n",
    ],
}

cell_crossfade = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {"trusted": True},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# CONCATENAÇÃO COM CROSSFADE DE 100ms\n",
        "# ============================================================\n",
        "SR = model.sr\n",
        "crossfade_samples = int(SR * CROSSFADE_MS / 1000)  # 100ms → amostras\n",
        "\n",
        "def crossfade_concat(segments: list, cf_samples: int) -> np.ndarray:\n",
        "    \"\"\"Concatena segmentos com crossfade linear suave.\"\"\"\n",
        "    if len(segments) == 0:\n",
        "        return np.array([], dtype=np.float32)\n",
        "    if len(segments) == 1:\n",
        "        return segments[0].astype(np.float32)\n",
        "\n",
        "    result = segments[0].astype(np.float32)\n",
        "\n",
        "    for seg in segments[1:]:\n",
        "        seg = seg.astype(np.float32)\n",
        "        cf = min(cf_samples, len(result), len(seg))\n",
        "\n",
        "        if cf <= 0:\n",
        "            result = np.concatenate([result, seg])\n",
        "            continue\n",
        "\n",
        "        # Janelas de fade-out e fade-in lineares\n",
        "        fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)\n",
        "        fade_in  = np.linspace(0.0, 1.0, cf, dtype=np.float32)\n",
        "\n",
        "        # Mistura na zona de sobreposição\n",
        "        overlap = result[-cf:] * fade_out + seg[:cf] * fade_in\n",
        "\n",
        "        result = np.concatenate([result[:-cf], overlap, seg[cf:]])\n",
        "\n",
        "    return result\n",
        "\n",
        "\n",
        "print(f'Concatenando {len(audio_segments)} segmentos com crossfade de {CROSSFADE_MS}ms ({crossfade_samples} amostras)...')\n",
        "final_audio = crossfade_concat(audio_segments, crossfade_samples)\n",
        "\n",
        "total_duration = len(final_audio) / SR\n",
        "rtf = total_gen_time / total_duration if total_duration > 0 else 0\n",
        "pipeline_total = time.time() - pipeline_start\n",
        "\n",
        "# Exportar\n",
        "sf.write(OUTPUT_PATH, final_audio, SR)\n",
        "\n",
        "print()\n",
        "print('=' * 62)\n",
        "print('=== SÍNTESE COMPLETA — CHATTERBOX TURBO GPU EN ===')\n",
        "print('=' * 62)\n",
        "print(f'Chunks gerados:          {total_chunks}')\n",
        "print(f'Crossfade aplicado:      {CROSSFADE_MS}ms entre cada chunk')\n",
        "print(f'Duração total do áudio:  {total_duration:.2f}s ({total_duration/60:.1f} min)')\n",
        "print(f'Tempo de geração GPU:    {total_gen_time:.2f}s')\n",
        "print(f'Tempo total pipeline:    {pipeline_total:.2f}s')\n",
        "print(f'RTF (Tempo Real Factor): {rtf:.3f}x')\n",
        "print(f'Aceleração GPU:          {1.0/rtf:.1f}x mais rápido que tempo real' if rtf > 0 else '')\n",
        "print(f'Arquivo exportado:       {OUTPUT_PATH}')\n",
        "print('=' * 62)\n",
    ],
}

# ---------------------------------------------------------------------------
# Montar o notebook
# ---------------------------------------------------------------------------
notebook = {
    "metadata": {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.12"},
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [
                {
                    "sourceType": "datasetVersion",
                    "sourceId": "chatterbox-voice-refs",
                    "mountPath": "/kaggle/input/chatterbox-voice-refs",
                }
            ],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True,
        },
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Pipeline de Narração Longa — Chatterbox Turbo GPU Kaggle (Tesla T4)\n",
                "**Fase 1.2** — Chunking Sentence-Aware + Crossfade de 100ms + Clonagem de Voz Turbo\n\n",
                "- Divisão inteligente por pontuação (≤250 chars/chunk, Estratégia B)\n",
                "- Geração iterativa determinística com seeds por chunk\n",
                "- Concatenação com crossfade linear suave de 100ms\n",
                "- Saída: `/kaggle/working/narracao_noticias_en_turbo_gpu.wav`\n",
            ],
        },
        cell_gpu_check,
        cell_install,
        cell_restore_ref,
        cell_preprocess_ref,
        cell_inline_text_real,
        cell_chunking,
        cell_load_model,
        cell_generate_loop,
        cell_crossfade,
    ],
}

# ---------------------------------------------------------------------------
# Gravar o notebook
# ---------------------------------------------------------------------------
output_path = "kaggle_runner/local-gpu.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[OK] Notebook gerado com sucesso: {output_path}")
print(f"   Roteiro: {len(roteiro_completo)} chars")
print(f"   Audio de referencia: lido do dataset no Kaggle (sem embutir Base64)")
