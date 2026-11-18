import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
input_txt_file = ROOT_DIR / "input_text.txt"

if input_txt_file.exists():
    with open(input_txt_file, "r", encoding="utf-8") as f:
        english_text = f.read().strip()
else:
    english_text = (
        "Ethiopia banned the import of combustion-engine cars in 2024. "
        "Thanks to cheap hydroelectric power and Chinese-made vehicles, "
        "electric vehicles (EVs) already account for 6% of the country's fleet. "
        "Challenges include high prices and instability."
    )

notebook = {
    "metadata": {
        "kernelspec": {
            "language": "python",
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [],
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": True
        }
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Chatterbox TTS - 100% GPU com Clonagem da Voz Real do Personagem EN\n",
                "Referência: Episódio 'Why Putin Is So Hard to Kill' (segmento documental 02:43-03:12 anexado via dataset)."
            ]
        },
        {
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
                "    print(f'VRAM Total: {vram_gb:.2f} GB')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                "!git clone --depth 1 https://github.com/resemble-ai/perth.git /tmp/perth\n",
                "!pip install --no-deps /tmp/perth\n",
                "!git clone --depth 1 https://github.com/resemble-ai/chatterbox.git /tmp/chatterbox\n",
                "!pip install --no-deps /tmp/chatterbox\n",
                "!pip install s3tokenizer conformer diffusers==0.29.0 pyloudnorm soundfile\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                "import os\n",
                "import glob\n",
                "\n",
                "# Localizar o áudio anexado pelo Dataset\n",
                "dataset_files = glob.glob('/kaggle/input/**/reference_narrator_en.wav', recursive=True)\n",
                "if dataset_files:\n",
                "    ref_audio_path = dataset_files[0]\n",
                "    print(f'Áudio de referência do personagem encontrado: {ref_audio_path}')\n",
                "else:\n",
                "    raise FileNotFoundError('Áudio de referência reference_narrator_en.wav não encontrado no /kaggle/input!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                "import sys\n",
                "import time\n",
                "import torch\n",
                "import soundfile as sf\n",
                "import perth\n",
                "from chatterbox.tts import ChatterboxTTS\n",
                "\n",
                "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "print(f'Carregando modelo ChatterboxTTS na GPU ({DEVICE})...')\n",
                "model = ChatterboxTTS.from_pretrained(DEVICE)\n",
                "\n",
                "print('Aplicando clonagem com a voz documental real...')\n",
                "model.prepare_conditionals(ref_audio_path)\n",
                "print('Voz do personagem carregada e aplicada com sucesso!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                f'text_input = \"\"\"{english_text}\"\"\"\n',
                "print('=== TEXTO A SER SINTETIZADO COM A VOZ REAL DO PERSONAGEM ===')\n",
                "print(text_input)\n",
                "\n",
                "print('\\nIniciando geração de áudio com a voz real...')\n",
                "gen_start = time.time()\n",
                "wav = model.generate(\n",
                "    text=text_input,\n",
                "    temperature=0.70,\n",
                "    cfg_weight=0.65\n",
                ")\n",
                "gen_time = time.time() - gen_start\n",
                "\n",
                "wav_np = wav.squeeze(0).cpu().numpy()\n",
                "audio_duration = len(wav_np) / model.sr\n",
                "rtf = gen_time / audio_duration if audio_duration > 0 else 0\n",
                "\n",
                "output_filename = 'narracao_personagem_en_gpu.wav'\n",
                "sf.write(output_filename, wav_np, model.sr)\n",
                "\n",
                "print('\\n=============================================================')\n",
                "print('=== SÍNTESE CONCLUÍDA COM A VOZ DO PERSONAGEM DOCUMENTAL ===')\n",
                "print('=============================================================')\n",
                "print(f'Voz Utilizada: Personagem Documental EN (Why Putin Is So Hard to Kill)')\n",
                "print(f'Duração do Áudio Gerado: {audio_duration:.2f} segundos')\n",
                "print(f'Tempo de Geração GPU: {gen_time:.2f} segundos')\n",
                "print(f'Fator de Tempo Real (RTF): {rtf:.3f}x')\n",
                "print(f'Arquivo de saída salvo: {os.path.abspath(output_filename)}')\n",
                "print('=============================================================')\n"
            ]
        }
    ]
}

output_nb_path = ROOT_DIR / "local-gpu.ipynb"
with open(output_nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)
print(f"[OK] Notebook gerado com sucesso em {output_nb_path}")
