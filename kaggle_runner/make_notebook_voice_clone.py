import json
import base64
from pathlib import Path

# Carregar o áudio de referência de 29s do personagem em Base64 para garantir inclusão direta no notebook
ref_audio_path = Path("kaggle_runner/reference_narrator_en.wav")
with open(ref_audio_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("utf-8")

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
                "Referência: Episódio 'Why Putin Is So Hard to Kill' (segmento documental 02:43-03:12)."
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
                "import base64\n",
                f'audio_b64 = \"{audio_b64}\"\n',
                "ref_file = 'narrator_ref.wav'\n",
                "with open(ref_file, 'wb') as f:\n",
                "    f.write(base64.b64decode(audio_b64))\n",
                "print(f'Arquivo de referência de voz restaurado com sucesso: {ref_file}')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                "import sys\n",
                "import os\n",
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
                "print('Aplicando clonagem de voz com referência documental (Why Putin Is So Hard to Kill)...')\n",
                "model.prepare_conditionals('narrator_ref.wav')\n",
                "print('Condicionais de voz e timbre aplicados com sucesso!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                f'text_input = \"\"\"{english_text}\"\"\"\n',
                "print('=== TEXTO A SER NARRADO COM A VOZ REAL DO PERSONAGEM ===')\n",
                "print(text_input)\n",
                "\n",
                "print('\\nIniciando geração acelerada na GPU...')\n",
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

with open('kaggle_runner/local-gpu.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('Notebook com voz embutida gerado com sucesso!')

