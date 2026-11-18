import json

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
                "# Chatterbox TTS - 100% GPU Execution on Kaggle (NVIDIA Tesla T4)\n",
                "Síntese acelerada por GPU sem dependências conflitantes."
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
                "# Instalação limpa e direta do Perth e dependências de áudio\n",
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
                "import sys\n",
                "import os\n",
                "import time\n",
                "import torch\n",
                "import soundfile as sf\n",
                "\n",
                "# Verificação dos módulos\n",
                "import perth\n",
                "from chatterbox.tts import ChatterboxTTS\n",
                "\n",
                "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                "print(f'Carregando ChatterboxTTS em: {DEVICE}...')\n",
                "load_start = time.time()\n",
                "model = ChatterboxTTS.from_pretrained(DEVICE)\n",
                "print(f'Modelo carregado com sucesso na GPU em {time.time() - load_start:.2f}s!')\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": [
                f'text_input = \"\"\"{english_text}\"\"\"\n',
                "print('=== TEXTO A SER SINTETIZADO ===')\n",
                "print(text_input)\n",
                "\n",
                "print('\\nIniciando síntese de voz 100% via GPU...')\n",
                "gen_start = time.time()\n",
                "wav = model.generate(\n",
                "    text=text_input,\n",
                "    temperature=0.7,\n",
                "    cfg_weight=0.65\n",
                ")\n",
                "gen_time = time.time() - gen_start\n",
                "\n",
                "wav_np = wav.squeeze(0).cpu().numpy()\n",
                "audio_duration = len(wav_np) / model.sr\n",
                "rtf = gen_time / audio_duration if audio_duration > 0 else 0\n",
                "\n",
                "output_filename = 'narracao_english_gpu.wav'\n",
                "sf.write(output_filename, wav_np, model.sr)\n",
                "\n",
                "print('\\n==================================================')\n",
                "print('=== RESULTADO E BENCHMARK DA GPU TESLA T4 KAGGLE ===')\n",
                "print('==================================================')\n",
                "print(f'Texto: {len(text_input)} caracteres / {len(text_input.split())} palavras')\n",
                "print(f'Duração do Áudio Gerado: {audio_duration:.2f} segundos')\n",
                "print(f'Tempo de Geração GPU: {gen_time:.2f} segundos')\n",
                "print(f'Fator de Tempo Real (RTF): {rtf:.3f}x')\n",
                "print(f'Aceleração GPU: {1.0/rtf:.1f}x mais rápido que tempo real!')\n",
                "print(f'Arquivo de saída salvo: {os.path.abspath(output_filename)}')\n",
                "print('==================================================')\n"
            ]
        }
    ]
}

with open('kaggle_runner/local-gpu.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('local-gpu.ipynb corrigido e pronto para execução!')
