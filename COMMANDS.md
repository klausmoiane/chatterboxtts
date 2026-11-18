# Comandos do Projeto Chatterbox (TTS)

Este arquivo contém os comandos básicos para configurar, rodar e interagir com o projeto Chatterbox utilizando o ambiente virtual local (`.venv`).

---

## 1. Configuração do Ambiente

Se for a primeira vez configurando ou se faltarem dependências no projeto, use os comandos abaixo:

```powershell
# Ativar o ambiente virtual (venv) no Windows (PowerShell)
.venv\Scripts\activate

# Instalar o pacote local em modo editável (-e) junto com todas as dependências (incluindo gradio, torch, etc.)
.venv\Scripts\pip install -e .
```

---

## 2. Como Rodar o Backend / Interface Web (Gradio)

Você pode iniciar as interfaces gráficas web do Chatterbox usando os comandos a seguir. Execute no terminal dentro da pasta raiz do projeto:

### Chatterbox-Turbo (Recomendado para Inglês de baixa latência)
```powershell
# Inicia a interface web focada na versão Turbo (mais leve e rápida)
.venv\Scripts\python gradio_tts_turbo_app.py
```

### Chatterbox Multilingual (Para múltiplos idiomas - 23+)
```powershell
# Inicia a interface web com suporte a múltiplos idiomas, incluindo Português (pt), Espanhol (es), etc.
.venv\Scripts\python multilingual_app.py
```

### Chatterbox TTS Padrão
```powershell
# Inicia a interface padrão do Chatterbox TTS (Inglês)
.venv\Scripts\python gradio_tts_app.py
```

### Chatterbox Voice Conversion
```powershell
# Inicia a interface web para conversão/clonagem de voz
.venv\Scripts\python gradio_vc_app.py
```

---

## 3. Scripts de Exemplo (Execução via Linha de Comando)

Se você preferir rodar scripts que salvam arquivos de áudio diretamente sem usar a interface web:

```powershell
# Executar exemplo básico de TTS
.venv\Scripts\python example_tts.py

# Executar exemplo de TTS com o modelo Turbo
.venv\Scripts\python example_tts_turbo.py

# Executar exemplo de conversão de voz (Voice Conversion)
.venv\Scripts\python example_vc.py
```
