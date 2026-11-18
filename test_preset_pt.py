"""
test_preset_pt.py
=================
Suite de testes rápidos e leves exclusivamente no backend (sem GPU/sem carregar modelo completo)
para validar:
1. Existência e integridade do preset JSON do Narrator PT.
2. Presença dos parâmetros fixos de voz natural.
3. Existência do arquivo de áudio de referência original 'voice_preview_filipe -.mp3'.
4. Integração correta com o preset_manager (load_preset).
5. Estratégia de chunking em português (garantindo que nenhum chunk ultrapasse 280 caracteres).
6. Verificação do estado das condicionais pré-computadas (se o .pt existe após o bake).
"""

import sys
import os
import json
from pathlib import Path
import importlib.util

# Configura encoding UTF-8 no stdout do Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from voice_templates.preset_manager import load_preset, list_presets, get_preset_names

# Importação direta e independente de chatterbox.text_preprocessor
preprocessor_path = ROOT / "src" / "chatterbox" / "text_preprocessor.py"
spec = importlib.util.spec_from_file_location("text_preprocessor", str(preprocessor_path))
text_preprocessor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(text_preprocessor)
split_into_chunks = text_preprocessor.split_into_chunks


def run_tests():
    print("=" * 65)
    print("🧪 INICIANDO TESTES DO PRESET NARRADOR PT (BACKEND)")
    print("=" * 65)

    errors = []

    # TESTE 1: Arquivo JSON existe
    preset_file = ROOT / "voice_templates" / "narrator_pt_documentary.json"
    if not preset_file.exists():
        errors.append("❌ narrator_pt_documentary.json não existe no diretório voice_templates/")
    else:
        print("✅ [TESTE 1] Arquivo narrator_pt_documentary.json encontrado.")

    # TESTE 2: Leitura do JSON e campos obrigatórios
    data = None
    try:
        with open(preset_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("name") == "Narrator PT (Documentary)", f"Nome incorreto: {data.get('name')}"
        assert data.get("language_id") == "pt", f"Language_id incorreto: {data.get('language_id')}"
        print("✅ [TESTE 2] Campos essenciais validados com sucesso (name, language_id='pt').")
    except Exception as e:
        errors.append(f"❌ Falha ao validar estrutura do JSON: {e}")

    # TESTE 3: Parâmetros de voz ativa e gaming
    expected_params = {
        "exaggeration": 0.55,
        "temperature": 0.85,
        "cfg_weight": 0.60,
        "seed": 42,
        "max_chars_per_chunk": 220
    }
    params = data.get("params", {}) if data else {}
    for param_name, expected_val in expected_params.items():
        actual_val = params.get(param_name)
        if actual_val != expected_val:
            errors.append(f"❌ Parâmetro '{param_name}' incorreto: esperado {expected_val}, obtido {actual_val}")
        else:
            print(f"✅ [TESTE 3] Parâmetro '{param_name}' validado ({actual_val}).")

    # TESTE 4: Áudio de referência existe no disco
    audio_ref = data.get("audio_ref") if data else None
    if not audio_ref or not os.path.exists(audio_ref):
        errors.append(f"❌ Áudio de referência não encontrado no disco: {audio_ref}")
    else:
        size_kb = Path(audio_ref).stat().st_size // 1024
        print(f"✅ [TESTE 4] Áudio de referência Filipe existe no disco ({size_kb} KB).")

    # TESTE 5: preset_manager carrega o preset pelo nome
    loaded = load_preset("Narrator PT (Documentary)")
    if not loaded:
        errors.append("❌ preset_manager.load_preset('Narrator PT (Documentary)') retornou None")
    else:
        all_names = get_preset_names()
        assert "Narrator PT (Documentary)" in all_names, "Nome ausente em get_preset_names()"
        print("✅ [TESTE 5] load_preset() e get_preset_names() integrados perfeitamente.")

    # TESTE 6: Verificação da estratégia de chunking (língua portuguesa)
    texto_longo_pt = (
        "No silêncio profundo das montanhas mais isoladas da Europa, onde a névoa esconde "
        "segredos milenares que desafiam a própria passagem do tempo, um pequeno grupo de investigadores "
        "encontrou vestígios inesperados. Cada palavra documentada revela uma narrativa intrigante e precisa. "
        "Não se trata apenas de história antiga; trata-se de um enigma contemporâneo que permaneceu oculto "
        "durante gerações e que agora emerge com uma clareza impressionante para quem souber ouvir atentamente."
    )
    max_limit = params.get("max_chars_per_chunk", 280)
    chunks = split_into_chunks(texto_longo_pt, max_chars=max_limit)

    if not chunks:
        errors.append("❌ split_into_chunks não produziu nenhum chunk.")
    else:
        print(f"✅ [TESTE 6] Texto longo ({len(texto_longo_pt)} caracteres) dividido em {len(chunks)} chunks.")
        for i, c in enumerate(chunks):
            if len(c) > max_limit:
                errors.append(f"❌ Chunk {i+1} excedeu o limite máximo ({len(c)} > {max_limit})")
            print(f"    - Chunk {i+1} ({len(c)} chars): '{c[:45]}...'")

    # TESTE 7: Verificação do arquivo de condicionais (.pt)
    conds_path = data.get("conds_pt") if data else None
    if conds_path and os.path.exists(conds_path):
        size_pt = Path(conds_path).stat().st_size // 1024
        print(f"✅ [TESTE 7] Condicionais pré-computadas (.pt) encontradas ({size_pt} KB): {conds_path}")
    else:
        print(f"ℹ️ [TESTE 7] Arquivo .pt ainda não compilado (necessário rodar bake_narrator_pt_voice.py para compilar).")

    print("=" * 65)
    if errors:
        print("❌ ALGUNS TESTES FALHARAM:")
        for err in errors:
            print(f"  {err}")
        return False
    else:
        print("🎉 TODOS OS TESTES DE BACKEND PASSARAM COM SUCESSO!")
        print("=" * 65)
        return True

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
