import sys
import os
from pathlib import Path

# Ajustar caminho do projeto
# Configurar encoding UTF-8 para stdout do Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import importlib.util
pt_path = ROOT / "src" / "chatterbox" / "pt_pronunciation.py"
spec = importlib.util.spec_from_file_location("pt_pronunciation", str(pt_path))
pt_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pt_mod)
apply_pt_pronunciation = pt_mod.apply_pt_pronunciation

def test_pt_pronunciation():
    print("=" * 65)
    print("🧪 INICIANDO TESTES LEVES DE PRONÚNCIA EM PORTUGUÊS")
    print("=" * 65)

    cases = [
        # Grupo A: Frase principal completa
        (
            "GTA 6 tem keys e trailer incríveis com Michael e Jason",
            "G. T. A. seis tem keis e tréiler incríveis com Maikel e Jeison"
        ),
        # Grupo B: Siglas e siglas compostas
        ("GTA", "G. T. A."),
        ("TVU", "T. V. U."),
        ("MJ", "M. J."),
        ("UFC", "U. F. C."),
        ("NBA", "N. B. A."),
        ("FIFA", "F. I. F. A."),
        ("PS5", "P. S. cinco"),
        ("Xbox", "Xbox"),  # Preservado
        ("GTA 6", "G. T. A. seis"),
        ("GTA 5", "G. T. A. cinco"),
        # Grupo C: Palavras em inglês / Gaming
        ("keys", "keis"),
        ("trailer", "tréiler"),
        ("vice", "vaise"),
        ("ultimate", "ultimeit"),
        ("gaming", "guéiming"),
        ("YouTube", "Iutuíbe"),
        ("online", "onlaini"),
        ("like", "laique"),
        ("stream", "striim"),
        ("live", "laive"),
        ("joinha", "Joinha"),
        # Grupo D: Nomes próprios e correções
        ("Jason", "Jeison"),
        ("Michael", "Maikel"),
        ("Lionida", "Leonida"),
        ("Leonida", "Leonida"),
        # Grupo F: Não-regressão
        ("O jogo é incrível", "O jogo é incrível"),
        ("Portugal ganhou", "Portugal ganhou"),
    ]

    passed = 0
    failed = 0

    for original, expected in cases:
        actual = apply_pt_pronunciation(original)
        if actual == expected:
            print(f"✅ PASSOU: '{original}' -> '{actual}'")
            passed += 1
        else:
            print(f"❌ FALHOU: '{original}'")
            print(f"    Esperado: '{expected}'")
            print(f"    Obtido:   '{actual}'")
            failed += 1

    print("=" * 65)
    print(f"Resultado: {passed} passaram, {failed} falharam.")
    print("=" * 65)
    return failed == 0

if __name__ == "__main__":
    success = test_pt_pronunciation()
    sys.exit(0 if success else 1)
