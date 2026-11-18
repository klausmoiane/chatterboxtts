"""
Script de teste para verificar se o dropdown está sendo carregado corretamente
"""
from voice_templates.preset_manager import get_preset_names

print("Testing Gradio dropdown initialization...")
print("=" * 50)

# Simular o que acontece quando o Gradio é iniciado
initial_choices = []  # Dropdown inicializa vazio
print(f"Initial dropdown choices: {initial_choices}")
print(f"Is empty: {len(initial_choices) == 0}")
print()

# Simular o demo.load() - o que deveria acontecer
def load_initial_presets():
    names = get_preset_names()
    print(f"Loading presets in demo.load(): {names}")
    first_name = names[0] if names else None
    print(f"First preset to select: {first_name}")
    return names, first_name

# Executar a função
new_choices, selected = load_initial_presets()
print()
print(f"✅ After load - Dropdown should have: {new_choices}")
print(f"✅ After load - Selected value: {selected}")
