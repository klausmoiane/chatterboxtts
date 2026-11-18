"""
Aplicativo minimalista para testar se o dropdown de presets funciona
"""
import gradio as gr
from voice_templates.preset_manager import get_preset_names, load_preset

# CARREGAR PRESETS ANTES DE CRIAR A INTERFACE
preset_names = get_preset_names()
print(f"[INIT] Presets encontrados: {preset_names}")

def on_select_preset(preset_name):
    """Chamado quando o usuário seleciona um preset"""
    print(f"[SELECT] Selecionado: {preset_name}")
    
    if not preset_name:
        return "Nenhum preset selecionado"
    
    data = load_preset(preset_name)
    if not data:
        return f"❌ Erro ao carregar preset '{preset_name}'"
    
    return f"✅ **{preset_name}**\n\n{data.get('description', 'Sem descrição')}"

with gr.Blocks() as demo:
    gr.Markdown("# 🧪 Teste do Dropdown de Presets")
    
    preset_dropdown = gr.Dropdown(
        choices=preset_names,
        value=preset_names[0] if preset_names else None,
        label="Selecione um Preset"
    )
    
    preset_info = gr.Markdown(
        on_select_preset(preset_names[0]) if preset_names else "Nenhum preset encontrado"
    )
    
    # Atualizar info quando o usuário seleciona
    preset_dropdown.change(
        fn=on_select_preset,
        inputs=[preset_dropdown],
        outputs=[preset_info]
    )

if __name__ == "__main__":
    print("Iniciando aplicativo de teste...")
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
