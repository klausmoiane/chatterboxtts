import gradio as gr

# Teste SUPER simples
with gr.Blocks() as demo:
    gr.Markdown("# Teste Dropdown")
    
    dropdown = gr.Dropdown(
        choices=["Opção 1", "Opção 2", "Opção 3"],
        value="Opção 1",
        label="Teste"
    )
    
    output = gr.Textbox(label="Você selecionou")
    
    dropdown.change(fn=lambda x: x, inputs=dropdown, outputs=output)

demo.launch(server_port=7862)
