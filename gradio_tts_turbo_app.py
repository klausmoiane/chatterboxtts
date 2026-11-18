import os
import random
import tempfile
import soundfile as sf
import numpy as np
import gradio as gr

from chatterbox.text_preprocessor import split_into_chunks, estimate_generation
from voice_templates.preset_manager import (
    list_presets,
    get_preset_names,
    load_preset,
    save_preset,
    delete_preset,
)
from narrator_pipeline import NarratorPipeline

# Instância global do pipeline
PIPELINE = NarratorPipeline()

# --- CSS CUSTOMIZADO ---
CUSTOM_CSS = """
.container { max-width: 1200px; margin: auto; }
.card-btn {
    border: 1px solid #cbd5e1 !important;
    background: #f8fafc !important;
    color: #0f172a !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    text-align: left !important;
    font-weight: 600 !important;
}
.card-btn:hover {
    background: #e2e8f0 !important;
    border-color: #94a3b8 !important;
}
.active-card {
    background: #e0e7ff !important;
    border-color: #6366f1 !important;
    color: #3730a3 !important;
}
.stats-box {
    background: #f1f5f9;
    border-radius: 6px;
    padding: 10px;
    font-size: 13px;
    color: #475569;
}
"""


def build_app():
    with gr.Blocks(title="Chatterbox Narrator Studio") as demo:
        gr.Markdown(
            """
            # 🎙️ Chatterbox Narrator Studio
            Sintetizador de Voz Humana e Narrador para Textos Longos (40+ min) com Presets de Personagens Persistentes.
            """
        )

        # Estados do Gradio
        current_preset_state = gr.State(None)

        with gr.Row():
            # --- COLUNA ESQUERDA: GERENCIADOR DE PERSONAGENS & CONFIGS ---
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Selecionar Personagem")

                preset_dropdown = gr.Dropdown(
                    choices=[],  # Inicializa vazio, será preenchido pelo demo.load
                    label="Vozes / Personagens Salvos"
                )

                preset_desc = gr.Markdown(
                    value="*Selecione um personagem para carregar suas configurações.*"
                )

                ref_audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Áudio de Referência (Opcional - Clone de Voz)",
                    value=None
                )

                with gr.Accordion("💾 Salvar / Criar Novo Personagem", open=False):
                    new_preset_name = gr.Textbox(label="Nome do Personagem", placeholder="Ex: Narrador Misterioso")
                    new_preset_desc = gr.Textbox(label="Descrição", placeholder="Ex: Voz grave e pausada para suspense")
                    save_preset_btn = gr.Button("💾 Salvar Voz Atual como Personagem", variant="secondary")
                    delete_preset_btn = gr.Button("🗑️ Apagar Personagem Selecionado", variant="stop")

                with gr.Accordion("⚙️ Ajustes Finos do Narrador", open=False):
                    seed_num = gr.Number(value=42, label="Base Seed (Offset por chunk)")
                    temp = gr.Slider(0.1, 2.0, step=0.05, label="Temperature (1.0 para mais fluidez humana)", value=1.0)
                    top_p = gr.Slider(0.0, 1.0, step=0.01, label="Top P", value=0.85)
                    top_k = gr.Slider(0, 1000, step=10, label="Top K", value=500)
                    rep_penalty = gr.Slider(1.0, 2.0, step=0.05, label="Repetition Penalty (1.05 evita 'pressa')", value=1.05)
                    min_p = gr.Slider(0.0, 1.0, step=0.01, label="Min P", value=0.0)
                    n_cfm = gr.Slider(1, 10, step=1, label="Timesteps Vocoder (n_cfm_timesteps: 4 recomendado)", value=4)
                    max_chars = gr.Slider(100, 400, step=10, label="Max Chars por Chunk", value=250)
                    norm_loudness = gr.Checkbox(value=True, label="Normalizar Loudness Global (-27 LUFS)")

            # --- COLUNA DIREITA: INPUT DE TEXTO, PROCESSO & OUTPUT ---
            with gr.Column(scale=2):
                gr.Markdown("### 📝 Texto para Narração")

                text_input = gr.Textbox(
                    label="Cole o texto aqui (suporta textos de 40+ minutos sem limite)",
                    placeholder="September 26, 2022. Just before 8:00 p.m., a sailing ship is drifting gently on the Baltic Sea...",
                    lines=12,
                    elem_id="main_text"
                )

                stats_display = gr.Markdown(
                    value="📊 **Estatísticas:** *Digite ou cole um texto para ver os chunks e tempo estimado.*",
                    elem_classes=["stats-box"]
                )

                with gr.Row():
                    preview_btn = gr.Button("👁️ Preview dos Chunks", variant="secondary")
                    generate_btn = gr.Button("⚡ GERAR NARRAÇÃO COMPLETA", variant="primary")

                preview_output = gr.JSON(label="Visualização dos Chunks Gerados", visible=False)

                gr.Markdown("### 🔊 Áudio de Saída")
                audio_output = gr.Audio(label="Narração Concluída", type="filepath")

        # --- EVENTOS E LOGICA ---

        def update_stats(text, max_c):
            if not text or not text.strip():
                return "📊 **Estatísticas:** *Digite ou cole um texto para ver os chunks e tempo estimado.*"
            chunks = split_into_chunks(text, max_chars=int(max_c))
            est = estimate_generation(chunks)
            return (
                f"📊 **Total de Chunks:** `{est['num_chunks']}` | "
                f"⏱️ **Tempo Estimado de Geração:** `~{est['estimated_minutes']} min` | "
                f"📏 **Tamanho Médio:** `{est['avg_chars']} chars/chunk`"
            )

        text_input.change(fn=update_stats, inputs=[text_input, max_chars], outputs=stats_display)
        max_chars.change(fn=update_stats, inputs=[text_input, max_chars], outputs=stats_display)

        def toggle_preview(text, max_c):
            chunks = split_into_chunks(text, max_chars=int(max_c))
            return gr.update(value=chunks, visible=True)

        preview_btn.click(fn=toggle_preview, inputs=[text_input, max_chars], outputs=preview_output)

        def on_select_preset(preset_name):
            print(f"[ON_SELECT] Preset selected: {preset_name}")
            if not preset_name:
                return [gr.update()] * 10

            data = load_preset(preset_name)
            if not data:
                print(f"[ON_SELECT] ERROR: Could not load preset '{preset_name}'")
                return [gr.update()] * 10

            params = data.get("params", {})
            desc = f"**Descrição:** {data.get('description', 'Sem descrição.')}"
            audio_ref = data.get("audio_ref")
            print(f"[ON_SELECT] Loaded preset data successfully")

            return (
                desc,
                audio_ref,
                params.get("seed", 42),
                params.get("temperature", 1.0),
                params.get("top_p", 0.85),
                params.get("top_k", 500),
                params.get("repetition_penalty", 1.05),
                params.get("min_p", 0.0),
                params.get("n_cfm_timesteps", 4),
                params.get("max_chars_per_chunk", 250),
            )

        def load_initial_presets():
            names = get_preset_names()
            print(f"[LOAD_INITIAL] Found {len(names)} presets: {names}")
            first_name = names[0] if names else None
            print(f"[LOAD_INITIAL] First preset to load: {first_name}")
            desc, ref_audio, seed, t, tp, tk, rp, mp, n_cfm_val, max_c = on_select_preset(first_name) if first_name else ("*Nenhum personagem selecionado.*", None, 42, 1.0, 0.85, 500, 1.05, 0.0, 4, 250)
            print(f"[LOAD_INITIAL] Returning choices={names}, value={first_name}")
            return (
                gr.update(choices=names, value=first_name),
                desc,
                ref_audio,
                seed,
                t,
                tp,
                tk,
                rp,
                mp,
                n_cfm_val,
                max_c
            )

        demo.load(
            fn=load_initial_presets,
            inputs=[],
            outputs=[
                preset_dropdown,
                preset_desc,
                ref_audio_input,
                seed_num,
                temp,
                top_p,
                top_k,
                rep_penalty,
                min_p,
                n_cfm,
                max_chars
            ]
        )

        preset_dropdown.change(
            fn=on_select_preset,
            inputs=[preset_dropdown],
            outputs=[
                preset_desc,
                ref_audio_input,
                seed_num,
                temp,
                top_p,
                top_k,
                rep_penalty,
                min_p,
                n_cfm,
                max_chars
            ]
        )

        def handle_save_preset(name, desc, ref_path, seed, t, tp, tk, rp, mp, n_cfm_val, max_c):
            if not name:
                raise gr.Error("Digite um nome para o personagem.")

            params = {
                "seed": int(seed),
                "temperature": float(t),
                "top_p": float(tp),
                "top_k": int(tk),
                "repetition_penalty": float(rp),
                "min_p": float(mp),
                "n_cfm_timesteps": int(n_cfm_val),
                "max_chars_per_chunk": int(max_c),
            }

            save_preset(name, desc, params, ref_path)
            new_choices = get_preset_names()
            return gr.update(choices=new_choices, value=name), f"✅ Personagem **{name}** salvo com sucesso!"

        save_preset_btn.click(
            fn=handle_save_preset,
            inputs=[
                new_preset_name,
                new_preset_desc,
                ref_audio_input,
                seed_num,
                temp,
                top_p,
                top_k,
                rep_penalty,
                min_p,
                n_cfm,
                max_chars
            ],
            outputs=[preset_dropdown, preset_desc]
        )

        def handle_delete_preset(name):
            if not name:
                return gr.update(), "Nenhum personagem selecionado."
            delete_preset(name)
            new_choices = get_preset_names()
            next_val = new_choices[0] if new_choices else None
            return gr.update(choices=new_choices, value=next_val), "🗑️ Personagem removido."

        delete_preset_btn.click(
            fn=handle_delete_preset,
            inputs=[preset_dropdown],
            outputs=[preset_dropdown, preset_desc]
        )

        def run_narration(
            preset_name,
            text,
            audio_ref,
            t,
            rp,
            tp,
            mp,
            tk,
            seed,
            n_cfm_val,
            max_c,
            norm_l,
            progress=gr.Progress(track_tqdm=True)
        ):
            if not text or not text.strip():
                raise gr.Error("Por favor, insira o texto a ser sintetizado.")

            def progress_hook(current, total, chunk_str):
                progress((current / total), desc=f"Sintetizando Chunk {current}/{total}: '{chunk_str[:30]}...'")

            preset_obj = load_preset(preset_name) if preset_name else None

            sr, wav_data = PIPELINE.generate_narration(
                text=text,
                preset=preset_obj,
                audio_prompt_path=audio_ref,
                temperature=float(t),
                repetition_penalty=float(rp),
                top_p=float(tp),
                min_p=float(mp),
                top_k=int(tk),
                base_seed=int(seed),
                n_cfm_timesteps=int(n_cfm_val),
                max_chars_per_chunk=int(max_c),
                norm_loudness=norm_l,
                progress_callback=progress_hook
            )

            # Salva em arquivo temporário para tocar e baixar no Gradio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, wav_data, sr)
            return temp_file.name

        generate_btn.click(
            fn=run_narration,
            inputs=[
                preset_dropdown,
                text_input,
                ref_audio_input,
                temp,
                rep_penalty,
                top_p,
                min_p,
                top_k,
                seed_num,
                n_cfm,
                max_chars,
                norm_loudness
            ],
            outputs=[audio_output]
        )

    return demo


if __name__ == "__main__":
    import sys
    
    # Verifica se está em modo de desenvolvimento (com --reload)
    dev_mode = "--reload" in sys.argv or "--dev" in sys.argv
    
    app = build_app()
    app.queue(max_size=20).launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,  # Gradio 6.x: css moved from Blocks() to launch()
        debug=dev_mode,  # Ativa debug mode em desenvolvimento
        show_error=True  # Mostra erros detalhados
    )
    
    print("\n💡 Dica: Use 'python gradio_tts_turbo_app.py --reload' para recarregar automaticamente ao editar código")
