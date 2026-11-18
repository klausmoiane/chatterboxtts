"""
app.py - Interface Visual Local (UI) 100% Python para Geração de Áudio na GPU Kaggle.

Iniciar:
  streamlit run app.py

Arquitetura:
  - app_bridge.py  : controle thread-safe da sessão Kaggle (KaggleBridge)
  - voice_profiles.py : gerenciador de perfis e hiperparâmetros de voz
"""

import time
import subprocess
from pathlib import Path

import streamlit as st

from app_bridge import get_bridge, STATUS_IDLE, STATUS_QUEUED, STATUS_RUNNING, STATUS_COMPLETE, STATUS_ERROR
from voice_profiles import list_profiles, get_profile, build_voice_config, PARAM_RANGES

# ------------------------------------------------------------------ #
#  Configuração da página                                             #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="Chatterbox TTS — GPU Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  CSS customizado — tema escuro profissional                         #
# ------------------------------------------------------------------ #

st.markdown("""
<style>
/* Fundo geral */
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #161b22; }

/* Cards de métrica */
[data-testid="metric-container"] {
    background-color: #1c2333;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}

/* Área de log */
.log-box {
    background-color: #0d1117;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 12px;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #58a6ff;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

/* Badge de status */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.badge-idle     { background: #21262d; color: #8b949e; border: 1px solid #30363d; }
.badge-queued   { background: #1f2d3d; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-running  { background: #1f3d2f; color: #3fb950; border: 1px solid #238636; }
.badge-complete { background: #1f3d2f; color: #3fb950; border: 1px solid #238636; }
.badge-error    { background: #3d1f1f; color: #f85149; border: 1px solid #da3633; }

/* Botão primário */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #238636, #2ea043);
    border: none;
    font-size: 16px;
    padding: 12px 24px;
    border-radius: 8px;
    width: 100%;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
#  Inicialização do session_state                                     #
# ------------------------------------------------------------------ #

def _init_state():
    defaults = {
        "logs": [],
        "script_text": "",
        "selected_profile_id": "narrator_en",
        "temperature": None,        # None = usa o padrão do perfil
        "repetition_penalty": None,
        "seed": None,
        "use_custom_params": False,
        "last_downloaded_files": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()
bridge = get_bridge()

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

RTF = 0.967          # Real-Time Factor medido na Tesla T4
CHARS_PER_SECOND = 14.5  # estimativa empírica: ~14.5 chars de texto = 1s de áudio
PUSH_OVERHEAD_S = 45     # overhead de push + inicialização do kernel (segundos)


def _estimate_audio_duration(text: str) -> float:
    """Estimativa de duração do áudio em segundos dado o texto."""
    return max(len(text) / CHARS_PER_SECOND, 0.0)


def _estimate_gpu_time(audio_duration_s: float) -> float:
    """Tempo estimado de geração na GPU (segundos)."""
    return audio_duration_s * RTF + PUSH_OVERHEAD_S


def _status_badge(status: str) -> str:
    labels = {
        STATUS_IDLE:     ("idle",     "⚪ Idle"),
        STATUS_QUEUED:   ("queued",   "🔵 Na fila"),
        STATUS_RUNNING:  ("running",  "🟢 Executando"),
        STATUS_COMPLETE: ("complete", "✅ Concluído"),
        STATUS_ERROR:    ("error",    "🔴 Erro"),
    }
    cls, label = labels.get(status, ("idle", status))
    return f'<span class="badge badge-{cls}">{label}</span>'


def _check_kaggle_connectivity() -> tuple[bool, str]:
    """Testa se o CLI do Kaggle responde corretamente."""
    try:
        result = subprocess.run(
            ["kaggle", "kernels", "status", "klausmultiuso/local-gpu"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 or "complete" in result.stdout.lower() or "running" in result.stdout.lower():
            return True, "Conectado"
        return False, result.stderr.strip()[:80] or "Sem resposta"
    except Exception as e:
        return False, str(e)[:80]


def _poll_bridge():
    """Coleta logs do bridge e retorna True se ainda estiver rodando."""
    new_lines = bridge.drain_logs()
    if new_lines:
        st.session_state.logs.extend(new_lines)
    if bridge.status in (STATUS_COMPLETE, STATUS_ERROR):
        st.session_state.last_downloaded_files = bridge.downloaded_files
    return bridge.is_running


# ------------------------------------------------------------------ #
#  SIDEBAR — Perfis de Voz e Hiperparâmetros                         #
# ------------------------------------------------------------------ #

with st.sidebar:
    st.markdown("## 🎚️ Perfil de Voz")

    profiles = list_profiles()
    profile_options = {p.name: p.id for p in profiles}
    selected_name = st.selectbox(
        "Selecionar perfil",
        options=list(profile_options.keys()),
        index=list(profile_options.values()).index(st.session_state.selected_profile_id),
        disabled=bridge.is_running,
    )
    st.session_state.selected_profile_id = profile_options[selected_name]

    current_profile = get_profile(st.session_state.selected_profile_id)

    # Descrição do perfil
    with st.expander("ℹ️ Sobre este perfil", expanded=False):
        st.caption(current_profile.description)
        if current_profile.ref_audio:
            ref_path = Path(current_profile.ref_audio)
            if ref_path.exists():
                st.audio(str(ref_path), format="audio/wav")
                st.caption(f"🔊 Referência: `{ref_path.name}`")
            else:
                st.warning(f"Arquivo de referência não encontrado:\n`{ref_path.name}`")
        else:
            st.info("Sem áudio de referência — voz padrão do modelo.")
        if current_profile.notes:
            st.caption(f"📝 {current_profile.notes}")

    st.divider()
    st.markdown("## ⚙️ Hiperparâmetros")

    use_custom = st.toggle(
        "Customizar parâmetros",
        value=st.session_state.use_custom_params,
        disabled=bridge.is_running,
    )
    st.session_state.use_custom_params = use_custom

    if use_custom:
        r = PARAM_RANGES
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=r["temperature"]["min"],
            max_value=r["temperature"]["max"],
            value=st.session_state.temperature or current_profile.temperature,
            step=r["temperature"]["step"],
            help=r["temperature"]["help"],
            disabled=bridge.is_running,
        )
        st.session_state.repetition_penalty = st.slider(
            "Repetition Penalty",
            min_value=r["repetition_penalty"]["min"],
            max_value=r["repetition_penalty"]["max"],
            value=st.session_state.repetition_penalty or current_profile.repetition_penalty,
            step=r["repetition_penalty"]["step"],
            help=r["repetition_penalty"]["help"],
            disabled=bridge.is_running,
        )
        st.session_state.seed = st.number_input(
            "Seed",
            min_value=r["seed"]["min"],
            max_value=r["seed"]["max"],
            value=st.session_state.seed if st.session_state.seed is not None else current_profile.seed,
            step=r["seed"]["step"],
            help=r["seed"]["help"],
            disabled=bridge.is_running,
        )
    else:
        p = current_profile
        st.caption(f"🌡️ Temperature: **{p.temperature}**")
        st.caption(f"🔁 Repetition Penalty: **{p.repetition_penalty}**")
        st.caption(f"🌱 Seed: **{p.seed}**")

# ------------------------------------------------------------------ #
#  MAIN — Métricas da GPU                                             #
# ------------------------------------------------------------------ #

m1, m2, m3, m4 = st.columns(4)
m1.metric("🖥️ GPU", "NVIDIA Tesla T4")
m2.metric("💾 VRAM", "16 GB GDDR6")
m3.metric("⚡ RTF (medido)", f"{RTF}x")
m4.metric("📡 API Kaggle", "klausmultiuso/local-gpu")

st.divider()

# ------------------------------------------------------------------ #
#  MAIN — Seção 3.2: Entrada de Roteiro                               #
# ------------------------------------------------------------------ #

st.markdown("## 📝 Roteiro")

script_input = st.text_area(
    "Cole o roteiro aqui",
    value=st.session_state.script_text,
    height=220,
    placeholder="Digite ou cole o texto que será narrado pela IA...",
    disabled=bridge.is_running,
    label_visibility="collapsed",
)
st.session_state.script_text = script_input

# Contador dinâmico
text_now = st.session_state.script_text or ""
char_count = len(text_now)
word_count = len(text_now.split()) if text_now.strip() else 0
audio_est = _estimate_audio_duration(text_now)
gpu_est   = _estimate_gpu_time(audio_est)

cc1, cc2, cc3, cc4 = st.columns(4)
cc1.metric("Caracteres", f"{char_count:,}")
cc2.metric("Palavras", f"{word_count:,}")
cc3.metric("Duração estimada", f"{audio_est:.0f}s" if audio_est > 0 else "—")
cc4.metric("Tempo GPU estimado", f"~{gpu_est:.0f}s" if audio_est > 0 else "—")

st.divider()

# ------------------------------------------------------------------ #
#  MAIN — Seção 3.3: Painel de Controle                               #
# ------------------------------------------------------------------ #

st.markdown("## 🚀 Controle de Geração")

can_start = (
    bool(text_now.strip())
    and not bridge.is_running
    and bridge.status != STATUS_RUNNING
)

btn_col, info_col = st.columns([2, 3])

with btn_col:
    if bridge.is_running:
        st.button("⏳ Geração em andamento...", disabled=True, type="primary", use_container_width=True)
    elif bridge.status == STATUS_COMPLETE:
        if st.button("🔄 Nova Geração", type="primary", use_container_width=True):
            bridge.reset()
            st.session_state.logs = []
            st.rerun()
    else:
        if st.button(
            "🚀 Iniciar Geração na GPU (Nuvem)",
            type="primary",
            disabled=not can_start,
            use_container_width=True,
        ):
            # Monta configuração de voz
            voice_cfg = build_voice_config(
                profile_id=st.session_state.selected_profile_id,
                temperature=st.session_state.temperature if st.session_state.use_custom_params else None,
                repetition_penalty=st.session_state.repetition_penalty if st.session_state.use_custom_params else None,
                seed=st.session_state.seed if st.session_state.use_custom_params else None,
            )
            st.session_state.logs = []
            bridge.start(text=text_now, voice_profile=voice_cfg)
            st.rerun()

with info_col:
    if bridge.status == STATUS_ERROR:
        st.error("❌ A última geração terminou com erro. Verifique os logs abaixo.")
    elif bridge.status == STATUS_COMPLETE:
        st.success("✅ Geração concluída! Ouça e baixe o áudio na seção abaixo.")
    elif bridge.is_running:
        st.info("⚡ GPU em execução no Kaggle. Logs sendo transmitidos em tempo real...")

st.divider()

# ------------------------------------------------------------------ #
#  MAIN — Seção 3.4: Monitor de Execução e Logs                       #
# ------------------------------------------------------------------ #

st.markdown("## 📡 Monitor de Execução")

# Barra de progresso
progress_map = {
    STATUS_IDLE:     0.0,
    STATUS_QUEUED:   0.1,
    STATUS_RUNNING:  0.6,
    STATUS_COMPLETE: 1.0,
    STATUS_ERROR:    1.0,
}
progress_val = progress_map.get(bridge.status, 0.0)

if bridge.status == STATUS_ERROR:
    st.progress(progress_val, text="❌ Encerrado com erro")
elif bridge.status == STATUS_COMPLETE:
    st.progress(1.0, text="✅ Concluído com sucesso!")
elif bridge.status == STATUS_RUNNING:
    st.progress(progress_val, text="⚡ GPU executando na nuvem...")
elif bridge.status == STATUS_QUEUED:
    st.progress(progress_val, text="🔵 Na fila — preparando notebook...")
else:
    st.progress(0.0, text="Aguardando início...")

# Terminal de logs retrátil
with st.expander("📋 Logs do terminal (ao vivo)", expanded=bridge.is_running):
    if st.session_state.logs:
        log_text = "\n".join(st.session_state.logs[-200:])  # últimas 200 linhas
        st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)
    else:
        st.caption("Nenhum log ainda. Os logs aparecerão aqui durante a geração.")

# Auto-refresh enquanto está rodando
if bridge.is_running:
    _poll_bridge()
    time.sleep(1)  # atualiza a cada 1 segundo
    st.rerun()
elif bridge.status in (STATUS_COMPLETE, STATUS_ERROR):
    _poll_bridge()   # drena logs finais

st.divider()

# ------------------------------------------------------------------ #
#  MAIN — Seção 3.5: Player de Áudio e Central de Downloads           #
# ------------------------------------------------------------------ #

st.markdown("## 🎧 Áudios Gerados")

outputs_dir = Path(__file__).resolve().parent / "outputs"

# Arquivos da última geração
last_files = st.session_state.last_downloaded_files

if bridge.status == STATUS_COMPLETE and last_files:
    st.success(f"✅ {len(last_files)} arquivo(s) disponível(is) da última geração:")
    for audio_file in last_files:
        if audio_file.exists() and audio_file.suffix.lower() in (".wav", ".mp3", ".ogg"):
            fc1, fc2 = st.columns([4, 1])
            with fc1:
                st.audio(str(audio_file), format=f"audio/{audio_file.suffix.lstrip('.')}")
                st.caption(f"📁 `{audio_file.name}` — {audio_file.stat().st_size / 1024:.1f} KB")
            with fc2:
                with open(audio_file, "rb") as fh:
                    st.download_button(
                        label="⬇️ Baixar",
                        data=fh,
                        file_name=audio_file.name,
                        mime=f"audio/{audio_file.suffix.lstrip('.')}",
                        use_container_width=True,
                    )

# Histórico completo da pasta outputs/
st.markdown("### 📂 Histórico — pasta `outputs/`")

if outputs_dir.exists():
    all_audio_files = sorted(
        [f for f in outputs_dir.glob("*") if f.suffix.lower() in (".wav", ".mp3", ".ogg")],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if all_audio_files:
        for idx, audio_file in enumerate(all_audio_files[:10]):  # máximo 10 no histórico
            mtime = time.strftime("%d/%m/%Y %H:%M", time.localtime(audio_file.stat().st_mtime))
            size_kb = audio_file.stat().st_size / 1024
            with st.expander(f"🔊 {audio_file.name} — {mtime} — {size_kb:.1f} KB", expanded=(idx == 0)):
                hc1, hc2 = st.columns([4, 1])
                with hc1:
                    st.audio(str(audio_file), format=f"audio/{audio_file.suffix.lstrip('.')}")
                with hc2:
                    with open(audio_file, "rb") as fh:
                        st.download_button(
                            label="⬇️ Baixar",
                            data=fh,
                            file_name=audio_file.name,
                            mime=f"audio/{audio_file.suffix.lstrip('.')}",
                            key=f"dl_{idx}",
                            use_container_width=True,
                        )
    else:
        st.info("Nenhum áudio encontrado em `outputs/` ainda.")
else:
    st.info("A pasta `outputs/` será criada após a primeira geração.")

# ------------------------------------------------------------------ #
#  Rodapé                                                             #
# ------------------------------------------------------------------ #

st.divider()
st.caption("Chatterbox TTS · GPU Studio · Powered by Kaggle NVIDIA Tesla T4 · RTF 0.967x")
