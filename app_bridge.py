"""
app_bridge.py - Interface assíncrona/thread-safe entre a UI Streamlit e o pipeline Kaggle.

Responsabilidades:
  - Disparar as etapas do run_on_kaggle.py em uma thread separada (não trava a UI).
  - Transmitir status em tempo real via callbacks: Queued → Running → Complete / Error.
  - Capturar stdout/stderr linha a linha e repassar para a UI via fila thread-safe.
  - Expor estado atual (status, logs, arquivos baixados) de forma segura para leitura no Streamlit.
"""

import sys
import time
import threading
import subprocess
from pathlib import Path
from queue import Queue, Empty
from typing import Callable

ROOT_DIR = Path(__file__).resolve().parent
KAGGLE_RUNNER_DIR = ROOT_DIR / "kaggle_runner"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VENV_KAGGLE_BIN = ROOT_DIR / ".venv" / "Scripts" / "kaggle.exe"
KERNEL_SLUG = "klausmultiuso/local-gpu"

# Possíveis estados da sessão
STATUS_IDLE = "idle"
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"


def _get_kaggle_cmd() -> str:
    """Retorna o executável correto do kaggle (venv ou global)."""
    if VENV_KAGGLE_BIN.exists():
        return str(VENV_KAGGLE_BIN)
    return "kaggle"


def _run_subprocess_streaming(cmd: list[str], log_queue: Queue) -> int:
    """
    Executa um subprocesso e enfileira cada linha de stdout/stderr na log_queue.
    Retorna o código de saída do processo.
    """
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in process.stdout:
        log_queue.put(line.rstrip())
    process.wait()
    return process.returncode


class KaggleBridge:
    """
    Controla uma única sessão de geração de áudio no Kaggle.
    Thread-safe: pode ser lida pela thread do Streamlit enquanto a thread worker avança.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._status: str = STATUS_IDLE
        self._log_queue: Queue = Queue()
        self._worker_thread: threading.Thread | None = None
        self._downloaded_files: list[Path] = []
        self._on_status_change: Callable[[str], None] | None = None

    # ------------------------------------------------------------------ #
    #  Propriedades thread-safe                                           #
    # ------------------------------------------------------------------ #

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def is_running(self) -> bool:
        return self.status in (STATUS_QUEUED, STATUS_RUNNING)

    @property
    def downloaded_files(self) -> list[Path]:
        with self._lock:
            return list(self._downloaded_files)

    def set_on_status_change(self, callback: Callable[[str], None]):
        """Registra um callback chamado sempre que o status muda."""
        self._on_status_change = callback

    # ------------------------------------------------------------------ #
    #  Leitura de logs (non-blocking)                                     #
    # ------------------------------------------------------------------ #

    def drain_logs(self) -> list[str]:
        """
        Retorna todas as linhas de log disponíveis no momento da chamada.
        Não bloqueia — retorna lista vazia se não houver nada.
        Deve ser chamado periodicamente no loop de polling do Streamlit (st.rerun).
        """
        lines = []
        while True:
            try:
                lines.append(self._log_queue.get_nowait())
            except Empty:
                break
        return lines

    # ------------------------------------------------------------------ #
    #  Controle de execução                                               #
    # ------------------------------------------------------------------ #

    def start(
        self,
        text: str | None = None,
        input_file: str | None = None,
        voice_profile: dict | None = None,
        poll_interval: int = 1,
        timeout: int = 1200,
    ):
        """
        Inicia a geração em background. Retorna imediatamente.

        Parâmetros:
          text          : texto direto a narrar (alternativo a input_file).
          input_file    : caminho para arquivo .txt com o roteiro.
          voice_profile : dicionário com campos do VoiceProfile (ver voice_profiles.py).
          poll_interval : segundos entre cada consulta de status ao Kaggle.
          timeout       : tempo máximo de espera em segundos.
        """
        if self.is_running:
            raise RuntimeError("Uma geração já está em andamento. Aguarde o término.")

        self._set_status(STATUS_QUEUED)
        self._downloaded_files = []

        self._worker_thread = threading.Thread(
            target=self._worker,
            args=(text, input_file, voice_profile, poll_interval, timeout),
            daemon=True,
        )
        self._worker_thread.start()

    def reset(self):
        """Volta ao estado idle. Só deve ser chamado após término da execução."""
        if self.is_running:
            raise RuntimeError("Não é possível resetar enquanto uma geração está ativa.")
        with self._lock:
            self._status = STATUS_IDLE
            self._downloaded_files = []
        # Descarta logs antigos
        while not self._log_queue.empty():
            try:
                self._log_queue.get_nowait()
            except Empty:
                break

    # ------------------------------------------------------------------ #
    #  Worker (executa em thread separada)                                #
    # ------------------------------------------------------------------ #

    def _set_status(self, new_status: str):
        with self._lock:
            self._status = new_status
        self._log_queue.put(f"[STATUS] → {new_status.upper()}")
        if self._on_status_change:
            try:
                self._on_status_change(new_status)
            except Exception:
                pass

    def _worker(
        self,
        text: str | None,
        input_file: str | None,
        voice_profile: dict | None,
        poll_interval: int,
        timeout: int,
    ):
        try:
            # --- Etapa 1: Preparar input ---
            self._log_queue.put("[1/3] Preparando roteiro e notebook...")
            self._prepare_input(text, input_file, voice_profile)

            # --- Etapa 2: Push do kernel ---
            self._log_queue.put("[2/3] Enviando notebook para o Kaggle (GPU)...")
            self._set_status(STATUS_RUNNING)
            rc = self._push_kernel()
            if rc != 0:
                raise RuntimeError(f"Push do kernel falhou (exit code {rc}).")

            # --- Etapa 3: Monitorar ---
            self._log_queue.put("[3/3] Monitorando execução remota na Tesla T4...")
            final_status = self._monitor_status(poll_interval, timeout)

            if final_status == "complete":
                # --- Etapa 4: Download ---
                self._log_queue.put("[4/4] Baixando áudios gerados...")
                files = self._download_outputs()
                with self._lock:
                    self._downloaded_files = files
                self._log_queue.put(f"[OK] {len(files)} arquivo(s) baixado(s) para ./outputs/")
                self._set_status(STATUS_COMPLETE)
            else:
                raise RuntimeError("Kernel terminou com erro no Kaggle.")

        except TimeoutError as e:
            self._log_queue.put(f"[TIMEOUT] {e}")
            self._set_status(STATUS_ERROR)
        except Exception as e:
            self._log_queue.put(f"[ERRO] {e}")
            self._set_status(STATUS_ERROR)

    # ------------------------------------------------------------------ #
    #  Etapas internas (espelham run_on_kaggle.py com streaming de logs)  #
    # ------------------------------------------------------------------ #

    def _prepare_input(self, text: str | None, input_file: str | None, voice_profile: dict | None):
        """Injeta texto e parâmetros de voz no notebook antes do push."""
        if input_file:
            with open(input_file, "r", encoding="utf-8") as f:
                script_text = f.read().strip()
        elif text:
            script_text = text.strip()
        else:
            script_text = None

        if script_text:
            temp_input = KAGGLE_RUNNER_DIR / "input_text.txt"
            temp_input.write_text(script_text, encoding="utf-8")
            self._log_queue.put(f"[OK] Texto gravado ({len(script_text)} chars) → {temp_input.name}")

        # Injeta hiperparâmetros do perfil de voz no notebook, se fornecidos
        if voice_profile:
            self._inject_voice_params(voice_profile)

        # Regenera o notebook
        make_script = KAGGLE_RUNNER_DIR / "make_notebook_noticias_en.py"
        if not make_script.exists():
            make_script = KAGGLE_RUNNER_DIR / "make_notebook.py"
        if make_script.exists():
            rc = _run_subprocess_streaming([sys.executable, str(make_script)], self._log_queue)
            if rc != 0:
                raise RuntimeError("Falha ao regenerar o notebook.")

    def _inject_voice_params(self, profile: dict):
        """
        Escreve um arquivo JSON de configuração que o notebook lê para aplicar
        os parâmetros de voz (temperature, repetition_penalty, seed, ref_audio).
        """
        import json
        config_path = KAGGLE_RUNNER_DIR / "voice_config.json"
        config_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
        self._log_queue.put(f"[OK] Configuração de voz gravada → {config_path.name}")

    def _push_kernel(self) -> int:
        kaggle_bin = _get_kaggle_cmd()
        return _run_subprocess_streaming(
            [kaggle_bin, "kernels", "push", "-p", str(KAGGLE_RUNNER_DIR), "--accelerator", "NvidiaTeslaT4"],
            self._log_queue,
        )

    def _monitor_status(self, poll_interval: int, timeout: int) -> str:
        kaggle_bin = _get_kaggle_cmd()
        start = time.time()
        last_line = ""

        while time.time() - start < timeout:
            proc = subprocess.run(
                [kaggle_bin, "kernels", "status", KERNEL_SLUG],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            output = proc.stdout.strip()
            status_line = output.split("\n")[0] if output else "desconhecido"

            if status_line != last_line:
                self._log_queue.put(f"[{time.strftime('%H:%M:%S')}] Kaggle status: {status_line}")
                last_line = status_line

            lower = output.lower()
            if "complete" in lower:
                return "complete"
            if "error" in lower or "failed" in lower:
                return "error"

            time.sleep(poll_interval)

        raise TimeoutError(f"Timeout de {timeout}s atingido aguardando o Kaggle.")

    def _download_outputs(self) -> list[Path]:
        kaggle_bin = _get_kaggle_cmd()
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        _run_subprocess_streaming(
            [kaggle_bin, "kernels", "output", KERNEL_SLUG, "-p", str(OUTPUTS_DIR), "--force"],
            self._log_queue,
        )
        files = sorted(OUTPUTS_DIR.glob("*.*"), key=lambda f: f.stat().st_mtime, reverse=True)
        for f in files:
            size_kb = f.stat().st_size / 1024
            self._log_queue.put(f"  ↳ {f.name} ({size_kb:.1f} KB)")
        return files


# ------------------------------------------------------------------ #
#  Instância global compartilhada pela UI Streamlit                   #
# ------------------------------------------------------------------ #
# O Streamlit reroda o script a cada interação, mas o estado vive em
# st.session_state. Instancie via get_bridge() para reutilizar.

_bridge_instance: KaggleBridge | None = None
_bridge_lock = threading.Lock()


def get_bridge() -> KaggleBridge:
    """Retorna (ou cria) a instância singleton do KaggleBridge."""
    global _bridge_instance
    with _bridge_lock:
        if _bridge_instance is None:
            _bridge_instance = KaggleBridge()
        return _bridge_instance
