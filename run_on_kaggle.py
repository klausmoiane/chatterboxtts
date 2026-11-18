"""
run_on_kaggle.py - Orquestrador Local para Síntese Chatterbox TTS 100% GPU no Kaggle

Uso:
  python run_on_kaggle.py --input meu_roteiro.txt
  python run_on_kaggle.py --text "Texto direto a ser narrado pela IA"
  python run_on_kaggle.py --status-only
  python run_on_kaggle.py --download-only
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
KAGGLE_RUNNER_DIR = ROOT_DIR / "kaggle_runner"
OUTPUTS_DIR = ROOT_DIR / "outputs"
VENV_KAGGLE_BIN = ROOT_DIR / ".venv" / "Scripts" / "kaggle.exe"

KERNEL_SLUG = "klausmultiuso/local-gpu"


def get_kaggle_cmd() -> str:
    """Retorna o executável correto do kaggle."""
    if VENV_KAGGLE_BIN.exists():
        return str(VENV_KAGGLE_BIN)
    return "kaggle"


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Executa comando no subprocess garantindo variáveis de ambiente e encoding."""
    env = os.environ.copy()
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, text=True, capture_output=True, encoding="utf-8", errors="replace")
    if check and result.returncode != 0:
        print(f"[ERRO STDERR]: {result.stderr.strip()}")
        raise RuntimeError(f"Comando falhou com código {result.returncode}")
    return result


def prepare_input(text: str | None, input_file: str | None, voice_ref: str | None):
    """Prepara o notebook do Kaggle Runner injetando o novo texto e referência se fornecido."""
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            script_text = f.read().strip()
    elif text:
        script_text = text.strip()
    else:
        script_text = None

    # Se há texto novo, injeta gerando a versão atualizada do local-gpu.ipynb
    make_notebook_script = KAGGLE_RUNNER_DIR / "make_notebook_noticias_en.py"
    if not make_notebook_script.exists():
        make_notebook_script = KAGGLE_RUNNER_DIR / "make_notebook.py"
    if make_notebook_script.exists():
        python_bin = sys.executable
        if script_text:
            temp_input_txt = KAGGLE_RUNNER_DIR / "input_text.txt"
            with open(temp_input_txt, "w", encoding="utf-8") as f:
                f.write(script_text)
            print(f"[OK] Texto de entrada preparado ({len(script_text)} caracteres).")

        run_command([python_bin, str(make_notebook_script)])


def push_kernel():
    """Envia a versão mais recente do notebook para execução na GPU do Kaggle."""
    kaggle_bin = get_kaggle_cmd()
    print(f"\n[1/3] Enviando notebook para Kaggle ({KERNEL_SLUG})...")
    res = run_command([kaggle_bin, "kernels", "push", "-p", str(KAGGLE_RUNNER_DIR), "--accelerator", "NvidiaTeslaT4"])
    print(res.stdout.strip())


def monitor_status(poll_interval: int = 15, timeout: int = 1200) -> str:
    """Monitora o status de execução até que esteja completo ou apresente erro."""
    kaggle_bin = get_kaggle_cmd()
    print(f"\n[2/3] Monitorando execução remota na GPU do Kaggle...")
    start_time = time.time()
    
    last_status = ""
    while time.time() - start_time < timeout:
        res = run_command([kaggle_bin, "kernels", "status", KERNEL_SLUG], check=False)
        output = res.stdout.strip()
        
        status_line = output.split("\n")[0] if output else "Desconhecido"
        if status_line != last_status:
            print(f"[{time.strftime('%H:%M:%S')}] Status: {status_line}")
            last_status = status_line
        
        lower_out = output.lower()
        if "complete" in lower_out:
            print(f"\n[SUCESSO] Processamento concluído com êxito na GPU!")
            return "complete"
        elif "error" in lower_out or "failed" in lower_out:
            print(f"\n[FALHA] O kernel terminou com erro no Kaggle:")
            print(output)
            return "error"
            
        time.sleep(poll_interval)

    raise TimeoutError(f"Tempo limite excedido ({timeout}s) aguardando o Kaggle.")


def download_outputs() -> list[Path]:
    """Baixa os arquivos resultantes gerados pela GPU."""
    kaggle_bin = get_kaggle_cmd()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/3] Baixando arquivos de saída (áudios gerados) para {OUTPUTS_DIR}...")
    res = run_command([kaggle_bin, "kernels", "output", KERNEL_SLUG, "-p", str(OUTPUTS_DIR), "--force"])
    print(res.stdout.strip())
    
    downloaded_files = list(OUTPUTS_DIR.glob("*.*"))
    print("\nArquivos disponíveis em ./outputs/:")
    for f in downloaded_files:
        size_kb = f.stat().st_size / 1024
        print(f" - {f.name} ({size_kb:.1f} KB)")
    return downloaded_files


def main():
    parser = argparse.ArgumentParser(description="Orquestrador Local para Chatterbox TTS no Kaggle (GPU)")
    parser.add_argument("--input", "-i", type=str, help="Caminho para o arquivo .txt com o roteiro")
    parser.add_argument("--text", "-t", type=str, help="Texto direto a ser narrado")
    parser.add_argument("--ref", "-r", type=str, help="Caminho para áudio de referência de voz")
    parser.add_argument("--status-only", action="store_true", help="Apenas consulta o status atual do kernel")
    parser.add_argument("--download-only", action="store_true", help="Apenas baixa os últimos outputs gerados")
    parser.add_argument("--no-wait", action="store_true", help="Faz o push e não aguarda o término")

    args = parser.parse_args()

    if args.status_only:
        kaggle_bin = get_kaggle_cmd()
        res = run_command([kaggle_bin, "kernels", "status", KERNEL_SLUG], check=False)
        print(res.stdout.strip())
        return

    if args.download_only:
        download_outputs()
        return

    # Fluxo Completo: Preparar -> Enviar -> Monitorar -> Baixar
    prepare_input(text=args.text, input_file=args.input, voice_ref=args.ref)
    push_kernel()

    if args.no_wait:
        print("\nNotebook submetido. Acompanhe em: https://www.kaggle.com/code/" + KERNEL_SLUG)
        return

    status = monitor_status()
    if status == "complete":
        download_outputs()


if __name__ == "__main__":
    main()

