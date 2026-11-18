-"""
Servidor de desenvolvimento com hot reload automático
Monitora alterações em arquivos .py e reinicia o servidor automaticamente
"""
import os
import sys
import time
import subprocess
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configurações
APP_FILE = "gradio_tts_turbo_app.py"
WATCH_PATHS = [
    ".",  # Diretório raiz
    "src/chatterbox",  # Código do Chatterbox
    "voice_templates",  # Templates de voz
]
WATCH_EXTENSIONS = [".py", ".json"]
DEBOUNCE_SECONDS = 1.0  # Aguarda 1 segundo após última mudança antes de reiniciar

class HotReloadHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_reload = 0
        self.process = None
        self.pending_reload = False
        
    def should_reload(self, path):
        """Verifica se o arquivo deve acionar um reload"""
        path_obj = Path(path)
        
        # Ignora arquivos temporários, cache, etc
        if any(part.startswith('.') for part in path_obj.parts):
            return False
        if '__pycache__' in str(path):
            return False
        if path_obj.suffix not in WATCH_EXTENSIONS:
            return False
            
        return True
    
    def on_modified(self, event):
        if event.is_directory:
            return
            
        if not self.should_reload(event.src_path):
            return
        
        # Debounce: evita múltiplos reloads em sequência rápida
        current_time = time.time()
        if current_time - self.last_reload < DEBOUNCE_SECONDS:
            self.pending_reload = True
            return
        
        self.reload_server(event.src_path)
    
    def reload_server(self, changed_file):
        """Reinicia o servidor Gradio"""
        self.last_reload = time.time()
        self.pending_reload = False
        
        print("\n" + "="*80)
        print(f"🔄 ARQUIVO ALTERADO: {Path(changed_file).name}")
        print("="*80)
        
        # Mata o processo anterior
        if self.process:
            print("⏹️  Parando servidor anterior...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        
        # Inicia novo processo
        print("🚀 Reiniciando servidor...\n")
        self.process = subprocess.Popen(
            [sys.executable, APP_FILE],
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Thread para mostrar output do processo
        import threading
        def print_output():
            for line in self.process.stdout:
                print(line, end='')
        
        threading.Thread(target=print_output, daemon=True).start()
    
    def check_pending_reload(self):
        """Verifica se há reload pendente após debounce"""
        if self.pending_reload:
            current_time = time.time()
            if current_time - self.last_reload >= DEBOUNCE_SECONDS:
                self.reload_server("(múltiplos arquivos)")

def main():
    print("="*80)
    print("🔥 SERVIDOR DE DESENVOLVIMENTO COM HOT RELOAD")
    print("="*80)
    print(f"📂 Monitorando: {', '.join(WATCH_PATHS)}")
    print(f"📝 Extensões: {', '.join(WATCH_EXTENSIONS)}")
    print(f"⏱️  Debounce: {DEBOUNCE_SECONDS}s")
    print("="*80)
    print()
    
    # Verifica se watchdog está instalado
    try:
        import watchdog
    except ImportError:
        print("❌ ERRO: watchdog não está instalado")
        print("💡 Instale com: pip install watchdog")
        sys.exit(1)
    
    # Cria handler e observer
    handler = HotReloadHandler()
    observer = Observer()
    
    # Adiciona os diretórios para monitorar
    for path in WATCH_PATHS:
        if os.path.exists(path):
            observer.schedule(handler, path, recursive=True)
            print(f"✅ Monitorando: {path}")
    
    print()
    
    # Inicia o servidor pela primeira vez
    handler.reload_server("(início)")
    
    # Inicia o observer
    observer.start()
    
    try:
        print("\n💡 Servidor rodando. Edite qualquer arquivo .py para recarregar automaticamente.")
        print("💡 Pressione Ctrl+C para parar.\n")
        
        while True:
            time.sleep(0.5)
            handler.check_pending_reload()
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Parando servidor de desenvolvimento...")
        observer.stop()
        if handler.process:
            handler.process.terminate()
            handler.process.wait()
    
    observer.join()
    print("✅ Servidor parado.")

if __name__ == "__main__":
    main()
