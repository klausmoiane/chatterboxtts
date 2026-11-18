#!/usr/bin/env python3
"""Monitora execução do notebook Kaggle e baixa output automaticamente"""

import subprocess
import time
import sys

kaggle_exe = r'..\.venv\Scripts\kaggle.exe'
kernel = 'klausmultiuso/local-gpu'

print('=' * 60)
print('MONITORANDO EXECUÇÃO DO NOTEBOOK KAGGLE')
print('=' * 60)
print(f'Kernel: {kernel}')
print(f'Aguardando conclusão...')
print('=' * 60)

for attempt in range(60):  # 60 tentativas = ~30 minutos max
    result = subprocess.run(
        [kaggle_exe, 'kernels', 'status', kernel],
        capture_output=True,
        text=True,
        cwd='.'
    )
    
    status = result.stdout.strip()
    timestamp = time.strftime('%H:%M:%S')
    
    print(f'[{timestamp}] Status: {status}')
    
    if 'complete' in status.lower():
        print('=' * 60)
        print('✅ EXECUÇÃO COMPLETA! Baixando outputs...')
        print('=' * 60)
        
        download_result = subprocess.run(
            [kaggle_exe, 'kernels', 'output', kernel, '-p', 'outputs/', '--force'],
            capture_output=True,
            text=True,
            cwd='.'
        )
        
        if download_result.returncode == 0:
            print('✅ Download concluído com sucesso!')
            print(f'📁 Arquivos salvos em: kaggle_runner/outputs/')
            print('🎵 Áudio gerado: narracao_noticias_en_turbo_gpu.wav')
        else:
            print(f'❌ Erro no download: {download_result.stderr}')
        
        sys.exit(0)
    
    elif 'error' in status.lower() or 'failed' in status.lower():
        print('=' * 60)
        print(f'❌ ERRO NA EXECUÇÃO: {status}')
        print('=' * 60)
        sys.exit(1)
    
    time.sleep(30)  # Aguarda 30 segundos entre verificações

print('=' * 60)
print('⏱️ Timeout: Execução ainda não concluída após 30 minutos')
print('=' * 60)
sys.exit(1)
