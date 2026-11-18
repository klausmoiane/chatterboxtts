import subprocess, time

kaggle = r'.venv\Scripts\kaggle.exe'
kernel = 'klausmultiuso/local-gpu'

print('Aguardando COMPLETE...')
for attempt in range(30):
    r = subprocess.run([kaggle, 'kernels', 'status', kernel], capture_output=True, text=True)
    status = r.stdout.strip()
    ts = time.strftime("%H:%M:%S")
    print(f'[{ts}] {status}')

    if 'complete' in status.lower():
        print('COMPLETO! Baixando outputs...')
        r2 = subprocess.run(
            [kaggle, 'kernels', 'output', kernel, '-p', 'outputs/', '--force'],
            capture_output=True, text=True
        )
        print('STDOUT:', r2.stdout.strip())
        print('STDERR:', r2.stderr.strip())
        print('RC:', r2.returncode)
        break
    elif 'error' in status.lower() or 'failed' in status.lower():
        print('ERRO no kernel - abortando.')
        break
    time.sleep(20)
else:
    print('Timeout atingido.')

