import json, os
from pathlib import Path

# Credenciais
creds_path = Path.home() / '.kaggle' / 'kaggle.json'
with open(creds_path) as f:
    creds = json.load(f)
print('Username:', creds.get('username'))
print('Key (last 4):', creds.get('key','')[-4:])

# Notebook
nb_path = 'kaggle_runner/local-gpu.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
print('Notebook cells:', len(nb['cells']))
nb_str = json.dumps(nb, ensure_ascii=False)
size_mb = len(nb_str.encode('utf-8')) / 1024 / 1024
print('Notebook size: %.2f MB' % size_mb)

# Metadata
with open('kaggle_runner/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)
print('Metadata:', json.dumps(meta, indent=2))

