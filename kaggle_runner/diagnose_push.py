"""
Diagnóstico: envia diretamente para a API do Kaggle e mostra o erro detalhado.
"""
import json, requests
from pathlib import Path

# Ler token
token_path = Path.home() / '.kaggle' / 'access_token'
with open(token_path) as f:
    token = f.read().strip()

# Ler notebook e metadata
with open('kaggle_runner/local-gpu.ipynb', 'r', encoding='utf-8') as f:
    nb_content = f.read()

with open('kaggle_runner/kernel-metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

# Preparar payload (mesmo formato que o CLI usa internamente)
payload = {
    "slug": meta["id"].split("/")[1],
    "newTitle": meta["title"],
    "source": nb_content,
    "language": "python",
    "kernelType": "notebook",
    "isPrivate": meta["is_private"],
    "enableGpu": meta["enable_gpu"],
    "enableTpu": meta["enable_tpu"],
    "enableInternet": meta["enable_internet"],
    "datasetDataSources": meta.get("dataset_sources", []),
    "kernelDataSources": [],
    "competitionDataSources": [],
    "modelDataSources": [],
    "categoryIds": [],
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

print("Enviando para Kaggle API...")
size_mb = len(json.dumps(payload).encode()) / 1024 / 1024
print("Payload size: %.2f MB" % size_mb)

resp = requests.post(
    "https://www.kaggle.com/api/v1/kernels/save",
    headers=headers,
    json=payload,
    timeout=60,
)

print("Status: %d" % resp.status_code)
print("Response:")
try:
    print(json.dumps(resp.json(), indent=2))
except Exception:
    print(resp.text[:3000])

