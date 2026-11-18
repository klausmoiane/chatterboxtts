#!/usr/bin/env python3
"""Remove n_cfm_timesteps do notebook (parâmetro inválido)"""

import json

notebook_path = 'local-gpu.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Encontrar Cell 8
cell8 = nb['cells'][8]

# Remover linha com n_cfm_timesteps
new_source = []
for line in cell8['source']:
    if 'n_cfm_timesteps' not in line:
        new_source.append(line)

cell8['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('✅ n_cfm_timesteps removido do notebook')
