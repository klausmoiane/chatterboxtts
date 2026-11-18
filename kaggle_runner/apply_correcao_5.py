#!/usr/bin/env python3
"""Script para aplicar CORREÇÃO 5 — Ajustar redução de ruído"""

import json
import sys
import re

def apply_correction_5():
    notebook_path = 'local-gpu.ipynb'
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar Cell 4 (índice 4 no array cells)
    cell4 = nb['cells'][4]
    
    # Processar source code
    source = cell4['source']
    new_source = []
    modified_params = []
    
    for line in source:
        new_line = line
        
        # Substituir prop_decrease=0.8 → 0.2
        if 'prop_decrease=' in line and '0.8' in line:
            new_line = re.sub(r'prop_decrease\s*=\s*0\.8', 'prop_decrease=0.2', new_line)
            if new_line != line:
                modified_params.append('prop_decrease: 0.8 → 0.2')
        
        # Aumentar janela de ruído de 0.5s → 1.0s
        if 'noise_sample = ref_wav[:int(ref_sr *' in line and '0.5' in line:
            new_line = re.sub(r'ref_sr\s*\*\s*0\.5', 'ref_sr * 1.0', new_line)
            if new_line != line:
                modified_params.append('janela de ruído: 0.5s → 1.0s')
        
        new_source.append(new_line)
    
    if len(modified_params) == 0:
        print('⚠️  Nenhuma modificação necessária ou parâmetros não encontrados.')
        return False
    
    cell4['source'] = new_source
    
    # Salvar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print('✅ CORREÇÃO 5 aplicada com sucesso!')
    print('   Parâmetros de noisereduce ajustados:')
    for param in modified_params:
        print(f'   - {param}')
    return True

if __name__ == '__main__':
    try:
        success = apply_correction_5()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'❌ Erro ao aplicar correção: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
