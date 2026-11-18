#!/usr/bin/env python3
"""Script para aplicar CORREÇÃO 3 — Parâmetros de Geração"""

import json
import sys
import re

def apply_correction_3():
    notebook_path = 'local-gpu.ipynb'
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar Cell 8 (índice 8 no array cells - loop de geração)
    if len(nb['cells']) < 9:
        print('❌ FAIL — Notebook não possui Cell 8')
        return False
    
    cell6 = nb['cells'][8]  # Mantém nome da variável mas usa índice correto
    
    # Processar source code
    source = cell6['source']
    new_source = []
    modified_params = []
    
    for line in source:
        new_line = line
        
        # Substituir temperature=0.8 → 0.7
        if 'temperature=' in line and '0.8' in line:
            new_line = re.sub(r'temperature\s*=\s*0\.8', 'temperature=0.7', new_line)
            if new_line != line:
                modified_params.append('temperature: 0.8 → 0.7')
        
        # Substituir top_p=0.95 → 0.90
        if 'top_p=' in line and '0.95' in line:
            new_line = re.sub(r'top_p\s*=\s*0\.95', 'top_p=0.90', new_line)
            if new_line != line:
                modified_params.append('top_p: 0.95 → 0.90')
        
        # Substituir min_p=0.05 → 0.0
        if 'min_p=' in line and '0.05' in line:
            new_line = re.sub(r'min_p\s*=\s*0\.05', 'min_p=0.0', new_line)
            if new_line != line:
                modified_params.append('min_p: 0.05 → 0.0')
        
        # Substituir repetition_penalty=1.2 → 1.15
        if 'repetition_penalty=' in line and '1.2' in line:
            new_line = re.sub(r'repetition_penalty\s*=\s*1\.2\b', 'repetition_penalty=1.15', new_line)
            if new_line != line:
                modified_params.append('repetition_penalty: 1.2 → 1.15')
        
        # Substituir cfg_weight=0.5 → 0.0
        if 'cfg_weight=' in line and '0.5' in line:
            new_line = re.sub(r'cfg_weight\s*=\s*0\.5', 'cfg_weight=0.0', new_line)
            if new_line != line:
                modified_params.append('cfg_weight: 0.5 → 0.0')
        
        new_source.append(new_line)
    
    if len(modified_params) == 0:
        print('⚠️  Nenhuma modificação necessária ou parâmetros não encontrados.')
        return False
    
    cell6['source'] = new_source
    
    # Salvar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print('✅ CORREÇÃO 3 aplicada com sucesso!')
    print('   Parâmetros modificados:')
    for param in modified_params:
        print(f'   - {param}')
    return True

if __name__ == '__main__':
    try:
        success = apply_correction_3()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'❌ Erro ao aplicar correção: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
