#!/usr/bin/env python3
"""Script para aplicar CORREÇÃO 2 — Target LUFS alinhado com o modelo (-27 dB)"""

import json
import sys
import re

def apply_correction_2():
    notebook_path = 'local-gpu.ipynb'
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar Cell 4 (índice 4 no array cells)
    cell4 = nb['cells'][4]
    
    # Processar source code
    source = cell4['source']
    new_source = []
    modified = False
    
    for line in source:
        # Substituir -23.0 por -27.0 na linha de gain_linear
        if 'gain_linear = 10.0 ** ((' in line and '-23.0' in line:
            new_line = line.replace('-23.0', '-27.0')
            new_source.append(new_line)
            modified = True
            print(f'   ✓ Linha modificada: {line.strip()} → {new_line.strip()}')
        
        # Atualizar comentário se existir
        elif '# 2.4 Normalização de loudness para -23 LUFS' in line:
            new_line = line.replace('-23 LUFS', '-27 LUFS')
            new_source.append(new_line)
            print(f'   ✓ Comentário atualizado: -23 LUFS → -27 LUFS')
        
        else:
            new_source.append(line)
    
    if not modified:
        print('⚠️  Nenhuma modificação necessária ou linha de normalização não encontrada.')
        return False
    
    cell4['source'] = new_source
    
    # Salvar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print('✅ CORREÇÃO 2 aplicada com sucesso!')
    print('   - Target LUFS alterado: -23.0 → -27.0')
    print('   - Alinhado com target interno do modelo')
    return True

if __name__ == '__main__':
    try:
        success = apply_correction_2()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'❌ Erro ao aplicar correção: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
