#!/usr/bin/env python3
"""Script para aplicar CORREÇÃO 6 — Adicionar n_cfm_timesteps=8"""

import json
import sys
import re

def apply_correction_6():
    notebook_path = 'local-gpu.ipynb'
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar Cell 8 (índice 8 no array cells - loop de geração)
    if len(nb['cells']) < 9:
        print('❌ FAIL — Notebook não possui Cell 8')
        return False
    
    cell8 = nb['cells'][8]
    
    # Processar source code
    source = cell8['source']
    new_source = []
    modified = False
    
    # Verificar se já existe
    source_text = ''.join(source)
    if 'n_cfm_timesteps=' in source_text:
        print('⚠️  n_cfm_timesteps já está presente. Nenhuma alteração feita.')
        return False
    
    for line in source:
        new_line = line
        
        # Adicionar n_cfm_timesteps=8 após cfg_weight (última linha antes do fechamento)
        if 'cfg_weight=' in line:
            # Manter a linha do cfg_weight mas sem vírgula final se tiver
            if line.rstrip().endswith(',\n'):
                new_line = line
            elif line.rstrip().endswith(','):
                new_line = line.rstrip() + '\n'
            else:
                new_line = line.rstrip() + ',\n'
            
            new_source.append(new_line)
            # Adicionar a nova linha
            new_source.append('        n_cfm_timesteps=8,\n')
            modified = True
            continue
        
        new_source.append(line)
    
    if not modified:
        print('❌ FAIL — Linha cfg_weight não encontrada para adicionar n_cfm_timesteps')
        return False
    
    cell8['source'] = new_source
    
    # Salvar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print('✅ CORREÇÃO 6 aplicada com sucesso!')
    print('   - n_cfm_timesteps=8 adicionado ao model.generate()')
    print('   - Padrão era 4, agora explicitamente 8')
    print('   - Melhora mel spectrogram HF (4-8kHz)')
    return True

if __name__ == '__main__':
    try:
        success = apply_correction_6()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'❌ Erro ao aplicar correção: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
