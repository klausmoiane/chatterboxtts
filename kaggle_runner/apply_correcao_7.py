#!/usr/bin/env python3
"""Script para aplicar CORREÇÃO 7 — Adicionar top_k=200"""

import json
import sys
import re

def apply_correction_7():
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
    if 'top_k=' in source_text:
        print('⚠️  top_k já está presente. Nenhuma alteração feita.')
        return False
    
    for line in source:
        new_line = line
        
        # Adicionar top_k=200 após top_p (mantendo a ordem lógica dos parâmetros)
        if 'top_p=' in line and 'top_k' not in line:
            # Manter a linha do top_p
            if line.rstrip().endswith(',\n'):
                new_line = line
            elif line.rstrip().endswith(','):
                new_line = line.rstrip() + '\n'
            else:
                new_line = line.rstrip() + ',\n'
            
            new_source.append(new_line)
            # Adicionar a nova linha top_k logo após top_p
            new_source.append('        top_k=200,\n')
            modified = True
            continue
        
        new_source.append(line)
    
    if not modified:
        print('❌ FAIL — Linha top_p não encontrada para adicionar top_k')
        return False
    
    cell8['source'] = new_source
    
    # Salvar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print('✅ CORREÇÃO 7 aplicada com sucesso!')
    print('   - top_k=200 adicionado ao model.generate()')
    print('   - Limita pool de tokens candidatos')
    print('   - Reduz tokens sibilantes de baixa qualidade')
    return True

if __name__ == '__main__':
    try:
        success = apply_correction_7()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'❌ Erro ao aplicar correção: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
