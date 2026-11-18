#!/usr/bin/env python3
"""
Teste para CORREÇÃO 6 — Adicionar n_cfm_timesteps=8
Verifica se o parâmetro foi adicionado ao model.generate()
"""

import json
import sys
import os
import re

def test_correcao_6():
    """Testa se a correção 6 foi aplicada corretamente"""
    
    notebook_path = 'kaggle_runner/local-gpu.ipynb'
    
    if not os.path.exists(notebook_path):
        print('❌ FAIL — Notebook não encontrado:', notebook_path)
        return False
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar a célula com model.generate()
    generate_cell = None
    cell_index = None
    
    for idx, cell in enumerate(nb['cells']):
        source_text = ''.join(cell['source'])
        if 'model.generate(' in source_text and 'temperature=' in source_text:
            generate_cell = cell
            cell_index = idx
            break
    
    if generate_cell is None:
        print('❌ FAIL — Célula com model.generate() não encontrada')
        return False
    
    source_code = ''.join(generate_cell['source'])
    
    # Testes
    tests_passed = []
    tests_failed = []
    
    # Teste 1: n_cfm_timesteps=8 presente
    if re.search(r'n_cfm_timesteps\s*=\s*8\b', source_code):
        tests_passed.append(f'✅ PASS — n_cfm_timesteps=8 presente na Cell {cell_index}')
    else:
        tests_failed.append(f'❌ FAIL — n_cfm_timesteps=8 NÃO encontrado na Cell {cell_index}')
    
    # Teste 2: Verificar que está dentro de model.generate()
    # Extrair o bloco model.generate()
    generate_match = re.search(r'model\.generate\((.*?)\)', source_code, re.DOTALL)
    if generate_match:
        generate_block = generate_match.group(1)
        if 'n_cfm_timesteps=8' in generate_block:
            tests_passed.append('✅ PASS — n_cfm_timesteps=8 está dentro de model.generate()')
        else:
            tests_failed.append('❌ FAIL — n_cfm_timesteps=8 NÃO está dentro de model.generate()')
    else:
        tests_failed.append('❌ FAIL — Bloco model.generate() não encontrado')
    
    # Teste 3: Verificar sintaxe (vírgula após o parâmetro ou antes do fechamento)
    lines = generate_cell['source']
    cfm_line_found = False
    for line in lines:
        if 'n_cfm_timesteps=8' in line:
            cfm_line_found = True
            # Verificar que tem vírgula ou é a última linha antes do )
            if line.strip().endswith(',') or ')' in line:
                tests_passed.append('✅ PASS — Sintaxe correta (vírgula presente ou última linha)')
            else:
                tests_failed.append('❌ FAIL — Sintaxe incorreta (faltando vírgula)')
            break
    
    if not cfm_line_found:
        tests_failed.append('❌ FAIL — Linha com n_cfm_timesteps=8 não encontrada')
    
    # Exibir resultados
    print('=' * 60)
    print('TESTE — CORREÇÃO 6: n_cfm_timesteps=8')
    print('=' * 60)
    
    for test in tests_passed:
        print(test)
    
    for test in tests_failed:
        print(test)
    
    print('=' * 60)
    
    if len(tests_failed) == 0:
        print(f'✅ TODOS OS TESTES PASSARAM ({len(tests_passed)}/{len(tests_passed)})')
        print('=' * 60)
        return True
    else:
        print(f'❌ ALGUNS TESTES FALHARAM ({len(tests_passed)}/{len(tests_passed) + len(tests_failed)})')
        print('=' * 60)
        return False

if __name__ == '__main__':
    success = test_correcao_6()
    sys.exit(0 if success else 1)
