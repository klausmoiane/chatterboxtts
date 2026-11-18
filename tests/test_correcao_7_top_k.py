#!/usr/bin/env python3
"""
Teste para CORREÇÃO 7 — Adicionar top_k=200
Verifica se o parâmetro foi adicionado ao model.generate()
"""

import json
import sys
import os
import re

def test_correcao_7():
    """Testa se a correção 7 foi aplicada corretamente"""
    
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
    
    # Teste 1: top_k=200 presente
    if re.search(r'top_k\s*=\s*200\b', source_code):
        tests_passed.append(f'✅ PASS — top_k=200 presente na Cell {cell_index}')
    else:
        tests_failed.append(f'❌ FAIL — top_k=200 NÃO encontrado na Cell {cell_index}')
    
    # Teste 2: Verificar que está dentro de model.generate()
    generate_match = re.search(r'model\.generate\((.*?)\)', source_code, re.DOTALL)
    if generate_match:
        generate_block = generate_match.group(1)
        if 'top_k=200' in generate_block:
            tests_passed.append('✅ PASS — top_k=200 está dentro de model.generate()')
        else:
            tests_failed.append('❌ FAIL — top_k=200 NÃO está dentro de model.generate()')
    else:
        tests_failed.append('❌ FAIL — Bloco model.generate() não encontrado')
    
    # Teste 3: Verificar ordem lógica (top_k deve vir após top_p)
    lines = generate_cell['source']
    top_p_line_idx = None
    top_k_line_idx = None
    
    for i, line in enumerate(lines):
        if 'top_p=' in line:
            top_p_line_idx = i
        if 'top_k=' in line:
            top_k_line_idx = i
    
    if top_p_line_idx is not None and top_k_line_idx is not None:
        if top_k_line_idx > top_p_line_idx:
            tests_passed.append('✅ PASS — Ordem correta (top_k após top_p)')
        else:
            tests_failed.append('❌ FAIL — Ordem incorreta (top_k ANTES de top_p)')
    elif top_k_line_idx is None:
        tests_failed.append('❌ FAIL — Linha com top_k=200 não encontrada')
    
    # Teste 4: Verificar sintaxe (vírgula após o parâmetro)
    if top_k_line_idx is not None:
        line = lines[top_k_line_idx]
        if line.strip().endswith(',') or ')' in line:
            tests_passed.append('✅ PASS — Sintaxe correta (vírgula presente)')
        else:
            tests_failed.append('❌ FAIL — Sintaxe incorreta (faltando vírgula)')
    
    # Exibir resultados
    print('=' * 60)
    print('TESTE — CORREÇÃO 7: top_k=200')
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
    success = test_correcao_7()
    sys.exit(0 if success else 1)
