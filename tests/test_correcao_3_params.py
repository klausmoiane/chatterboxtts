#!/usr/bin/env python3
"""
Teste para CORREÇÃO 3 — Parâmetros de Geração
Verifica se os parâmetros de model.generate() foram ajustados corretamente
"""

import json
import sys
import os
import re

def test_correcao_3():
    """Testa se a correção 3 foi aplicada corretamente"""
    
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
    
    # Teste 1: temperature=0.7
    if re.search(r'temperature\s*=\s*0\.7\b', source_code):
        tests_passed.append(f'✅ PASS — temperature=0.7 na Cell {cell_index}')
    else:
        tests_failed.append(f'❌ FAIL — temperature NÃO é 0.7 na Cell {cell_index}')
    
    # Teste 2: top_p=0.90 ou 0.9
    if re.search(r'top_p\s*=\s*0\.9\d*\b', source_code):
        tests_passed.append(f'✅ PASS — top_p=0.90 na Cell {cell_index}')
    else:
        tests_failed.append(f'❌ FAIL — top_p NÃO é 0.90 na Cell {cell_index}')
    
    # Teste 3: min_p=0.0
    if re.search(r'min_p\s*=\s*0\.0\b', source_code):
        tests_passed.append(f'✅ PASS — min_p=0.0 na Cell {cell_index}')
    else:
        tests_failed.append(f'❌ FAIL — min_p NÃO é 0.0 na Cell {cell_index}')
    
    # Teste 4: repetition_penalty=1.15
    if re.search(r'repetition_penalty\s*=\s*1\.15\b', source_code):
        tests_passed.append(f'✅ PASS — repetition_penalty=1.15 na Cell {cell_index}')
    else:
        tests_failed.append(f'❌ FAIL — repetition_penalty NÃO é 1.15 na Cell {cell_index}')
    
    # Teste 5: cfg_weight=0.0 (opcional)
    if re.search(r'cfg_weight\s*=\s*0\.0\b', source_code):
        tests_passed.append(f'✅ PASS — cfg_weight=0.0 na Cell {cell_index}')
    
    # Teste 6: Verificar que valores antigos não existem
    old_values_check = []
    if re.search(r'temperature\s*=\s*0\.8\b', source_code):
        old_values_check.append('temperature=0.8 ainda presente')
    if re.search(r'top_p\s*=\s*0\.95\b', source_code):
        old_values_check.append('top_p=0.95 ainda presente')
    if re.search(r'min_p\s*=\s*0\.05\b', source_code):
        old_values_check.append('min_p=0.05 ainda presente')
    if re.search(r'repetition_penalty\s*=\s*1\.2\b', source_code):
        old_values_check.append('repetition_penalty=1.2 ainda presente')
    
    if len(old_values_check) == 0:
        tests_passed.append('✅ PASS — Valores antigos removidos')
    else:
        tests_failed.append(f'❌ FAIL — Valores antigos ainda presentes: {", ".join(old_values_check)}')
    
    # Exibir resultados
    print('=' * 60)
    print('TESTE — CORREÇÃO 3: Parâmetros de Geração')
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
    success = test_correcao_3()
    sys.exit(0 if success else 1)
