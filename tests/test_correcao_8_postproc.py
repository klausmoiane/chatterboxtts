#!/usr/bin/env python3
"""
Teste para CORREÇÃO 8 — Pós-processamento final: Lowpass + Normalização
Verifica se o pós-processamento foi adicionado corretamente
"""

import json
import sys
import os
import re

def test_correcao_8():
    """Testa se a correção 8 foi aplicada corretamente"""
    
    notebook_path = 'kaggle_runner/local-gpu.ipynb'
    
    if not os.path.exists(notebook_path):
        print('❌ FAIL — Notebook não encontrado:', notebook_path)
        return False
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar a célula com deess_output e sf.write
    target_cell = None
    cell_index = None
    
    for idx, cell in enumerate(nb['cells']):
        source_text = ''.join(cell['source'])
        if 'deess_output' in source_text and 'sf.write(OUTPUT_PATH' in source_text:
            target_cell = cell
            cell_index = idx
            break
    
    if target_cell is None:
        print('❌ FAIL — Célula de output com sf.write() não encontrada')
        return False
    
    source_code = ''.join(target_cell['source'])
    
    # Testes
    tests_passed = []
    tests_failed = []
    
    # Teste 1: Função apply_lowpass presente
    if 'def apply_lowpass(' in source_code:
        tests_passed.append('✅ PASS — Função apply_lowpass presente')
    else:
        tests_failed.append('❌ FAIL — Função apply_lowpass NÃO encontrada')
    
    # Teste 2: Função final_normalization presente
    if 'def final_normalization(' in source_code:
        tests_passed.append('✅ PASS — Função final_normalization presente')
    else:
        tests_failed.append('❌ FAIL — Função final_normalization NÃO encontrada')
    
    # Teste 3: Chamada apply_lowpass(final_audio, SR, cutoff=7500)
    if re.search(r'final_audio\s*=\s*apply_lowpass\s*\(\s*final_audio\s*,\s*SR', source_code):
        tests_passed.append('✅ PASS — Chamada apply_lowpass(final_audio, SR) presente')
    else:
        tests_failed.append('❌ FAIL — Chamada apply_lowpass NÃO encontrada')
    
    # Teste 4: Chamada final_normalization
    if re.search(r'final_audio\s*=\s*final_normalization\s*\(\s*final_audio', source_code):
        tests_passed.append('✅ PASS — Chamada final_normalization(final_audio) presente')
    else:
        tests_failed.append('❌ FAIL — Chamada final_normalization NÃO encontrada')
    
    # Teste 5: Parâmetro cutoff=7500
    if 'cutoff=7500' in source_code:
        tests_passed.append('✅ PASS — Parâmetro cutoff=7500 correto')
    else:
        tests_failed.append('❌ FAIL — Parâmetro cutoff=7500 não encontrado')
    
    # Teste 6: Parâmetro target_peak=0.95
    if 'target_peak=0.95' in source_code:
        tests_passed.append('✅ PASS — Parâmetro target_peak=0.95 correto')
    
    # Teste 7: Posição correta (após deess_output, antes de sf.write)
    lines = target_cell['source']
    deess_line_idx = None
    lowpass_line_idx = None
    sfwrite_line_idx = None
    
    for i, line in enumerate(lines):
        if 'final_audio = deess_output' in line:
            deess_line_idx = i
        if 'final_audio = apply_lowpass' in line:
            lowpass_line_idx = i
        if 'sf.write(OUTPUT_PATH,' in line:
            sfwrite_line_idx = i
    
    if all([deess_line_idx is not None, lowpass_line_idx is not None, sfwrite_line_idx is not None]):
        if deess_line_idx < lowpass_line_idx < sfwrite_line_idx:
            tests_passed.append('✅ PASS — Ordem correta (deess → lowpass → sf.write)')
        else:
            tests_failed.append('❌ FAIL — Ordem incorreta das operações')
    
    # Exibir resultados
    print('=' * 60)
    print('TESTE — CORREÇÃO 8: Pós-processamento Final')
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
    success = test_correcao_8()
    sys.exit(0 if success else 1)
