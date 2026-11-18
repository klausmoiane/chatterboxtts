#!/usr/bin/env python3
"""
Teste para CORREÇÃO 4 — De-esser Dinâmico no Output (Opcional)
Verifica se o de-esser dinâmico foi adicionado ao output final
"""

import json
import sys
import os
import re

def test_correcao_4():
    """Testa se a correção 4 foi aplicada corretamente"""
    
    notebook_path = 'kaggle_runner/local-gpu.ipynb'
    
    if not os.path.exists(notebook_path):
        print('❌ FAIL — Notebook não encontrado:', notebook_path)
        return False
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar a célula com crossfade_concat e sf.write
    target_cell = None
    cell_index = None
    
    for idx, cell in enumerate(nb['cells']):
        source_text = ''.join(cell['source'])
        if 'crossfade_concat' in source_text and 'sf.write(OUTPUT_PATH' in source_text:
            target_cell = cell
            cell_index = idx
            break
    
    if target_cell is None:
        print('❌ FAIL — Célula de concatenação com sf.write() não encontrada')
        return False
    
    source_code = ''.join(target_cell['source'])
    
    # Testes
    tests_passed = []
    tests_failed = []
    
    # Teste 1: Função deess_output presente
    if 'def deess_output(' in source_code:
        tests_passed.append('✅ PASS — Função deess_output presente no notebook')
    else:
        tests_failed.append('❌ FAIL — Função deess_output NÃO encontrada')
    
    # Teste 2: Chamada final_audio = deess_output(final_audio, SR)
    if re.search(r'final_audio\s*=\s*deess_output\s*\(\s*final_audio\s*,\s*SR\s*\)', source_code):
        tests_passed.append('✅ PASS — Chamada final_audio = deess_output(final_audio, SR) presente')
    else:
        tests_failed.append('❌ FAIL — Chamada de deess_output(final_audio, SR) NÃO encontrada')
    
    # Teste 3: Posição correta (deess_output antes de sf.write)
    lines = target_cell['source']
    deess_line_idx = None
    sfwrite_line_idx = None
    
    for i, line in enumerate(lines):
        if 'final_audio = deess_output' in line:
            deess_line_idx = i
        if 'sf.write(OUTPUT_PATH,' in line:
            sfwrite_line_idx = i
    
    if deess_line_idx is not None and sfwrite_line_idx is not None:
        if deess_line_idx < sfwrite_line_idx:
            tests_passed.append('✅ PASS — Posição correta (após crossfade, antes de sf.write)')
        else:
            tests_failed.append('❌ FAIL — Posição incorreta (deess_output DEPOIS de sf.write)')
    elif deess_line_idx is None:
        tests_failed.append('❌ FAIL — Linha de chamada deess_output não encontrada')
    
    # Teste 4: Parâmetros corretos (threshold, ratio, freq_low, freq_high)
    if 'threshold_db=-18.0' in source_code or 'threshold_db = -18.0' in source_code:
        tests_passed.append('✅ PASS — Parâmetro threshold_db=-18.0 correto')
    else:
        tests_failed.append('❌ FAIL — Parâmetro threshold_db incorreto ou faltando')
    
    if 'ratio=3.0' in source_code or 'ratio = 3.0' in source_code:
        tests_passed.append('✅ PASS — Parâmetro ratio=3.0 correto')
    
    # Teste 5: Import scipy (já deve estar presente de Correção 1, mas verificar)
    if 'from scipy.signal import butter, sosfilt' in source_code:
        tests_passed.append('✅ PASS — Import scipy.signal presente')
    
    # Exibir resultados
    print('=' * 60)
    print('TESTE — CORREÇÃO 4: De-esser Dinâmico no Output')
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
    success = test_correcao_4()
    sys.exit(0 if success else 1)
