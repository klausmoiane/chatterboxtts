#!/usr/bin/env python3
"""
Teste para CORREÇÃO 5 — Ajustar redução de ruído
Verifica se os parâmetros de noisereduce foram ajustados corretamente
"""

import json
import sys
import os
import re

def test_correcao_5():
    """Testa se a correção 5 foi aplicada corretamente"""
    
    notebook_path = 'kaggle_runner/local-gpu.ipynb'
    
    if not os.path.exists(notebook_path):
        print('❌ FAIL — Notebook não encontrado:', notebook_path)
        return False
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Verificar Cell 4 (índice 4)
    if len(nb['cells']) < 5:
        print('❌ FAIL — Notebook não possui Cell 4')
        return False
    
    cell4 = nb['cells'][4]
    source_code = ''.join(cell4['source'])
    
    # Testes
    tests_passed = []
    tests_failed = []
    
    # Teste 1: prop_decrease=0.2
    if re.search(r'prop_decrease\s*=\s*0\.2\b', source_code):
        tests_passed.append('✅ PASS — prop_decrease=0.2 (reduzido de 0.8)')
    else:
        tests_failed.append('❌ FAIL — prop_decrease NÃO é 0.2')
    
    # Teste 2: Janela de ruído aumentada para 1.0s
    if re.search(r'ref_sr\s*\*\s*1\.0', source_code):
        tests_passed.append('✅ PASS — Janela de ruído = 1.0s (aumentada de 0.5s)')
    else:
        tests_failed.append('❌ FAIL — Janela de ruído NÃO é 1.0s')
    
    # Teste 3: Verificar que valores antigos não existem
    if re.search(r'prop_decrease\s*=\s*0\.8\b', source_code):
        tests_failed.append('❌ FAIL — prop_decrease=0.8 ainda presente')
    else:
        tests_passed.append('✅ PASS — Valor antigo prop_decrease=0.8 removido')
    
    if re.search(r'ref_sr\s*\*\s*0\.5', source_code):
        tests_failed.append('❌ FAIL — Janela 0.5s ainda presente')
    else:
        tests_passed.append('✅ PASS — Valor antigo janela 0.5s removido')
    
    # Teste 4: noisereduce ainda está presente (não foi removido)
    if 'nr.reduce_noise' in source_code:
        tests_passed.append('✅ PASS — noisereduce mantido (não removido)')
    else:
        tests_failed.append('❌ FAIL — noisereduce foi removido (deveria estar presente)')
    
    # Exibir resultados
    print('=' * 60)
    print('TESTE — CORREÇÃO 5: Ajustar Redução de Ruído')
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
    success = test_correcao_5()
    sys.exit(0 if success else 1)
