#!/usr/bin/env python3
"""
Teste para CORREÇÃO 1 — De-esser na Referência
Verifica se o de-esser foi corretamente adicionado à Cell 4 do notebook
"""

import json
import sys
import os

def test_correcao_1():
    """Testa se a correção 1 foi aplicada corretamente"""
    
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
    
    # Teste 1: Função apply_deesser presente
    if 'def apply_deesser' in source_code:
        tests_passed.append('✅ PASS — Função apply_deesser presente na Cell 4')
    else:
        tests_failed.append('❌ FAIL — Função apply_deesser NÃO encontrada na Cell 4')
    
    # Teste 2: Parâmetros corretos
    if 'freq_low=6000' in source_code and 'freq_high=10000' in source_code and 'attenuation_db=4.0' in source_code:
        tests_passed.append('✅ PASS — Parâmetros corretos (freq_low=6000, freq_high=10000, attenuation_db=4.0)')
    else:
        tests_failed.append('❌ FAIL — Parâmetros incorretos ou faltando')
    
    # Teste 3: Import scipy.signal presente
    if 'from scipy.signal import butter, sosfilt' in source_code:
        tests_passed.append('✅ PASS — Import scipy.signal presente')
    else:
        tests_failed.append('❌ FAIL — Import scipy.signal NÃO encontrado')
    
    # Teste 4: Chamada da função presente
    if 'ref_wav = apply_deesser(ref_wav' in source_code:
        tests_passed.append('✅ PASS — Chamada de apply_deesser presente')
    else:
        tests_failed.append('❌ FAIL — Chamada de apply_deesser NÃO encontrada')
    
    # Teste 5: Print de confirmação
    if 'De-esser aplicado' in source_code:
        tests_passed.append('✅ PASS — Print de confirmação presente')
    else:
        tests_failed.append('❌ FAIL — Print de confirmação NÃO encontrado')
    
    # Exibir resultados
    print('=' * 60)
    print('TESTE — CORREÇÃO 1: De-esser na Referência')
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
    success = test_correcao_1()
    sys.exit(0 if success else 1)
