#!/usr/bin/env python3
"""
Teste para CORREÇÃO 2 — Target LUFS alinhado com o modelo (-27 dB)
Verifica se o target de normalização foi alterado de -23 para -27 LUFS
"""

import json
import sys
import os
import re

def test_correcao_2():
    """Testa se a correção 2 foi aplicada corretamente"""
    
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
    
    # Teste 1: Target LUFS é -27.0
    pattern_27 = r'gain_linear\s*=\s*10\.0\s*\*\*\s*\(\(\s*-27\.0\s*-\s*loudness_cur\s*\)\s*/\s*20\.0\s*\)'
    if re.search(pattern_27, source_code):
        tests_passed.append('✅ PASS — Target LUFS é -27.0 na Cell 4')
    else:
        tests_failed.append('❌ FAIL — Target LUFS NÃO é -27.0 (esperado na linha gain_linear)')
    
    # Teste 2: Não há referência ativa a -23.0 LUFS no código de normalização
    # (comentários podem ter -23, mas o código ativo deve ter -27)
    lines_with_23 = []
    for line in cell4['source']:
        if '-23.0' in line and 'gain_linear' in line and not line.strip().startswith('#'):
            lines_with_23.append(line.strip())
    
    if len(lines_with_23) == 0:
        tests_passed.append('✅ PASS — Sem referência a -23.0 LUFS no código ativo')
    else:
        tests_failed.append(f'❌ FAIL — Encontrada referência a -23.0 LUFS no código ativo: {lines_with_23[0]}')
    
    # Teste 3: Comentário atualizado (opcional, não crítico)
    if '-27 LUFS' in source_code or '-27.0 LUFS' in source_code:
        tests_passed.append('✅ PASS — Comentário atualizado para -27 LUFS')
    else:
        # Não é falha crítica
        tests_passed.append('⚠️  INFO — Comentário não atualizado (não crítico)')
    
    # Exibir resultados
    print('=' * 60)
    print('TESTE — CORREÇÃO 2: Target LUFS -27 dB')
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
    success = test_correcao_2()
    sys.exit(0 if success else 1)
