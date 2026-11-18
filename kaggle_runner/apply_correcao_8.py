#!/usr/bin/env python3
"""Script para aplicar CORREÇÃO 8 — Pós-processamento final: Lowpass + Normalização"""

import json
import sys

def apply_correction_8():
    notebook_path = 'local-gpu.ipynb'
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar a célula de concatenação (Cell 9 - última célula atual)
    concat_cell_idx = None
    for idx, cell in enumerate(nb['cells']):
        source_text = ''.join(cell['source'])
        if 'deess_output' in source_text and 'sf.write(OUTPUT_PATH' in source_text:
            concat_cell_idx = idx
            break
    
    if concat_cell_idx is None:
        print('❌ FAIL — Célula de concatenação com deess_output e sf.write() não encontrada')
        return False
    
    concat_cell = nb['cells'][concat_cell_idx]
    source = concat_cell['source']
    
    # Verificar se já foi aplicado
    source_text = ''.join(source)
    if 'apply_lowpass' in source_text or 'final_normalization' in source_text:
        print('⚠️  Pós-processamento final já está presente. Nenhuma alteração feita.')
        return False
    
    # Código do pós-processamento final
    postproc_code = [
        '\n',
        '# Pós-processamento final: Lowpass + Normalização\n',
        'def apply_lowpass(wav, sr, cutoff=7500):\n',
        '    """Filtro lowpass suave para remover crueza de alta frequência."""\n',
        '    from scipy.signal import butter, sosfilt\n',
        '    sos = butter(4, cutoff, btype=\'lowpass\', fs=sr, output=\'sos\')\n',
        '    return sosfilt(sos, wav).astype(np.float32)\n',
        '\n',
        'def final_normalization(wav, target_peak=0.95):\n',
        '    """Normalização final para target_peak sem clipping."""\n',
        '    peak = np.abs(wav).max()\n',
        '    if peak > 0:\n',
        '        wav = wav * (target_peak / peak)\n',
        '    return np.clip(wav, -1.0, 1.0).astype(np.float32)\n',
        '\n',
        'final_audio = apply_lowpass(final_audio, SR, cutoff=7500)\n',
        'print(\'Lowpass 7500Hz aplicado ao output final\')\n',
        'final_audio = final_normalization(final_audio, target_peak=0.95)\n',
        'print(\'Normalização final aplicada (peak=0.95)\')\n',
        '\n'
    ]
    
    # Procurar a linha de sf.write e inserir ANTES dela
    new_source = []
    inserted = False
    
    for i, line in enumerate(source):
        # Inserir antes de sf.write(OUTPUT_PATH, ...)
        if 'sf.write(OUTPUT_PATH,' in line and not inserted:
            new_source.extend(postproc_code)
            inserted = True
        
        new_source.append(line)
    
    if not inserted:
        print('❌ FAIL — Linha sf.write(OUTPUT_PATH, ...) não encontrada')
        return False
    
    concat_cell['source'] = new_source
    
    # Salvar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print('✅ CORREÇÃO 8 aplicada com sucesso!')
    print(f'   - Pós-processamento final adicionado à Cell {concat_cell_idx}')
    print('   - Lowpass 7500Hz: remove crueza de alta frequência')
    print('   - Normalização final: peak=0.95 sem clipping')
    return True

if __name__ == '__main__':
    try:
        success = apply_correction_8()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'❌ Erro ao aplicar correção: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
