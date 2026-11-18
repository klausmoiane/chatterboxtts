#!/usr/bin/env python3
"""Script para aplicar CORREÇÃO 4 — De-esser Dinâmico no Output (Opcional)"""

import json
import sys

def apply_correction_4():
    notebook_path = 'local-gpu.ipynb'
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar a célula de concatenação (Cell 9 - última célula atual)
    # Precisamos inserir código ANTES de sf.write()
    concat_cell_idx = None
    for idx, cell in enumerate(nb['cells']):
        source_text = ''.join(cell['source'])
        if 'crossfade_concat' in source_text and 'sf.write(OUTPUT_PATH' in source_text:
            concat_cell_idx = idx
            break
    
    if concat_cell_idx is None:
        print('❌ FAIL — Célula de concatenação com sf.write() não encontrada')
        return False
    
    concat_cell = nb['cells'][concat_cell_idx]
    source = concat_cell['source']
    
    # Verificar se já foi aplicado
    source_text = ''.join(source)
    if 'deess_output' in source_text:
        print('⚠️  De-esser de output já está presente. Nenhuma alteração feita.')
        return False
    
    # Código do de-esser de output
    deesser_output_code = [
        '\n',
        '# De-esser dinâmico no output final\n',
        'def deess_output(wav, sr, threshold_db=-18.0, ratio=3.0, freq_low=6000, freq_high=10000):\n',
        '    from scipy.signal import butter, sosfilt\n',
        '    sos = butter(4, [freq_low, freq_high], btype=\'bandpass\', fs=sr, output=\'sos\')\n',
        '    band = sosfilt(sos, wav)\n',
        '    frame = int(sr * 0.01)\n',
        '    gain_env = np.ones_like(wav)\n',
        '    threshold_linear = 10.0 ** (threshold_db / 20.0)\n',
        '    for i in range(0, len(wav) - frame, frame):\n',
        '        rms = np.sqrt(np.mean(band[i:i+frame]**2))\n',
        '        if rms > threshold_linear:\n',
        '            reduction = threshold_linear / (rms * ratio)\n',
        '            reduction = max(reduction, 0.25)\n',
        '            gain_env[i:i+frame] = reduction\n',
        '    return (wav - band + band * gain_env).astype(np.float32)\n',
        'final_audio = deess_output(final_audio, SR)\n',
        'print(\'De-esser dinâmico aplicado ao output final\')\n',
        '\n'
    ]
    
    # Procurar a linha de sf.write e inserir ANTES dela
    new_source = []
    inserted = False
    
    for i, line in enumerate(source):
        # Inserir antes de sf.write(OUTPUT_PATH, ...)
        if 'sf.write(OUTPUT_PATH,' in line and not inserted:
            new_source.extend(deesser_output_code)
            inserted = True
        
        new_source.append(line)
    
    if not inserted:
        print('❌ FAIL — Linha sf.write(OUTPUT_PATH, ...) não encontrada')
        return False
    
    concat_cell['source'] = new_source
    
    # Salvar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print('✅ CORREÇÃO 4 aplicada com sucesso!')
    print(f'   - De-esser dinâmico adicionado à Cell {concat_cell_idx}')
    print('   - Posição: após crossfade_concat, antes de sf.write()')
    print('   - Parâmetros: threshold=-18dB, ratio=3.0, banda 6-10kHz')
    return True

if __name__ == '__main__':
    try:
        success = apply_correction_4()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'❌ Erro ao aplicar correção: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)
