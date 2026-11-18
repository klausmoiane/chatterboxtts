#!/usr/bin/env python3
"""Script para aplicar CORREÇÃO 1 — De-esser na Referência"""

import json
import sys

def apply_correction_1():
    notebook_path = 'local-gpu.ipynb'
    
    # Ler notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Encontrar Cell 4 (índice 4 no array cells)
    cell4 = nb['cells'][4]
    
    # Código do de-esser a inserir
    deesser_code = [
        '\n',
        '# 2.3.1 De-esser na banda de sibilância (6-10 kHz)\n',
        'from scipy.signal import butter, sosfilt\n',
        'def apply_deesser(wav, sr, freq_low=6000, freq_high=10000, attenuation_db=4.0):\n',
        '    sos_bp = butter(4, [freq_low, freq_high], btype=\'bandpass\', fs=sr, output=\'sos\')\n',
        '    band = sosfilt(sos_bp, wav)\n',
        '    gain = 10.0 ** (-attenuation_db / 20.0)\n',
        '    wav_deessed = wav - band + band * gain\n',
        '    return wav_deessed.astype(np.float32)\n',
        'ref_wav = apply_deesser(ref_wav, ref_sr, freq_low=6000, freq_high=10000, attenuation_db=4.0)\n',
        'print(f\'De-esser aplicado: banda 6–10 kHz atenuada em 4 dB\')\n'
    ]
    
    # Procurar a linha do reduce_noise e inserir após ela
    source = cell4['source']
    new_source = []
    
    for i, line in enumerate(source):
        new_source.append(line)
        
        # Após a linha do reduce_noise, inserir o de-esser
        if 'reduce_noise' in line and 'prop_decrease=0.8' in line:
            # Verificar se já não foi aplicado
            if i + 1 < len(source) and 'apply_deesser' in source[i + 1]:
                print('⚠️  De-esser já está presente na Cell 4. Nenhuma alteração feita.')
                return False
            
            new_source.extend(deesser_code)
    
    cell4['source'] = new_source
    
    # Salvar notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print('✅ CORREÇÃO 1 aplicada com sucesso!')
    print('   - De-esser adicionado à Cell 4')
    print('   - Banda: 6-10 kHz')
    print('   - Atenuação: 4 dB')
    return True

if __name__ == '__main__':
    try:
        success = apply_correction_1()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f'❌ Erro ao aplicar correção: {e}')
        sys.exit(1)
