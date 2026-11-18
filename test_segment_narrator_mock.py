"""
test_segment_narrator_mock.py
==============================
Suite de testes mock para verificar a integridade da configuração do narrador,
suporte a segmentos de áudio (offset/duration) e prioridades de condicionais
sem carregar os pesos pesados do modelo real.
"""

import sys
import os
import json
import tempfile
import unittest.mock as mock
import numpy as np

# Path setup
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from narrator_pipeline import NarratorPipeline

# Cores para o terminal
GREEN  = "\033[92m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

_results = {"passed": 0, "failed": 0}

def section(title: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")

def run_test(name: str, fn):
    try:
        fn()
        print(f"  {GREEN}[PASS]{RESET}  {name}")
        _results["passed"] += 1
    except Exception as e:
        print(f"  {RED}[FAIL]{RESET}  {name} -> {type(e).__name__}: {e}")
        _results["failed"] += 1


# ── 1. TESTES DE CONFIGURAÇÃO DO JSON DO NARRADOR ─────────────────────────
section("1. PRESET JSON — Narrator EN (Documentary)")

def t_preset_json_exists():
    path = os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json")
    assert os.path.exists(path), "Arquivo JSON do preset não existe."

def t_preset_json_has_segment_fields():
    path = os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data.get("audio_ref") is not None, "audio_ref não pode ser None."
    assert data.get("audio_ref_start_sec") == 163, f"audio_ref_start_sec esperado: 163, obtido: {data.get('audio_ref_start_sec')}"
    assert data.get("audio_ref_end_sec") == 192, f"audio_ref_end_sec esperado: 192, obtido: {data.get('audio_ref_end_sec')}"
    assert data.get("conds_pt") is None, "conds_pt deve ser None neste modo."

def t_preset_audio_file_accessible():
    path = os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    audio_path = data.get("audio_ref")
    assert os.path.exists(audio_path), f"Arquivo MP3 de referência não encontrado em: {audio_path}"


# ── 2. TESTES DE FLUXO DE SEGMENTO E MOCK PIPELINE ───────────────────────
section("2. PIPELINE MOCK — Carregamento de Segmento de Áudio")

def t_pipeline_extracts_segment_correctly():
    """Verifica se o pipeline chama librosa.load com o offset e duration corretos se o preset contiver os campos."""
    preset_data = {
        "name": "Test Segment Narrator",
        "audio_ref": os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json"), # Apenas um arquivo existente para passar pelo os.path.exists
        "audio_ref_start_sec": 163,
        "audio_ref_end_sec": 192,
    }
    
    pipeline = NarratorPipeline()
    mock_model = mock.MagicMock()
    mock_model.sr = 24000
    mock_model.conds = mock.MagicMock()
    pipeline.model = mock_model

    # Mock librosa.load e soundfile.write para evitar I/O real e leitura de áudio pesado
    with mock.patch("librosa.load", return_value=(np.zeros(24000), 24000)) as mock_librosa, \
         mock.patch("soundfile.write") as mock_sf_write:
        
        pipeline._load_conds_for_generation(preset=preset_data, audio_prompt_path=None, norm_loudness=True)
        
        # Verificar se librosa.load recebeu offset=163.0 e duration=29.0
        mock_librosa.assert_called_once()
        _, kwargs = mock_librosa.call_args
        assert kwargs.get("offset") == 163.0, f"Offset incorreto: {kwargs.get('offset')}"
        assert kwargs.get("duration") == 29.0, f"Duration incorreto: {kwargs.get('duration')}"
        
        # Verificar se prepare_conditionals foi chamado no modelo
        mock_model.prepare_conditionals.assert_called_once()

def t_pipeline_fallback_full_audio_ref():
    """Verifica se utiliza o arquivo completo caso audio_ref_start_sec não seja fornecido."""
    preset_data = {
        "name": "Full Ref Narrator",
        "audio_ref": os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json"),
    }
    
    pipeline = NarratorPipeline()
    mock_model = mock.MagicMock()
    mock_model.sr = 24000
    mock_model.conds = mock.MagicMock()
    pipeline.model = mock_model

    with mock.patch("librosa.load") as mock_librosa:
        pipeline._load_conds_for_generation(preset=preset_data, audio_prompt_path=None, norm_loudness=True)
        
        # librosa.load NÃO deve ser chamado pois não há recorte de segmento
        mock_librosa.assert_not_called()
        mock_model.prepare_conditionals.assert_called_once_with(preset_data["audio_ref"], norm_loudness=True)

def t_pipeline_priority_conds_pt_over_audio_ref():
    """Verifica se conds_pt tem prioridade total se estiver presente no preset."""
    # Criar um arquivo .pt temporário
    tmp_pt = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp_pt.close()

    try:
        preset_data = {
            "name": "PT Priority Narrator",
            "conds_pt": tmp_pt.name,
            "audio_ref": "ignored.mp3",
            "audio_ref_start_sec": 10,
            "audio_ref_end_sec": 20,
        }

        pipeline = NarratorPipeline()
        mock_model = mock.MagicMock()
        mock_model.sr = 24000
        pipeline.model = mock_model

        with mock.patch("chatterbox.tts_turbo.Conditionals.load") as mock_conds_load, \
             mock.patch("librosa.load") as mock_librosa:
            
            mock_conds_obj = mock.MagicMock()
            mock_conds_load.return_value = mock_conds_obj
            
            pipeline._load_conds_for_generation(preset=preset_data, audio_prompt_path=None, norm_loudness=True)
            
            # Deve carregar o .pt e NUNCA chamar librosa ou prepare_conditionals
            mock_conds_load.assert_called_once_with(tmp_pt.name, map_location=mock.ANY)
            mock_librosa.assert_not_called()
            mock_model.prepare_conditionals.assert_not_called()
    finally:
        os.unlink(tmp_pt.name)


# ── EXECUÇÃO DA SUITE DE TESTES ───────────────────────────────────────────
if __name__ == "__main__":
    run_test("narrator_en_documentary.json existe", t_preset_json_exists)
    run_test("Campos de segmento (163s-192s) presentes e validados no JSON", t_preset_json_has_segment_fields)
    run_test("Arquivo MP3 de referência acessível no disco", t_preset_audio_file_accessible)
    
    run_test("Pipeline extrai segmento 163s-192s via librosa.load (Mock)", t_pipeline_extracts_segment_correctly)
    run_test("Pipeline usa áudio completo se sem timestamps (Mock)", t_pipeline_fallback_full_audio_ref)
    run_test("Prioridade de conds_pt sobre audio_ref mantida (Mock)", t_pipeline_priority_conds_pt_over_audio_ref)
    
    total = _results["passed"] + _results["failed"]
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  RESULTADO: {total} testes  |  {GREEN}{_results['passed']} passou{RESET}  |  {RED}{_results['failed']} falhou{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")
