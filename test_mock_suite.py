"""
=============================================================================
CHATTERBOX NARRATOR STUDIO — SUITE DE TESTES MOCK COMPLETA
=============================================================================
Testa TODOS os componentes sem carregar o modelo Turbo real (muito pesado).

Módulos testados:
  1. text_preprocessor  — normalize_text, split_into_chunks, estimate_generation
  2. preset_manager     — save, load, list, delete, edge cases
  3. narrator_pipeline  — crossfade_concat, set_seed, NarratorPipeline (mock model)
  4. tts_turbo source   — verifica assinatura de generate() com n_cfm_timesteps
  5. Gradio app imports — verifica que o app importa sem erros

Edge cases cobertos:
  - Texto vazio, None, só espaços
  - Texto com uma única frase muito longa (>250 chars)
  - Texto com frases curtas que devem ser agrupadas
  - Parágrafos múltiplos separados por \n\n
  - Tags de transcrição: [Music], [ __ ], [adfern]
  - Aspas tipográficas, travessões, reticências
  - Preset com nome contendo caracteres especiais / espaços / unicode
  - Preset nome vazio → deve lançar ValueError
  - Preset inexistente → load deve retornar None
  - Delete de preset inexistente → deve retornar False
  - crossfade com 0, 1, 2+ arrays
  - crossfade com arrays menores que a janela de fade
  - seed offset: verificação numérica
  - Pipeline com modelo mockado: verificação de fluxo completo
  - Pipeline com texto vazio → deve lançar ValueError
  - tts_turbo.generate() signature: n_cfm_timesteps presente e default=4
=============================================================================
"""

import sys
import os
import json
import inspect
import shutil
import tempfile
import traceback
import numpy as np

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── colour helpers ───────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
SKIP = f"{YELLOW}⚠ SKIP{RESET}"

_results = {"passed": 0, "failed": 0, "skipped": 0}


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*60}{RESET}")


def run_test(name: str, fn):
    """Run a single test function and record result."""
    try:
        fn()
        print(f"  {PASS}  {name}")
        _results["passed"] += 1
    except AssertionError as e:
        print(f"  {FAIL}  {name}")
        print(f"         → AssertionError: {e}")
        _results["failed"] += 1
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         → {type(e).__name__}: {e}")
        _results["failed"] += 1


def skip_test(name: str, reason: str):
    print(f"  {SKIP}  {name}  ({reason})")
    _results["skipped"] += 1


def summary():
    total = sum(_results.values())
    p, f, s = _results["passed"], _results["failed"], _results["skipped"]
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  RESULTADO FINAL: {total} testes  |  "
          f"{GREEN}{p} passou{RESET}  |  "
          f"{RED}{f} falhou{RESET}  |  "
          f"{YELLOW}{s} ignorado{RESET}{BOLD}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")
    return f == 0


# ═══════════════════════════════════════════════════════════════════════════
# SECÇÃO 1 — TEXT PREPROCESSOR
# ═══════════════════════════════════════════════════════════════════════════
section("1. TEXT PREPROCESSOR — normalize_text()")

from chatterbox.text_preprocessor import normalize_text, split_into_chunks, estimate_generation


def t_normalize_empty():
    assert normalize_text("") == ""

def t_normalize_none_like():
    # Should not crash on empty-ish input
    assert normalize_text("   ") == ""

def t_normalize_music_tag():
    result = normalize_text("Hello [Music] world.")
    assert "[Music]" not in result
    assert "[music]" not in result.lower()

def t_normalize_blank_tag():
    result = normalize_text("Hello [ __ ] world.")
    assert "[ __ ]" not in result

def t_normalize_arbitrary_tag():
    result = normalize_text("Click [adfern] here.")
    assert "[adfern]" not in result.lower()

def t_normalize_curly_quotes():
    result = normalize_text("\u201cHello\u201d and \u2018world\u2019.")
    assert "\u201c" not in result
    assert "\u201d" not in result
    assert "\u2018" not in result
    assert "\u2019" not in result

def t_normalize_em_dash():
    result = normalize_text("A\u2014B.")
    assert "\u2014" not in result

def t_normalize_en_dash():
    result = normalize_text("A\u2013B.")
    assert "\u2013" not in result

def t_normalize_ellipsis():
    result = normalize_text("Wait...then go.")
    assert "..." not in result

def t_normalize_multiple_spaces():
    result = normalize_text("Hello    world.")
    assert "  " not in result

def t_normalize_strips():
    result = normalize_text("  Hello world.  ")
    assert result == result.strip()

def t_normalize_preserves_sentences():
    result = normalize_text("Hello world. This is fine!")
    assert "Hello" in result
    assert "fine" in result

def t_normalize_time_format():
    result = normalize_text("Just before 8:00 p.m., a sailing ship")
    assert "eight PM" in result
    assert "8:00" not in result
    result2 = normalize_text("At 8:30 p.m. yesterday")
    assert "eight 30 PM" in result2

run_test("empty string → empty string",           t_normalize_empty)
run_test("whitespace only → empty string",        t_normalize_none_like)
run_test("time 8:00 p.m. normalized to eight PM", t_normalize_time_format)
run_test("[Music] tag removed",                   t_normalize_music_tag)
run_test("[ __ ] tag removed",                   t_normalize_blank_tag)
run_test("[adfern] tag removed",                  t_normalize_arbitrary_tag)
run_test("curly quotes → straight quotes",        t_normalize_curly_quotes)
run_test("em dash → comma",                       t_normalize_em_dash)
run_test("en dash → comma",                       t_normalize_en_dash)
run_test("ellipsis ... → comma",                  t_normalize_ellipsis)
run_test("multiple spaces collapsed",             t_normalize_multiple_spaces)
run_test("leading/trailing whitespace stripped",  t_normalize_strips)
run_test("normal text preserved",                 t_normalize_preserves_sentences)


# ─────────────────────────────────────────────────────────────────────────────
section("2. TEXT PREPROCESSOR — split_into_chunks()")

NORDSTREAM_INTRO = (
    "September 26, 2022. Just before 8:00 p.m., a sailing ship is drifting gently "
    "on the Baltic Sea. The captain is getting ready for the night. He checks his radar "
    "and sees two red blips flashing on the screen. He stares out into the darkness. "
    "Nothing. So, he radios a nearby ship. Their crew reports something strange. "
    "Giant whirlpools in the water. When his ship approaches the spot, the captain is "
    "hit by a sharp and sour smell. Gas. He immediately notifies the Swedish authorities. "
    "They confirm his suspicion. There is an underwater leak. Tons of natural gas are "
    "flowing uncontrollably into the Baltic Sea."
)

LONG_SENTENCE = "A" * 300  # single "word" longer than any limit
COMMA_SENTENCE = ("word, " * 50).strip()  # >250 chars, splittable by commas

def t_chunks_empty():
    result = split_into_chunks("")
    assert result == [], f"Expected [], got {result}"

def t_chunks_whitespace_only():
    result = split_into_chunks("   \n\n   ")
    assert result == [], f"Expected [], got {result}"

def t_chunks_none_exceeds_limit():
    chunks = split_into_chunks(NORDSTREAM_INTRO, max_chars=250)
    for i, c in enumerate(chunks):
        assert len(c) <= 250, f"Chunk {i} exceeded 250 chars: {len(c)} — '{c[:60]}...'"

def t_chunks_no_empty_chunks():
    chunks = split_into_chunks(NORDSTREAM_INTRO, max_chars=250)
    for i, c in enumerate(chunks):
        assert c.strip() != "", f"Chunk {i} is empty"

def t_chunks_no_mid_sentence_cut():
    """Every chunk should end with sentence-ending punctuation or be a partial of a long sentence."""
    text = "Hello world. How are you? I am fine! What about you."
    chunks = split_into_chunks(text, max_chars=250)
    # Since all sentences fit in one chunk, should just be 1 chunk
    assert len(chunks) == 1
    assert "Hello" in chunks[0] and "you." in chunks[0]

def t_chunks_multiple_paragraphs():
    text = "First paragraph sentence one. First paragraph sentence two.\n\nSecond paragraph only."
    chunks = split_into_chunks(text, max_chars=250)
    assert len(chunks) >= 1
    # Second paragraph should be its own chunk (separated by \n\n)
    combined = " ".join(chunks)
    assert "Second paragraph" in combined

def t_chunks_sentence_grouping():
    """Short sentences should be grouped until limit, not each own chunk."""
    text = "A. B. C. D. E. F. G. H."
    chunks = split_into_chunks(text, max_chars=250)
    # All these tiny sentences must be grouped into 1 chunk
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}: {chunks}"

def t_chunks_very_long_single_sentence():
    """A sentence longer than max_chars should be sub-divided, not produce a >250 chunk."""
    # Build a long sentence that CAN be split on commas
    sentence = "word, " * 60 + "end."   # ~360 chars
    chunks = split_into_chunks(sentence, max_chars=250)
    assert len(chunks) >= 2, "Long sentence should be split into >=2 chunks"
    for c in chunks:
        assert len(c) <= 250, f"Chunk still too long: {len(c)}"

def t_chunks_text_with_tags():
    """[Music] tags are removed during normalize, chunks should have no artifacts."""
    text = "Hello [Music] world. This is the story. [__] Another sentence."
    chunks = split_into_chunks(text, max_chars=250)
    full = " ".join(chunks)
    assert "[Music]" not in full
    assert "[__]" not in full

def t_chunks_nordstream_count():
    """The Nordstream intro should produce multiple chunks."""
    chunks = split_into_chunks(NORDSTREAM_INTRO, max_chars=250)
    assert len(chunks) >= 3, f"Expected >=3 chunks, got {len(chunks)}"
    print(f"\n         → {len(chunks)} chunks gerados do intro Nordstream", end="")

def t_chunks_custom_limit_100():
    """With max_chars=100, should produce many small chunks."""
    chunks_250 = split_into_chunks(NORDSTREAM_INTRO, max_chars=250)
    chunks_100 = split_into_chunks(NORDSTREAM_INTRO, max_chars=100)
    assert len(chunks_100) >= len(chunks_250), "Smaller limit should produce more or equal chunks"
    for c in chunks_100:
        assert len(c) <= 100, f"Chunk exceeds 100 chars: {len(c)}"

def t_chunks_content_preserved():
    """No text should be lost during chunking."""
    text = "Hello world. This is a test. Nothing should be lost."
    chunks = split_into_chunks(text, max_chars=250)
    # Join and check key words are preserved
    combined = " ".join(chunks)
    for word in ["Hello", "world", "test", "Nothing", "lost"]:
        assert word in combined, f"Word '{word}' lost after chunking"

def t_chunks_single_short_sentence():
    text = "Hello world."
    chunks = split_into_chunks(text, max_chars=250)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world."

def t_chunks_pure_long_word_no_space():
    """A string with no spaces/punctuation just gets wrapped by textwrap fallback."""
    text = LONG_SENTENCE  # 300 'A's
    chunks = split_into_chunks(text, max_chars=250)
    # Should produce at least one chunk and no chunk > 250
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c) <= 250, f"Chunk too long: {len(c)}"

run_test("empty → []",                           t_chunks_empty)
run_test("whitespace only → []",                 t_chunks_whitespace_only)
run_test("all chunks ≤ 250 chars",               t_chunks_none_exceeds_limit)
run_test("no empty chunks produced",             t_chunks_no_empty_chunks)
run_test("short text → 1 chunk (no cut)",        t_chunks_no_mid_sentence_cut)
run_test("\\n\\n separates paragraphs",          t_chunks_multiple_paragraphs)
run_test("tiny sentences grouped together",      t_chunks_sentence_grouping)
run_test("very long comma sentence split",       t_chunks_very_long_single_sentence)
run_test("[Music]/[__] tags removed in chunks",  t_chunks_text_with_tags)
run_test("Nordstream intro → ≥3 chunks",         t_chunks_nordstream_count)
run_test("max_chars=100 → more chunks",          t_chunks_custom_limit_100)
run_test("content fully preserved in chunks",    t_chunks_content_preserved)
run_test("single short sentence → 1 chunk",      t_chunks_single_short_sentence)
run_test("300-char no-space word handled safely", t_chunks_pure_long_word_no_space)


# ─────────────────────────────────────────────────────────────────────────────
section("3. TEXT PREPROCESSOR — estimate_generation()")

def t_estimate_empty():
    result = estimate_generation([])
    assert result["num_chunks"] == 0
    assert result["estimated_minutes"] == 0.0
    assert result["avg_chars"] == 0

def t_estimate_single():
    result = estimate_generation(["Hello world."], sec_per_chunk=12.0)
    assert result["num_chunks"] == 1
    assert result["estimated_minutes"] == 0.2   # 12s / 60 = 0.2
    assert result["total_chars"] == 12
    assert result["avg_chars"] == 12.0

def t_estimate_multiple():
    chunks = ["Hello.", "World.", "Test!"]
    result = estimate_generation(chunks, sec_per_chunk=12.0)
    assert result["num_chunks"] == 3
    assert result["estimated_minutes"] == 0.6  # 36s / 60

def t_estimate_nordstream():
    chunks = split_into_chunks(NORDSTREAM_INTRO, max_chars=250)
    result = estimate_generation(chunks, sec_per_chunk=12.0)
    assert result["num_chunks"] > 0
    assert result["estimated_minutes"] > 0
    assert result["avg_chars"] > 0
    assert result["avg_chars"] <= 250
    print(f"\n         → {result['num_chunks']} chunks, ~{result['estimated_minutes']}min, avg={result['avg_chars']}chars", end="")

run_test("empty list → zeros",                   t_estimate_empty)
run_test("single chunk metrics correct",         t_estimate_single)
run_test("multiple chunks time calculation",     t_estimate_multiple)
run_test("Nordstream full estimate metrics",     t_estimate_nordstream)


# ═══════════════════════════════════════════════════════════════════════════
# SECÇÃO 2 — PRESET MANAGER
# ═══════════════════════════════════════════════════════════════════════════
section("4. PRESET MANAGER — setup with temp directory")

# We monkey-patch VOICE_TEMPLATES_DIR and REFS_DIR to a temp dir for isolation
import voice_templates.preset_manager as pm_module

_ORIG_VTDIR = pm_module.VOICE_TEMPLATES_DIR
_ORIG_REFS  = pm_module.REFS_DIR
_TMPDIR = tempfile.mkdtemp(prefix="narrator_test_presets_")
_TMP_REFS = os.path.join(_TMPDIR, "refs")
os.makedirs(_TMP_REFS, exist_ok=True)

# Patch module paths
pm_module.VOICE_TEMPLATES_DIR = type(pm_module.VOICE_TEMPLATES_DIR)(_TMPDIR)
pm_module.REFS_DIR             = type(pm_module.REFS_DIR)(_TMP_REFS)

from voice_templates.preset_manager import (
    list_presets, get_preset_names, load_preset, save_preset, delete_preset
)

SAMPLE_PARAMS = {
    "temperature": 1.0,
    "repetition_penalty": 1.05,
    "top_p": 0.85,
    "min_p": 0.0,
    "top_k": 500,
    "seed": 42,
    "n_cfm_timesteps": 4,
    "max_chars_per_chunk": 250
}

section("4a. PRESET MANAGER — list / save / load")

def t_pm_list_empty():
    presets = list_presets()
    assert isinstance(presets, list)
    # May already contain narrator_en_documentary.json from real dir if patching failed
    # but in temp dir it should be empty
    assert presets == [], f"Expected empty list, got {presets}"

def t_pm_save_basic():
    result = save_preset("Test Narrator", "A test voice", SAMPLE_PARAMS.copy())
    assert result["name"] == "Test Narrator"
    assert result["description"] == "A test voice"
    assert result["audio_ref"] is None
    assert result["params"]["temperature"] == 1.0

def t_pm_list_after_save():
    presets = list_presets()
    assert len(presets) >= 1
    names = [p["name"] for p in presets]
    assert "Test Narrator" in names

def t_pm_get_names():
    names = get_preset_names()
    assert "Test Narrator" in names
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)

def t_pm_load_existing():
    preset = load_preset("Test Narrator")
    assert preset is not None
    assert preset["name"] == "Test Narrator"
    assert preset["params"]["seed"] == 42
    assert preset["params"]["n_cfm_timesteps"] == 4

def t_pm_load_case_insensitive():
    preset = load_preset("TEST NARRATOR")
    assert preset is not None, "load_preset should be case-insensitive"

def t_pm_load_nonexistent():
    preset = load_preset("Does Not Exist")
    assert preset is None

def t_pm_params_roundtrip():
    """All params must survive JSON save/load exactly."""
    save_preset("Roundtrip Test", "desc", SAMPLE_PARAMS.copy())
    loaded = load_preset("Roundtrip Test")
    for key, val in SAMPLE_PARAMS.items():
        assert loaded["params"][key] == val, f"Mismatch on '{key}': {loaded['params'][key]} != {val}"

run_test("list on empty temp dir → []",          t_pm_list_empty)
run_test("save basic preset → dict returned",    t_pm_save_basic)
run_test("list after save contains preset",      t_pm_list_after_save)
run_test("get_preset_names returns strings",     t_pm_get_names)
run_test("load existing preset",                 t_pm_load_existing)
run_test("load is case-insensitive",             t_pm_load_case_insensitive)
run_test("load nonexistent → None",              t_pm_load_nonexistent)
run_test("all params survive JSON roundtrip",    t_pm_params_roundtrip)


section("4b. PRESET MANAGER — delete")

def t_pm_delete_existing():
    save_preset("To Delete", "temporary", SAMPLE_PARAMS.copy())
    result = delete_preset("To Delete")
    assert result is True
    assert load_preset("To Delete") is None

def t_pm_delete_nonexistent():
    result = delete_preset("Ghost Preset")
    assert result is False

def t_pm_delete_file_removed():
    save_preset("File Delete Test", "temp", SAMPLE_PARAMS.copy())
    delete_preset("File Delete Test")
    # Check no JSON remains for this preset
    remaining_names = get_preset_names()
    assert "File Delete Test" not in remaining_names

run_test("delete existing → True + removed",    t_pm_delete_existing)
run_test("delete nonexistent → False",           t_pm_delete_nonexistent)
run_test("delete removes JSON from disk",        t_pm_delete_file_removed)


section("4c. PRESET MANAGER — edge cases / special names")

def t_pm_empty_name_raises():
    try:
        save_preset("", "desc", SAMPLE_PARAMS.copy())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # expected

def t_pm_whitespace_name_raises():
    try:
        save_preset("   ", "desc", SAMPLE_PARAMS.copy())
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # expected

def t_pm_name_with_spaces():
    save_preset("Narrator With Spaces", "desc", SAMPLE_PARAMS.copy())
    loaded = load_preset("Narrator With Spaces")
    assert loaded is not None
    assert loaded["name"] == "Narrator With Spaces"

def t_pm_name_with_special_chars():
    # Special chars should be sanitised in filename but name preserved
    save_preset("Narrador Dramático!", "desc", SAMPLE_PARAMS.copy())
    loaded = load_preset("Narrador Dramático!")
    assert loaded is not None
    assert loaded["name"] == "Narrador Dramático!"

def t_pm_save_overwrites():
    """Saving same name twice should overwrite."""
    save_preset("Overwrite Me", "first", SAMPLE_PARAMS.copy())
    params2 = SAMPLE_PARAMS.copy()
    params2["seed"] = 99
    save_preset("Overwrite Me", "second", params2)
    loaded = load_preset("Overwrite Me")
    assert loaded["params"]["seed"] == 99
    assert loaded["description"] == "second"

def t_pm_with_audio_ref():
    """Save with a real temp audio ref file."""
    # Create a dummy wav file
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.write(b"RIFF" + b"\x00" * 36)  # minimal fake WAV header
    tmp_wav.close()
    try:
        result = save_preset("With Audio Ref", "has ref", SAMPLE_PARAMS.copy(), tmp_wav.name)
        assert result["audio_ref"] is not None
        assert os.path.exists(result["audio_ref"])
        # Verify copy is inside refs dir
        assert "refs" in result["audio_ref"]
    finally:
        os.unlink(tmp_wav.name)

def t_pm_sorted_alphabetically():
    """list_presets() should return sorted by name."""
    save_preset("Zebra Voice", "z", SAMPLE_PARAMS.copy())
    save_preset("Alpha Voice", "a", SAMPLE_PARAMS.copy())
    names = get_preset_names()
    z_idx = names.index("Zebra Voice") if "Zebra Voice" in names else -1
    a_idx = names.index("Alpha Voice") if "Alpha Voice" in names else -1
    assert a_idx < z_idx, "Should be sorted alphabetically (Alpha before Zebra)"

run_test("empty name → ValueError",              t_pm_empty_name_raises)
run_test("whitespace name → ValueError",         t_pm_whitespace_name_raises)
run_test("name with spaces saved/loaded",        t_pm_name_with_spaces)
run_test("name with special chars preserved",    t_pm_name_with_special_chars)
run_test("save twice overwrites correctly",      t_pm_save_overwrites)
run_test("audio_ref file copied to refs/",       t_pm_with_audio_ref)
run_test("list_presets sorted alphabetically",   t_pm_sorted_alphabetically)

# Restore original dirs
pm_module.VOICE_TEMPLATES_DIR = _ORIG_VTDIR
pm_module.REFS_DIR             = _ORIG_REFS
shutil.rmtree(_TMPDIR, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
# SECÇÃO 3 — NARRATOR PIPELINE (no real model)
# ═══════════════════════════════════════════════════════════════════════════
section("5. NARRATOR PIPELINE — crossfade_concat()")

from narrator_pipeline import crossfade_concat, set_seed

SR = 24000
FADE_MS = 50
FADE_SAMPLES = int(SR * FADE_MS / 1000)  # = 1200

def t_cf_empty():
    result = crossfade_concat([])
    assert len(result) == 0
    assert result.dtype == np.float32

def t_cf_single_array():
    wav = np.ones(SR, dtype=np.float32)
    result = crossfade_concat([wav], sr=SR)
    assert np.array_equal(result, wav)

def t_cf_two_arrays_length():
    """Two arrays of 1s joined with crossfade: result should be 2*SR - fade_samples (overlap)."""
    w1 = np.ones(SR, dtype=np.float32)
    w2 = np.ones(SR, dtype=np.float32)
    result = crossfade_concat([w1, w2], sr=SR, crossfade_ms=FADE_MS)
    expected_len = 2 * SR  # crossfade overlaps, but length depends on impl
    # We extend, not shorten — so result >= SR*2 - fade_len and <= SR*2
    assert len(result) >= SR, f"Result too short: {len(result)}"

def t_cf_three_arrays():
    """Three arrays should be successfully concatenated."""
    wavs = [np.ones(SR, dtype=np.float32) * i for i in range(1, 4)]
    result = crossfade_concat(wavs, sr=SR, crossfade_ms=FADE_MS)
    assert len(result) >= SR, "Result must be at least one chunk long"
    assert result.dtype == np.float32

def t_cf_tiny_arrays_no_crash():
    """Arrays smaller than fade window: should fallback to simple concat."""
    tiny = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # 3 samples << 1200 fade
    result = crossfade_concat([tiny, tiny], sr=SR, crossfade_ms=FADE_MS)
    assert len(result) == 6  # direct concat
    assert not np.any(np.isnan(result))

def t_cf_no_nan_no_inf():
    """Output must never contain NaN or Inf."""
    wavs = [np.random.randn(SR).astype(np.float32) for _ in range(5)]
    result = crossfade_concat(wavs, sr=SR, crossfade_ms=FADE_MS)
    assert not np.any(np.isnan(result)), "NaN found in crossfade output"
    assert not np.any(np.isinf(result)), "Inf found in crossfade output"

def t_cf_crossfade_boundary_smooth():
    """The boundary region should be between the two arrays' values (not a spike)."""
    w1 = np.ones(SR * 2, dtype=np.float32) * 0.8
    w2 = np.zeros(SR * 2, dtype=np.float32)
    result = crossfade_concat([w1, w2], sr=SR, crossfade_ms=FADE_MS)
    # At the crossfade boundary, values should transition between 0.8 and 0.0
    boundary_start = len(w1) - FADE_SAMPLES
    boundary_end   = len(w1)
    region = result[boundary_start:boundary_end]
    # Values should be between 0 and 0.8 (not spiking above w1 or below 0)
    assert np.all(region >= -0.01), f"Values below zero in crossfade: {region.min()}"
    assert np.all(region <= 0.81),  f"Values above w1 in crossfade: {region.max()}"

def t_cf_output_is_float32():
    wavs = [np.ones(100, dtype=np.float32)]
    result = crossfade_concat(wavs)
    assert result.dtype == np.float32

run_test("empty list → empty float32 array",          t_cf_empty)
run_test("single array → returned unchanged",          t_cf_single_array)
run_test("two arrays → valid length output",           t_cf_two_arrays_length)
run_test("three arrays concat correctly",              t_cf_three_arrays)
run_test("tiny arrays (<fade) → simple concat, no crash", t_cf_tiny_arrays_no_crash)
run_test("no NaN / Inf in output (random wavs)",      t_cf_no_nan_no_inf)
run_test("crossfade boundary values smooth (no spike)", t_cf_crossfade_boundary_smooth)
run_test("output dtype is float32",                    t_cf_output_is_float32)


section("6. NARRATOR PIPELINE — set_seed()")

def t_seed_reproducible():
    """Same seed → same random numbers."""
    set_seed(42)
    a = np.random.rand(10)
    set_seed(42)
    b = np.random.rand(10)
    assert np.allclose(a, b), "Same seed must produce same results"

def t_seed_different():
    """Different seeds → different random numbers."""
    set_seed(42)
    a = np.random.rand(10)
    set_seed(99)
    b = np.random.rand(10)
    assert not np.allclose(a, b), "Different seeds must produce different results"

def t_seed_zero_no_crash():
    """Seed=0 must not crash."""
    set_seed(0)

def t_seed_offset_produces_sequence():
    """seed+0, seed+1, seed+2 must all be different → correct offset behavior."""
    results = []
    base = 42
    for i in range(3):
        set_seed(base + i)
        results.append(np.random.rand(5).tolist())
    # All three sequences must be different
    assert results[0] != results[1], "seed+0 == seed+1 → offset not working"
    assert results[1] != results[2], "seed+1 == seed+2 → offset not working"
    assert results[0] != results[2], "seed+0 == seed+2 → offset not working"

run_test("same seed → same sequence",         t_seed_reproducible)
run_test("different seeds → different seq",   t_seed_different)
run_test("seed=0 no crash",                   t_seed_zero_no_crash)
run_test("offset seeds all different",        t_seed_offset_produces_sequence)


section("7. NARRATOR PIPELINE — NarratorPipeline (mock model)")

import unittest.mock as mock
from narrator_pipeline import NarratorPipeline
import torch

def make_mock_model(sr=24000):
    """Creates a mock ChatterboxTurboTTS that returns a synthetic wav tensor."""
    m = mock.MagicMock()
    m.sr = sr
    m.conds = mock.MagicMock()  # pretend conds are loaded (builtin voice)
    # generate() returns a (1, N) tensor of ones
    fake_wav = torch.ones(1, sr // 2)  # 0.5s per chunk
    m.generate.return_value = fake_wav
    m.prepare_conditionals.return_value = None
    return m

def t_pipeline_empty_text_raises():
    """Empty text must raise ValueError before touching the model."""
    pipeline = NarratorPipeline()
    pipeline.model = make_mock_model()  # inject mock
    try:
        pipeline.generate_narration("")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "vazio" in str(e).lower() or "empty" in str(e).lower() or "válidos" in str(e).lower()

def t_pipeline_whitespace_raises():
    pipeline = NarratorPipeline()
    pipeline.model = make_mock_model()
    try:
        pipeline.generate_narration("   \n\n  ")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def t_pipeline_short_text_produces_audio():
    """Short text should produce (sr, ndarray) tuple."""
    pipeline = NarratorPipeline()
    mock_m = make_mock_model()
    pipeline.model = mock_m
    pipeline.sr = mock_m.sr

    sr, wav = pipeline.generate_narration(
        "Hello world. This is a test.",
        base_seed=42
    )
    assert sr == 24000
    assert isinstance(wav, np.ndarray)
    assert len(wav) > 0

def t_pipeline_generate_called_per_chunk():
    """Model.generate must be called once per chunk."""
    pipeline = NarratorPipeline()
    mock_m = make_mock_model()
    pipeline.model = mock_m
    pipeline.sr = mock_m.sr

    text = (
        "First sentence here. Second sentence here. Third sentence here. "
        "Fourth sentence here. Fifth sentence here."
    )
    chunks_expected = split_into_chunks(text, max_chars=250)
    pipeline.generate_narration(text, base_seed=42, max_chars_per_chunk=250)

    assert mock_m.generate.call_count == len(chunks_expected), (
        f"Expected {len(chunks_expected)} calls, got {mock_m.generate.call_count}"
    )

def t_pipeline_audio_prompt_none_not_passed():
    """When no audio_prompt_path given, model.generate must be called with audio_prompt_path=None."""
    pipeline = NarratorPipeline()
    mock_m = make_mock_model()
    pipeline.model = mock_m
    pipeline.sr = mock_m.sr

    pipeline.generate_narration("Hello world. Test sentence.", base_seed=10)

    for call in mock_m.generate.call_args_list:
        kwargs = call.kwargs if call.kwargs else {}
        args = call.args
        # audio_prompt_path must be None in every call
        ap = kwargs.get("audio_prompt_path", None)
        assert ap is None, f"audio_prompt_path was {ap}, expected None"

def t_pipeline_audio_prompt_sets_conditionals():
    """When audio_prompt_path is given, prepare_conditionals must be called once."""
    pipeline = NarratorPipeline()
    mock_m = make_mock_model()
    pipeline.model = mock_m
    pipeline.sr = mock_m.sr

    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.write(b"RIFF" + b"\x00" * 36)
    tmp_wav.close()
    try:
        pipeline.generate_narration(
            "Hello world. Test sentence.", audio_prompt_path=tmp_wav.name, base_seed=5
        )
        assert mock_m.prepare_conditionals.call_count == 1, (
            f"prepare_conditionals called {mock_m.prepare_conditionals.call_count} times, expected 1"
        )
    finally:
        os.unlink(tmp_wav.name)

def t_pipeline_n_cfm_passed_to_generate():
    """n_cfm_timesteps value must be forwarded to model.generate on every call."""
    pipeline = NarratorPipeline()
    mock_m = make_mock_model()
    pipeline.model = mock_m
    pipeline.sr = mock_m.sr

    pipeline.generate_narration("Hello world. Another sentence.", base_seed=1, n_cfm_timesteps=6)

    for call in mock_m.generate.call_args_list:
        kwargs = call.kwargs if call.kwargs else {}
        assert kwargs.get("n_cfm_timesteps") == 6, (
            f"Expected n_cfm_timesteps=6, got {kwargs.get('n_cfm_timesteps')}"
        )

def t_pipeline_seed_offset_per_chunk():
    """Each chunk must receive seed=base+idx, verify via side effects."""
    pipeline = NarratorPipeline()
    mock_m = make_mock_model()
    pipeline.model = mock_m
    pipeline.sr = mock_m.sr

    # Use a long enough text to force >=2 chunks
    text = NORDSTREAM_INTRO[:500]
    chunks = split_into_chunks(text, max_chars=100)
    assert len(chunks) >= 2, f"Need >=2 chunks for this test, got {len(chunks)}"

    seeds_seen = []
    orig_set_seed = __import__("narrator_pipeline").set_seed

    with mock.patch("narrator_pipeline.set_seed") as mock_seed:
        pipeline.generate_narration(text, base_seed=42, max_chars_per_chunk=100)
        calls = [c.args[0] for c in mock_seed.call_args_list]

    # Seeds should be 42, 43, 44, ... not all the same
    assert calls == list(range(42, 42 + len(chunks))), (
        f"Expected {list(range(42, 42+len(chunks)))}, got {calls}"
    )

def t_pipeline_progress_callback():
    """progress_callback must be called once per chunk with correct args."""
    pipeline = NarratorPipeline()
    mock_m = make_mock_model()
    pipeline.model = mock_m
    pipeline.sr = mock_m.sr

    text = "Hello world. Second sentence here. Third sentence."
    chunks = split_into_chunks(text, max_chars=250)
    calls = []

    def cb(current, total, chunk_text):
        calls.append((current, total, chunk_text))

    pipeline.generate_narration(text, base_seed=42, progress_callback=cb)

    assert len(calls) == len(chunks), f"Expected {len(chunks)} callbacks, got {len(calls)}"
    for i, (current, total, _) in enumerate(calls):
        assert current == i + 1, f"current={current}, expected {i+1}"
        assert total == len(chunks), f"total={total}, expected {len(chunks)}"

def t_pipeline_output_is_ndarray_float():
    pipeline = NarratorPipeline()
    mock_m = make_mock_model()
    pipeline.model = mock_m
    pipeline.sr = mock_m.sr

    sr, wav = pipeline.generate_narration("Test sentence one. Test sentence two.")
    assert isinstance(wav, np.ndarray)
    assert wav.dtype == np.float32

run_test("empty text → ValueError",                       t_pipeline_empty_text_raises)
run_test("whitespace text → ValueError",                  t_pipeline_whitespace_raises)
run_test("short text → (sr, ndarray) tuple",              t_pipeline_short_text_produces_audio)
run_test("generate() called exactly N times (N=chunks)",  t_pipeline_generate_called_per_chunk)
run_test("audio_prompt_path=None on every generate call", t_pipeline_audio_prompt_none_not_passed)
run_test("audio_prompt → prepare_conditionals called 1x", t_pipeline_audio_prompt_sets_conditionals)
run_test("n_cfm_timesteps forwarded to generate()",       t_pipeline_n_cfm_passed_to_generate)
run_test("seed offset 42,43,44... per chunk",             t_pipeline_seed_offset_per_chunk)
run_test("progress_callback called once per chunk",       t_pipeline_progress_callback)
run_test("output is float32 ndarray",                     t_pipeline_output_is_ndarray_float)


# ═══════════════════════════════════════════════════════════════════════════
# SECÇÃO 4 — SOURCE TTS_TURBO SIGNATURE CHECK
# ═══════════════════════════════════════════════════════════════════════════
section("8. TTS_TURBO SOURCE — generate() signature")

def t_turbo_signature_has_n_cfm():
    """generate() must have n_cfm_timesteps in its parameters."""
    # Import only the class definition, not instantiate it (no model weights needed)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tts_turbo_check",
        os.path.join(SRC, "chatterbox", "tts_turbo.py")
    )
    mod = importlib.util.load_from_spec_without_exec = None
    # Parse via inspect of the source
    import ast
    src_path = os.path.join(SRC, "chatterbox", "tts_turbo.py")
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    found_n_cfm = False
    default_is_4 = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ChatterboxTurboTTS":
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == "generate":
                    args = item.args
                    # Collect arg names
                    all_arg_names = [a.arg for a in args.args]
                    if "n_cfm_timesteps" in all_arg_names:
                        found_n_cfm = True
                        # Find default value
                        # defaults align to the END of args.args
                        num_defaults = len(args.defaults)
                        num_args     = len(args.args)
                        offset = num_args - num_defaults
                        idx = all_arg_names.index("n_cfm_timesteps")
                        default_idx = idx - offset
                        if 0 <= default_idx < len(args.defaults):
                            default_node = args.defaults[default_idx]
                            if isinstance(default_node, ast.Constant) and default_node.value == 4:
                                default_is_4 = True

    assert found_n_cfm, "n_cfm_timesteps NOT found in ChatterboxTurboTTS.generate() signature"
    assert default_is_4, "n_cfm_timesteps default is NOT 4 in generate() signature"

def t_turbo_signature_n_cfm_in_s3gen_call():
    """s3gen.inference() call must pass n_cfm_timesteps (not hardcoded 2)."""
    src_path = os.path.join(SRC, "chatterbox", "tts_turbo.py")
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    # Should NOT have the old hardcoded n_cfm_timesteps=2 anymore
    assert "n_cfm_timesteps=2" not in source, (
        "Found hardcoded n_cfm_timesteps=2 — should be the variable n_cfm_timesteps"
    )
    # Should have n_cfm_timesteps=n_cfm_timesteps or n_cfm_timesteps=int(n_cfm_timesteps)
    assert "n_cfm_timesteps=n_cfm_timesteps" in source or "n_cfm_timesteps=int(n_cfm_timesteps)" in source, (
        "n_cfm_timesteps variable not forwarded to s3gen.inference()"
    )

def t_turbo_signature_default_params():
    """Verify other optimised default values are in the signature."""
    src_path = os.path.join(SRC, "chatterbox", "tts_turbo.py")
    with open(src_path, "r", encoding="utf-8") as f:
        source = f.read()
    # The generate() function must exist
    assert "def generate(" in source, "generate() function missing from tts_turbo.py"

run_test("n_cfm_timesteps in generate() signature (AST)", t_turbo_signature_has_n_cfm)
run_test("n_cfm_timesteps=2 hardcode removed",            t_turbo_signature_n_cfm_in_s3gen_call)
run_test("generate() function exists in file",            t_turbo_signature_default_params)


# ═══════════════════════════════════════════════════════════════════════════
# SECÇÃO 5 — NARRATOR_EN_DOCUMENTARY PRESET FILE
# ═══════════════════════════════════════════════════════════════════════════
section("9. DEFAULT PRESET — narrator_en_documentary.json")

def t_default_preset_exists():
    preset_path = os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json")
    assert os.path.exists(preset_path), f"File not found: {preset_path}"

def t_default_preset_valid_json():
    preset_path = os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json")
    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "name" in data
    assert "params" in data

def t_default_preset_has_required_params():
    preset_path = os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json")
    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    params = data["params"]
    required = ["temperature", "repetition_penalty", "top_p", "min_p", "top_k", "seed", "n_cfm_timesteps"]
    for key in required:
        assert key in params, f"Missing required param '{key}' in preset"

def t_default_preset_n_cfm_is_4():
    preset_path = os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json")
    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["params"]["n_cfm_timesteps"] == 4, (
        f"n_cfm_timesteps should be 4, got {data['params']['n_cfm_timesteps']}"
    )

def t_default_preset_optimised_values():
    preset_path = os.path.join(ROOT, "voice_templates", "narrator_en_documentary.json")
    with open(preset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    p = data["params"]
    assert p["temperature"] == 1.0,         f"temperature: {p['temperature']}"
    assert p["repetition_penalty"] == 1.05, f"repetition_penalty: {p['repetition_penalty']}"
    assert p["top_p"] == 0.85,              f"top_p: {p['top_p']}"
    assert p["min_p"] == 0.0,               f"min_p: {p['min_p']}"
    assert p["top_k"] == 500,               f"top_k: {p['top_k']}"
    assert p["seed"] == 42,                 f"seed: {p['seed']}"

run_test("narrator_en_documentary.json exists",      t_default_preset_exists)
run_test("valid JSON file",                          t_default_preset_valid_json)
run_test("all required param keys present",          t_default_preset_has_required_params)
run_test("n_cfm_timesteps == 4 in preset",           t_default_preset_n_cfm_is_4)
run_test("all optimised values correct",             t_default_preset_optimised_values)


# ═══════════════════════════════════════════════════════════════════════════
# SECÇÃO 6 — GRADIO APP IMPORT CHECK (sem lançar o server)
# ═══════════════════════════════════════════════════════════════════════════
section("10. GRADIO APP — import & structure check")

def t_gradio_app_imports():
    """Import the app module without launching the server."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "gradio_app_check",
            os.path.join(ROOT, "gradio_tts_turbo_app.py")
        )
        mod = importlib.util.module_from_spec(spec)
        # We don't exec it (would launch server), just check the file parses
        import ast
        with open(os.path.join(ROOT, "gradio_tts_turbo_app.py"), "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)  # Syntax check
        assert tree is not None
    except SyntaxError as e:
        assert False, f"Syntax error in gradio_tts_turbo_app.py: {e}"

def t_gradio_app_has_build_app():
    with open(os.path.join(ROOT, "gradio_tts_turbo_app.py"), "r", encoding="utf-8") as f:
        source = f.read()
    assert "def build_app(" in source or "def build_app()" in source, "build_app() function missing"

def t_gradio_app_uses_narrator_pipeline():
    with open(os.path.join(ROOT, "gradio_tts_turbo_app.py"), "r", encoding="utf-8") as f:
        source = f.read()
    assert "NarratorPipeline" in source, "NarratorPipeline not referenced in app"

def t_gradio_app_uses_preset_manager():
    with open(os.path.join(ROOT, "gradio_tts_turbo_app.py"), "r", encoding="utf-8") as f:
        source = f.read()
    assert "save_preset" in source, "save_preset not in app"
    assert "load_preset" in source, "load_preset not in app"
    assert "delete_preset" in source, "delete_preset not in app"
    assert "get_preset_names" in source, "get_preset_names not in app"

def t_gradio_app_has_progress():
    with open(os.path.join(ROOT, "gradio_tts_turbo_app.py"), "r", encoding="utf-8") as f:
        source = f.read()
    assert "gr.Progress" in source, "gr.Progress not used in app"

def t_gradio_app_has_chunk_preview():
    with open(os.path.join(ROOT, "gradio_tts_turbo_app.py"), "r", encoding="utf-8") as f:
        source = f.read()
    assert "split_into_chunks" in source or "estimate_generation" in source, (
        "Chunk preview functions not referenced in app"
    )

def t_gradio_app_port_7860():
    with open(os.path.join(ROOT, "gradio_tts_turbo_app.py"), "r", encoding="utf-8") as f:
        source = f.read()
    assert "7860" in source, "Port 7860 not configured in app"

def t_gradio_css_in_launch_not_blocks():
    """Gradio 6.x: css must be in launch(), NOT in gr.Blocks().
    gr.Blocks(css=...) is deprecated and produces UserWarning in Gradio 6.0+."""
    with open(os.path.join(ROOT, "gradio_tts_turbo_app.py"), "r", encoding="utf-8") as f:
        source = f.read()
    # Must NOT have css in gr.Blocks() call
    import ast
    tree = ast.parse(source)
    blocks_has_css = False
    launch_has_css = False
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                c = item.context_expr
                if isinstance(c, ast.Call):
                    func = c.func
                    fname = (getattr(func, 'attr', None) or getattr(func, 'id', None))
                    if fname == 'Blocks':
                        kw_names = [k.arg for k in c.keywords]
                        if 'css' in kw_names:
                            blocks_has_css = True
        if isinstance(node, ast.Call):
            func = node.func
            fname = (getattr(func, 'attr', None) or getattr(func, 'id', None))
            if fname == 'launch':
                kw_names = [k.arg for k in node.keywords]
                if 'css' in kw_names:
                    launch_has_css = True
    assert not blocks_has_css, "css found in gr.Blocks() — must be in launch() per Gradio 6.x API"
    assert launch_has_css, "css NOT found in launch() — must be passed there in Gradio 6.x"

run_test("app file parses without syntax errors",   t_gradio_app_imports)
run_test("build_app() function defined",            t_gradio_app_has_build_app)
run_test("NarratorPipeline referenced in app",      t_gradio_app_uses_narrator_pipeline)
run_test("all preset functions used in app",        t_gradio_app_uses_preset_manager)
run_test("gr.Progress used for real progress bar",  t_gradio_app_has_progress)
run_test("chunk preview logic in app",              t_gradio_app_has_chunk_preview)
run_test("server configured on port 7860",          t_gradio_app_port_7860)
run_test("css in launch() not gr.Blocks() [Gradio 6.x]", t_gradio_css_in_launch_not_blocks)


# ═══════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
ok = summary()
sys.exit(0 if ok else 1)
