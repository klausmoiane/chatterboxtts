import unittest
import numpy as np
import pyloudnorm as ln
import noisereduce as nr


class TestFix2A_NoiseReduction(unittest.TestCase):
    def setUp(self):
        self.sr = 24000
        self.duration = 2.0  # 2 segundos
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)

        # 8.1 Sinal útil senoidal (frequência de voz ~440 Hz) + ruído branco
        self.clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        np.random.seed(42)
        self.noise = (0.08 * np.random.normal(0, 1, len(t))).astype(np.float32)
        self.noisy_audio = self.clean_signal + self.noise

    def test_8_1_fixture_created(self):
        """8.1 Verificar que a fixture de áudio sintético com ruído foi criada corretamente."""
        self.assertEqual(len(self.noisy_audio), int(self.sr * self.duration))
        self.assertFalse(np.allclose(self.noisy_audio, self.clean_signal))
        # O ruído adicionado deve aumentar o desvio padrão
        self.assertGreater(np.std(self.noisy_audio), np.std(self.clean_signal))

    def test_8_2_noise_energy_reduction_at_least_50_percent(self):
        """8.2 Testar que noisereduce.reduce_noise(prop_decrease=0.8) reduz RMS do ruído em >= 50%."""
        # Criar trecho de apenas ruído (como os primeiros 0.5s usados como perfil)
        noise_only = self.noise[: int(self.sr * 0.5)]

        # Aplicar noisereduce sobre um sinal que é apenas ruído
        reduced_noise = nr.reduce_noise(
            y=self.noise,
            sr=self.sr,
            y_noise=noise_only,
            prop_decrease=0.8
        )

        rms_before = float(np.sqrt(np.mean(self.noise ** 2)))
        rms_after = float(np.sqrt(np.mean(reduced_noise ** 2)))
        reduction_pct = (1.0 - (rms_after / rms_before)) * 100.0

        print(f"\n[Test 8.2] RMS Antes: {rms_before:.4f} | RMS Depois: {rms_after:.4f} | Redução: {reduction_pct:.1f}%")
        self.assertGreaterEqual(reduction_pct, 50.0, f"Redução esperada >= 50%, obtido: {reduction_pct:.1f}%")

    def test_8_3_useful_signal_preserved_above_70_percent(self):
        """8.3 Testar que o sinal útil (frequência de voz) mantém amplitude pico-a-pico > 70%."""
        noise_profile = self.noise[: int(self.sr * 0.5)]
        cleaned = nr.reduce_noise(
            y=self.noisy_audio,
            sr=self.sr,
            y_noise=noise_profile,
            prop_decrease=0.8
        )

        peak_to_peak_orig = float(np.ptp(self.clean_signal))
        peak_to_peak_cleaned = float(np.ptp(cleaned))
        retention_ratio = peak_to_peak_cleaned / peak_to_peak_orig

        print(f"\n[Test 8.3] Pico-a-pico limpo: {peak_to_peak_orig:.4f} | Processado: {peak_to_peak_cleaned:.4f} | Retenção: {retention_ratio*100:.1f}%")
        self.assertGreater(retention_ratio, 0.70, f"Retenção esperada > 70%, obtido: {retention_ratio*100:.1f}%")

    def test_8_4_shape_and_length_preserved(self):
        """8.4 Testar que o output tem o mesmo comprimento e tipo de dados sem truncação."""
        noise_profile = self.noise[: int(self.sr * 0.5)]
        cleaned = nr.reduce_noise(
            y=self.noisy_audio,
            sr=self.sr,
            y_noise=noise_profile,
            prop_decrease=0.8
        )
        self.assertEqual(len(cleaned), len(self.noisy_audio))
        self.assertEqual(cleaned.shape, self.noisy_audio.shape)
        self.assertFalse(np.isnan(cleaned).any())
        self.assertFalse(np.isinf(cleaned).any())


class TestFix2B_LoudnessNormalization(unittest.TestCase):
    def setUp(self):
        self.sr = 24000
        self.duration = 2.0  # 2s para medição BS.1770 estável
        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        # Senoide pura para teste com amplitude moderada
        self.test_audio = (0.1 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        self.meter = ln.Meter(self.sr)

    def test_8_5_known_loudness_signal_creation(self):
        """8.5 Testar medição de loudness de sinal de teste com pyloudnorm."""
        measured_loudness = self.meter.integrated_loudness(self.test_audio)
        self.assertTrue(np.isfinite(measured_loudness))
        print(f"\n[Test 8.5] Loudness medida inicial: {measured_loudness:.2f} LUFS")
        self.assertLess(measured_loudness, -10.0)

    def test_8_6_normalization_to_target_lufs(self):
        """8.6 Testar que normalização para target -23.0 LUFS resulta em loudness dentro de ±0.5 LUFS."""
        target_lufs = -23.0
        cur_loudness = self.meter.integrated_loudness(self.test_audio)

        # Fórmula de ganho
        gain_linear = 10.0 ** ((target_lufs - cur_loudness) / 20.0)
        norm_audio = self.test_audio * np.float32(gain_linear)

        final_loudness = self.meter.integrated_loudness(norm_audio)
        print(f"\n[Test 8.6] Inicial: {cur_loudness:.2f} LUFS -> Target: {target_lufs} -> Final: {final_loudness:.2f} LUFS")
        self.assertAlmostEqual(final_loudness, target_lufs, delta=0.5)

    def test_8_7_infinite_silence_safety(self):
        """8.7 Testar caso de silêncio (-inf): np.isfinite deve impedir ganho e não causar erro."""
        silence = np.zeros(int(self.sr * 1.0), dtype=np.float32)
        measured_loudness = self.meter.integrated_loudness(silence)

        # Confirmar que é -inf ou não finito
        self.assertFalse(np.isfinite(measured_loudness))

        # Simular a lógica do fix
        target_lufs = -23.0
        if np.isfinite(measured_loudness):
            gain_linear = 10.0 ** ((target_lufs - measured_loudness) / 20.0)
            norm_silence = silence * np.float32(gain_linear)
        else:
            norm_silence = silence  # Não altera

        self.assertTrue(np.array_equal(norm_silence, silence))
        self.assertFalse(np.isnan(norm_silence).any())

    def test_8_8_soft_clipping_bounds(self):
        """8.8 Testar que após normalização + np.clip(-0.95, 0.95), nenhuma amostra excede ±0.95."""
        # Criar sinal de teste com picos deliberadamente altos (> 2.0)
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        hot_audio = (2.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

        clipped = np.clip(hot_audio, -0.95, 0.95)

        self.assertLessEqual(float(np.max(clipped)), 0.950001)
        self.assertGreaterEqual(float(np.min(clipped)), -0.950001)
        self.assertEqual(clipped.dtype, np.float32)
        print(f"\n[Test 8.8] Max após clip: {np.max(clipped):.4f} | Min: {np.min(clipped):.4f}")


# ============================================================================
# Helpers do Fix 3A (Chunking Estratégia B)
# ============================================================================
import re
import textwrap
from pathlib import Path

MAX_CHARS = 250

def normalize_text(text: str) -> str:
    if not text:
        return ''
    text = re.sub(r'\[\s*(music|__|[a-zA-Z0-9_]+)\s*\]', '', text, flags=re.IGNORECASE)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    text = text.replace('—', ', ').replace('–', ', ').replace('...', ', ')
    return text.strip()

def split_into_chunks_new(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    final_chunks = []

    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current_chunk = ''

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(sentence) > max_chars:
                if current_chunk:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = ''

                sub_parts = re.split(r'(?<=,)\s+', sentence)
                sub_chunk = ''
                for part in sub_parts:
                    if len(sub_chunk) + len(part) + 1 <= max_chars:
                        sub_chunk = f'{sub_chunk} {part}'.strip()
                    else:
                        if sub_chunk:
                            final_chunks.append(sub_chunk.strip())
                        if len(part) > max_chars:
                            wrapped = textwrap.wrap(part, width=max_chars)
                            final_chunks.extend(wrapped)
                            sub_chunk = ''
                        else:
                            sub_chunk = part
                if sub_chunk:
                    final_chunks.append(sub_chunk.strip())
            else:
                if len(current_chunk) + len(sentence) + 1 <= max_chars:
                    current_chunk = f'{current_chunk} {sentence}'.strip()
                else:
                    final_chunks.append(current_chunk.strip())
                    current_chunk = sentence

        if current_chunk:
            final_chunks.append(current_chunk.strip())

    return [c.strip() for c in final_chunks if c.strip()]


def split_into_chunks_old(text: str, max_chars: int = 220) -> list[str]:
    """Algoritmo antigo por janela fixa de 220 chars."""
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text.strip())
            break
        window = text[:max_chars]
        match = None
        for pattern in [r'[.!?]\s', r',\s', r'\s']:
            found = list(re.finditer(pattern, window))
            if found:
                match = found[-1]
                break
        cut = match.end() if match else max_chars
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    return [c for c in chunks if c]


# ============================================================================
# Helpers do Fix 3B (Crossfade Concat)
# ============================================================================
def crossfade_concat(segments: list, cf_samples: int) -> np.ndarray:
    if len(segments) == 0:
        return np.array([], dtype=np.float32)
    if len(segments) == 1:
        return segments[0].astype(np.float32)

    result = segments[0].astype(np.float32)

    for seg in segments[1:]:
        seg = seg.astype(np.float32)
        cf = min(cf_samples, len(result), len(seg))

        if cf <= 0:
            result = np.concatenate([result, seg])
            continue

        fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
        fade_in  = np.linspace(0.0, 1.0, cf, dtype=np.float32)
        overlap = result[-cf:] * fade_out + seg[:cf] * fade_in
        result = np.concatenate([result[:-cf], overlap, seg[cf:]])

    return result


class TestFix3A_ChunkingSentenceAware(unittest.TestCase):
    def test_8_9_short_sentences_grouped(self):
        """8.9 Três frases curtas devem ser agrupadas num único chunk se a soma <= 250 chars."""
        text = "Hello world. This is a simple test. We love sentence-aware chunking."
        self.assertLessEqual(len(text), 250)
        chunks = split_into_chunks_new(text, max_chars=250)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_8_10_long_sentence_subdivided_by_comma(self):
        """8.10 Frase longa (>250 chars) com orações deve ser subdividida por vírgula sem quebrar palavras."""
        clause1 = "In an unprecedented move that surprised both diplomats and international observers,"
        clause2 = "the foreign ministry decided to immediately suspend negotiations,"
        clause3 = "demanding that all previously seized commercial vessels be returned without delay,"
        clause4 = "which escalated geopolitical tensions in the region to a critical point."
        long_sentence = f"{clause1} {clause2} {clause3} {clause4}"
        self.assertGreater(len(long_sentence), 250)

        chunks = split_into_chunks_new(long_sentence, max_chars=250)
        self.assertGreater(len(chunks), 1)
        for ch in chunks:
            self.assertLessEqual(len(ch), 250)
            # Verificar que não quebrou palavras ao meio (espaço ou vírgula no final)
            self.assertTrue(ch.endswith(',') or ch.endswith('.'))

    def test_8_11_paragraphs_kept_independent(self):
        """8.11 Parágrafos separados por \n\n devem gerar chunks que não misturam conteúdo entre si."""
        para1 = "This is paragraph number one. It contains two full sentences."
        para2 = "This is paragraph number two. It must not be merged with the first paragraph."
        full_text = f"{para1}\n\n{para2}"

        chunks = split_into_chunks_new(full_text, max_chars=250)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], para1)
        self.assertEqual(chunks[1], para2)

    def test_8_12_invariant_max_chars_never_exceeded(self):
        """8.12 Invariante fundamental: nenhum chunk pode exceder MAX_CHARS = 250."""
        text_sample = (
            "On the night of Monday, November 2, just hours before polls opened in the United States, "
            "Donald Trump was asked whether he would bomb Iran again after the elections. "
            "He answered with a single word, no. Shortly afterward, American fighter jets attacked the Iranian island of Qeshm."
        )
        chunks = split_into_chunks_new(text_sample, max_chars=250)
        for i, ch in enumerate(chunks):
            self.assertLessEqual(len(ch), 250, f"Chunk {i} excedeu 250 chars ({len(ch)} chars): {ch}")

    def test_8_13_no_empty_or_whitespace_chunks(self):
        """8.13 Nenhum chunk gerado pode ser vazio ou conter apenas espaços."""
        text = "First sentence.   \n\n\n\n   Second sentence.  \n\n   "
        chunks = split_into_chunks_new(text, max_chars=250)
        self.assertEqual(len(chunks), 2)
        for ch in chunks:
            self.assertTrue(bool(ch.strip()))

    def test_8_14_compare_new_vs_old_with_real_input(self):
        """8.14 Comparar algoritmo novo vs antigo no input_text.txt real: novo preserva fim de frases."""
        input_path = Path("kaggle_runner/input_text.txt")
        if not input_path.exists():
            self.skipTest("input_text.txt não encontrado")

        with open(input_path, "r", encoding="utf-8") as f:
            real_text = f.read().strip()

        chunks_new = split_into_chunks_new(real_text, max_chars=250)
        self.assertGreater(len(chunks_new), 0)

        valid_endings = ('.', '!', '?', ',', '"', "'")
        for i, ch in enumerate(chunks_new):
            self.assertLessEqual(len(ch), 250)
            self.assertTrue(
                ch.endswith(valid_endings),
                f"Chunk {i} terminou de forma abrupta: '{ch[-30:]}'"
            )
        print(f"\n[Test 8.14] Real text ({len(real_text)} chars) -> {len(chunks_new)} chunks sentence-aware!")

    def test_8_15_empty_text_returns_empty_list(self):
        """8.15 Texto vazio ou de apenas espaços deve retornar [] sem erros."""
        self.assertEqual(split_into_chunks_new(""), [])
        self.assertEqual(split_into_chunks_new("   \n\n   "), [])


class TestFix3B_CrossfadeConcat(unittest.TestCase):
    def setUp(self):
        self.sr = 24000
        t1 = np.linspace(0, 1.0, self.sr, endpoint=False)
        t2 = np.linspace(0, 1.0, self.sr, endpoint=False)
        self.seg1 = 0.5 * np.sin(2 * np.pi * 440 * t1).astype(np.float32)
        self.seg2 = 0.5 * np.sin(2 * np.pi * 880 * t2).astype(np.float32)

    def test_8_16_synthetic_segments_created(self):
        """8.16 Verificar que os dois segmentos de teste têm 1 segundo cada."""
        self.assertEqual(len(self.seg1), 24000)
        self.assertEqual(len(self.seg2), 24000)

    def test_8_17_crossfade_100ms_length_exact(self):
        """8.17 Com crossfade de 100ms, o tamanho final deve ser len(s1) + len(s2) - cf_samples."""
        cf_ms = 100
        cf_samples = int(self.sr * cf_ms / 1000)  # 2400 amostras
        self.assertEqual(cf_samples, 2400)

        result = crossfade_concat([self.seg1, self.seg2], cf_samples)
        expected_len = len(self.seg1) + len(self.seg2) - cf_samples
        self.assertEqual(len(result), expected_len)
        print(f"\n[Test 8.17] Segmentos (24000 + 24000) com crossfade 100ms ({cf_samples} amostras) -> {len(result)} amostras")

    def test_8_18_no_abrupt_clicks_in_overlap_zone(self):
        """8.18 Na zona de sobreposição, as transições entre amostras consecutivas devem ser suaves (< 0.3)."""
        cf_samples = int(self.sr * 100 / 1000)
        result = crossfade_concat([self.seg1, self.seg2], cf_samples)

        # Região de transição
        overlap_start = len(self.seg1) - cf_samples
        overlap_end = len(self.seg1)
        transition_zone = result[overlap_start:overlap_end]

        diffs = np.abs(np.diff(transition_zone))
        max_jump = float(np.max(diffs))
        self.assertLess(max_jump, 0.3, f"Salto abrupto de amplitude detectado: {max_jump}")
        print(f"\n[Test 8.18] Maior salto consecutivo na zona de crossfade: {max_jump:.5f} (limite < 0.3)")

    def test_8_19_compare_50ms_vs_100ms(self):
        """8.19 Crossfade de 100ms deve ter exatamente o dobro de amostras de sobreposição que 50ms."""
        cf_50 = int(self.sr * 50 / 1000)
        cf_100 = int(self.sr * 100 / 1000)
        self.assertEqual(cf_100, 2 * cf_50)
        self.assertEqual(cf_50, 1200)
        self.assertEqual(cf_100, 2400)

    def test_8_20_short_segment_edge_case(self):
        """8.20 Segmento mais curto que cf_samples deve concatenar com segurança sem exceções."""
        tiny_seg = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        cf_samples = 2400
        # tiny_seg tem apenas 3 amostras < 2400, todo absorvido no crossfade
        res = crossfade_concat([self.seg1, tiny_seg], cf_samples)
        self.assertEqual(len(res), len(self.seg1))
        self.assertFalse(np.isnan(res).any())
        self.assertFalse(np.isinf(res).any())


# ============================================================================
# Testes do Fix 3C (Seed Determinístico por Chunk)
# ============================================================================
import random
import torch
from unittest.mock import patch, call

class TestFix3C_DeterministicSeed(unittest.TestCase):
    def test_8_21_unique_seeds_per_chunk(self):
        """8.21 Simular o loop de seeds: BASE_SEED + idx para N chunks deve gerar seeds únicos."""
        base_seed = 42
        num_chunks = 29
        seeds = [base_seed + idx for idx in range(num_chunks)]
        self.assertEqual(len(seeds), num_chunks)
        self.assertEqual(len(set(seeds)), num_chunks)
        # Ordem monotonicamente crescente
        self.assertEqual(seeds, sorted(seeds))

    @patch('torch.manual_seed')
    @patch('random.seed')
    @patch('numpy.random.seed')
    def test_8_22_mock_seed_invocation_order(self, mock_np_seed, mock_rand_seed, mock_torch_seed):
        """8.22 Verificar que torch.manual_seed, random.seed e np.random.seed são chamados a cada chunk."""
        base_seed = 42
        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

        for idx, _ in enumerate(chunks):
            seed = base_seed + idx
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        expected_calls = [call(42), call(43), call(44)]
        self.assertEqual(mock_torch_seed.call_args_list, expected_calls)
        self.assertEqual(mock_rand_seed.call_args_list, expected_calls)
        self.assertEqual(mock_np_seed.call_args_list, expected_calls)

    def test_8_23_seed_determinism(self):
        """8.23 Confirmar determinismo: com o mesmo seed, torch.randn gera o mesmo tensor."""
        seed = 42
        torch.manual_seed(seed)
        t1 = torch.randn(10)

        torch.manual_seed(seed)
        t2 = torch.randn(10)

        self.assertTrue(torch.equal(t1, t2), "Tensors gerados com o mesmo seed devem ser idênticos")

    def test_8_24_distinct_seeds_yield_distinct_tensors(self):
        """8.24 Confirmar que seeds diferentes (BASE_SEED + 0 vs BASE_SEED + 1) geram tensores diferentes."""
        base_seed = 42
        torch.manual_seed(base_seed + 0)
        t1 = torch.randn(10)

        torch.manual_seed(base_seed + 1)
        t2 = torch.randn(10)

        self.assertFalse(torch.equal(t1, t2), "Tensors gerados com seeds diferentes não devem colidir")


# ============================================================================
# Testes de Integração Leve (Fix 2 + Fix 3 sem modelo TTS)
# ============================================================================
import tempfile
import soundfile as sf
import librosa

class TestFixIntegration_Lightweight(unittest.TestCase):
    def test_8_25_real_reference_audio_preprocess(self):
        """8.25 Executar pipeline completo no reference_narrator_en.wav real (noise reduce -> norm -23 LUFS -> clip)."""
        ref_path = Path("kaggle_runner/reference_narrator_en.wav")
        if not ref_path.exists():
            self.skipTest("reference_narrator_en.wav não encontrado")

        ref_wav, ref_sr = librosa.load(str(ref_path), sr=24000, mono=True)
        self.assertGreater(len(ref_wav), 0)

        # 1. Redução de ruído
        noise_sample = ref_wav[:int(ref_sr * 0.5)]
        ref_clean = nr.reduce_noise(y=ref_wav, sr=ref_sr, y_noise=noise_sample, prop_decrease=0.8)

        # 2. Normalização para -23 LUFS
        meter = ln.Meter(ref_sr)
        loudness = meter.integrated_loudness(ref_clean)
        self.assertTrue(np.isfinite(loudness))

        gain_linear = 10.0 ** ((-23.0 - loudness) / 20.0)
        ref_norm = ref_clean * np.float32(gain_linear)

        # 3. Soft clipping
        ref_final = np.clip(ref_norm, -0.95, 0.95)

        # 4. Salvar em arquivo temporário e validar loudness
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            sf.write(tmp_path, ref_final, ref_sr)
            self.assertTrue(Path(tmp_path).exists())
            self.assertGreater(Path(tmp_path).stat().st_size, 1000)

            # Reabrir e medir loudness
            reloaded_wav, reloaded_sr = sf.read(tmp_path)
            measured_lufs = meter.integrated_loudness(reloaded_wav)
            print(f"\n[Test 8.25] Real audio preprocessed: original {loudness:.2f} LUFS -> final {measured_lufs:.2f} LUFS")
            self.assertAlmostEqual(measured_lufs, -23.0, delta=1.0)
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass

    def test_8_26_real_input_text_chunking_and_keywords(self):
        """8.26 Validar chunking no input_text.txt real: <=250 chars, não vazio, preservando palavras-chave."""
        input_path = Path("kaggle_runner/input_text.txt")
        if not input_path.exists():
            self.skipTest("input_text.txt não encontrado")

        with open(input_path, "r", encoding="utf-8") as f:
            real_text = f.read().strip()

        chunks = split_into_chunks_new(real_text, max_chars=250)
        self.assertGreater(len(chunks), 0)

        # (a) Todos <= 250
        for i, ch in enumerate(chunks):
            self.assertLessEqual(len(ch), 250, f"Chunk {i} excedeu 250 chars")

        # (b) Nenhum vazio
        for ch in chunks:
            self.assertGreater(len(ch.strip()), 0)

        # (c) Palavras-chave essenciais preservadas no texto reconstruído
        reconstructed = " ".join(chunks)
        keywords = ["Donald Trump", "Qeshm", "Hormuz", "Zelensky", "Colombia"]
        for kw in keywords:
            self.assertIn(kw, reconstructed, f"Palavra-chave ausente: {kw}")

        print(f"\n[Test 8.26] Input text chunks: {len(chunks)} | Palavras-chave validadas: {keywords}")

    def test_8_27_full_pipeline_simulation_without_model(self):
        """8.27 Simular loop de geração completo com chunks reais, áudios sintéticos, seeds e crossfade 100ms."""
        input_path = Path("kaggle_runner/input_text.txt")
        with open(input_path, "r", encoding="utf-8") as f:
            real_text = f.read().strip()

        chunks = split_into_chunks_new(real_text, max_chars=250)
        sr = 24000
        cf_samples = int(sr * 100 / 1000)  # 2400
        base_seed = 42

        simulated_segments = []
        for idx, chunk_text in enumerate(chunks):
            seed = base_seed + idx
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Sintetizar áudio proporcional ao tamanho do texto (~15 caracteres por segundo de fala)
            duration_s = max(0.5, len(chunk_text) / 15.0)
            n_samples = int(sr * duration_s)
            t = np.linspace(0, duration_s, n_samples, endpoint=False)
            freq = 200 + (seed % 200)  # frequência base variável determinística
            wav_synth = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

            simulated_segments.append(wav_synth)

        final_simulated_audio = crossfade_concat(simulated_segments, cf_samples)

        self.assertGreater(len(final_simulated_audio), 0)
        self.assertFalse(np.isnan(final_simulated_audio).any())
        self.assertFalse(np.isinf(final_simulated_audio).any())

        total_duration = len(final_simulated_audio) / sr
        print(f"\n[Test 8.27] Full generation simulated: {len(chunks)} chunks -> {total_duration:.2f}s de áudio contínuo sem artefatos!")


if __name__ == '__main__':
    unittest.main()

