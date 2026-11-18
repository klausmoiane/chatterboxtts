"""
voice_profiles.py - Gerenciador de Perfis e Personagens de Voz para o pipeline Chatterbox TTS.

Responsabilidades:
  - Definir os perfis de voz disponíveis com seus hiperparâmetros padrão.
  - Permitir customização interativa dos parâmetros (temperature, repetition_penalty, seed).
  - Serializar/deserializar perfis para o formato consumido pelo app_bridge.py (voice_config.json).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import json

ROOT_DIR = Path(__file__).resolve().parent
KAGGLE_RUNNER_DIR = ROOT_DIR / "kaggle_runner"
DATASET_VOICE_DIR = KAGGLE_RUNNER_DIR / "dataset_voice"


# ------------------------------------------------------------------ #
#  Dataclass de Perfil                                                #
# ------------------------------------------------------------------ #

@dataclass
class VoiceProfile:
    """Representa um preset completo de voz para o modelo Chatterbox TTS."""

    # Identificação
    id: str
    name: str
    description: str
    language: str                           # "en" | "pt" | ...

    # Referência de áudio para clonagem de voz (None = usa voz padrão do modelo)
    ref_audio: Optional[str] = None         # caminho relativo à ROOT_DIR

    # Hiperparâmetros do modelo
    temperature: float = 0.8                # criatividade; 0.1 = conservador, 1.0 = criativo
    repetition_penalty: float = 1.1         # penalidade de repetição de tokens
    seed: int = 42                          # semente para reprodutibilidade (-1 = aleatório)

    # Metadados extras (exibição na UI)
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # ---------------------------------------------------------------- #

    def to_dict(self) -> dict:
        """Serializa para dicionário (usado pelo app_bridge para voice_config.json)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "VoiceProfile":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def with_overrides(
        self,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> "VoiceProfile":
        """Retorna uma cópia do perfil com os parâmetros substituídos."""
        import copy
        clone = copy.deepcopy(self)
        if temperature is not None:
            clone.temperature = temperature
        if repetition_penalty is not None:
            clone.repetition_penalty = repetition_penalty
        if seed is not None:
            clone.seed = seed
        return clone


# ------------------------------------------------------------------ #
#  Perfis pré-definidos                                               #
# ------------------------------------------------------------------ #

_REF_NARRATOR_EN = str(DATASET_VOICE_DIR / "reference_narrator_en.wav")

PROFILES: dict[str, VoiceProfile] = {

    "narrator_en": VoiceProfile(
        id="narrator_en",
        name="🎙️ Narrador Documental EN",
        description=(
            "Voz clonada a partir do trecho 02:43–03:12 do vídeo "
            "'Why Putin Is So Hard to Kill'. Tom sério, ritmo médio, "
            "sotaque neutro americano. Ideal para documentários e narração jornalística."
        ),
        language="en",
        ref_audio=_REF_NARRATOR_EN,
        temperature=0.75,
        repetition_penalty=1.1,
        seed=42,
        tags=["documentary", "english", "cloned"],
        notes="Referência: Why Putin Is So Hard to Kill — Mr. Science",
    ),

    "chatterbox_default": VoiceProfile(
        id="chatterbox_default",
        name="🤖 Voz Padrão Chatterbox",
        description=(
            "Voz interna padrão do modelo Chatterbox TTS, sem clonagem de referência. "
            "Boa para testes rápidos e quando não há áudio de referência disponível."
        ),
        language="en",
        ref_audio=None,
        temperature=0.8,
        repetition_penalty=1.1,
        seed=42,
        tags=["default", "english", "no-clone"],
        notes="Nenhum áudio de referência — usa a voz embutida do checkpoint.",
    ),

    "narrator_pt_placeholder": VoiceProfile(
        id="narrator_pt_placeholder",
        name="🇧🇷 Narrador PT (em breve)",
        description=(
            "Espaço reservado para preset em Português Brasileiro. "
            "Adicione um áudio de referência em PT-BR para ativar este perfil."
        ),
        language="pt",
        ref_audio=None,
        temperature=0.8,
        repetition_penalty=1.1,
        seed=42,
        tags=["portuguese", "placeholder"],
        notes="Perfil ainda não configurado. Adicione ref_audio para ativar.",
    ),
}


# ------------------------------------------------------------------ #
#  API pública                                                        #
# ------------------------------------------------------------------ #

def list_profiles() -> list[VoiceProfile]:
    """Retorna todos os perfis disponíveis como lista ordenada."""
    return list(PROFILES.values())


def get_profile(profile_id: str) -> VoiceProfile:
    """Retorna um perfil pelo ID. Lança KeyError se não existir."""
    if profile_id not in PROFILES:
        raise KeyError(f"Perfil '{profile_id}' não encontrado. Disponíveis: {list(PROFILES.keys())}")
    return PROFILES[profile_id]


def get_profile_choices() -> list[tuple[str, str]]:
    """
    Retorna lista de (label, id) para uso em componentes de seleção da UI.
    Exemplo: [("🎙️ Narrador Documental EN", "narrator_en"), ...]
    """
    return [(p.name, p.id) for p in list_profiles()]


def build_voice_config(
    profile_id: str,
    temperature: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Constrói o dicionário de configuração final, aplicando sobrescritas opcionais
    sobre o perfil base. Este dicionário é o que vai para voice_config.json.

    Parâmetros:
      profile_id         : ID do perfil base (ver PROFILES).
      temperature        : sobrescreve o valor do perfil se fornecido.
      repetition_penalty : sobrescreve o valor do perfil se fornecido.
      seed               : sobrescreve o valor do perfil se fornecido.
    """
    profile = get_profile(profile_id)
    final = profile.with_overrides(
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        seed=seed,
    )
    return final.to_dict()


def save_custom_profile(profile: VoiceProfile, path: Optional[Path] = None) -> Path:
    """
    Persiste um perfil customizado em disco como JSON.
    Padrão: ROOT_DIR/voice_profiles/<id>.json
    """
    profiles_dir = path or (ROOT_DIR / "voice_profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)
    out = profiles_dir / f"{profile.id}.json"
    out.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_custom_profile(path: Path) -> VoiceProfile:
    """Carrega um perfil customizado de um arquivo JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return VoiceProfile.from_dict(data)


# ------------------------------------------------------------------ #
#  Constantes de intervalo para sliders da UI                        #
# ------------------------------------------------------------------ #

PARAM_RANGES = {
    "temperature": {
        "min": 0.1,
        "max": 1.5,
        "step": 0.05,
        "help": (
            "Controla a variabilidade da síntese. "
            "Valores baixos (0.1–0.5) são mais estáveis e consistentes; "
            "valores altos (0.9–1.5) introduzem mais expressividade e variação."
        ),
    },
    "repetition_penalty": {
        "min": 1.0,
        "max": 2.0,
        "step": 0.05,
        "help": (
            "Penaliza repetições de tokens de áudio. "
            "1.0 = sem penalidade; valores acima de 1.2 reduzem gagueiras e loops."
        ),
    },
    "seed": {
        "min": -1,
        "max": 99999,
        "step": 1,
        "help": (
            "Semente aleatória para reprodutibilidade. "
            "-1 gera um resultado diferente a cada execução."
        ),
    },
}
