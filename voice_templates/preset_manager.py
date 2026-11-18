import os
import json
import shutil
from pathlib import Path

# Diretórios padrão
BASE_DIR = Path(__file__).resolve().parent.parent
VOICE_TEMPLATES_DIR = BASE_DIR / "voice_templates"
REFS_DIR = VOICE_TEMPLATES_DIR / "refs"

VOICE_TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
REFS_DIR.mkdir(parents=True, exist_ok=True)


def list_presets() -> list[dict]:
    """Lê e retorna a lista de todos os presets de voz salvos."""
    presets = []
    for file_path in VOICE_TEMPLATES_DIR.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["_file_stem"] = file_path.stem
                presets.append(data)
        except Exception as e:
            print(f"Erro ao carregar preset {file_path}: {e}")

    # Ordenar por nome
    presets.sort(key=lambda x: x.get("name", "").lower())
    return presets


def get_preset_names() -> list[str]:
    """Retorna apenas os nomes dos presets para seletores/dropdowns."""
    presets = list_presets()
    return [p["name"] for p in presets]


def load_preset(name: str) -> dict | None:
    """Busca um preset pelo nome e retorna suas configurações."""
    for preset in list_presets():
        if preset.get("name", "").strip().lower() == name.strip().lower():
            return preset
    return None


def save_preset(name: str, description: str, params: dict, audio_ref_path: str | None = None, conds_pt_path: str | None = None) -> dict:
    """
    Salva um novo preset ou atualiza um existente.
    - audio_ref_path: ficheiro WAV/MP3 de referência (copiado para refs/)
    - conds_pt_path: embeddings pré-computados (.pt) para carga instantânea
    """
    if not name or not name.strip():
        raise ValueError("O nome do personagem/preset não pode ser vazio.")

    clean_name = name.strip()
    safe_filename = "".join(c for c in clean_name if c.isalnum() or c in (" ", "_", "-")).strip().replace(" ", "_")
    json_path = VOICE_TEMPLATES_DIR / f"{safe_filename}.json"

    saved_ref_path = None
    if audio_ref_path and os.path.exists(audio_ref_path):
        ext = Path(audio_ref_path).suffix or ".wav"
        target_ref_file = REFS_DIR / f"{safe_filename}{ext}"
        shutil.copy2(audio_ref_path, target_ref_file)
        saved_ref_path = str(target_ref_file)

    preset_data = {
        "name": clean_name,
        "description": description or "",
        "audio_ref": saved_ref_path,
        "conds_pt": str(conds_pt_path) if conds_pt_path else None,
        "params": params,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(preset_data, f, indent=2, ensure_ascii=False)

    return preset_data


def delete_preset(name: str) -> bool:
    """Deleta um preset pelo nome."""
    preset = load_preset(name)
    if not preset:
        return False

    file_stem = preset.get("_file_stem")
    if file_stem:
        json_path = VOICE_TEMPLATES_DIR / f"{file_stem}.json"
        if json_path.exists():
            json_path.unlink()

        # Opcional: Se houver áudio ref específico no diretório refs/
        if preset.get("audio_ref"):
            ref_path = Path(preset["audio_ref"])
            if ref_path.exists() and REFS_DIR in ref_path.parents:
                try:
                    ref_path.unlink()
                except Exception:
                    pass
        return True
    return False
