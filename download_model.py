import os
from pathlib import Path
from huggingface_hub import hf_hub_download

repo_id = "ResembleAI/chatterbox-turbo"
files = [
    "added_tokens.json",
    "conds.pt",
    "merges.txt",
    "s3gen_meanflow.safetensors",
    "special_tokens_map.json",
    "t3_turbo_v1.safetensors",
    "t3_turbo_v1.yaml",
    "tokenizer_config.json",
    "ve.safetensors",
    "vocab.json"
]

dest_dir = Path("model_weights")
dest_dir.mkdir(exist_ok=True)

print(f"Starting download of {len(files)} files to {dest_dir.absolute()}...")
for file in files:
    print(f"Downloading {file}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=file,
            local_dir=dest_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded {file} to {path}")
    except Exception as e:
        print(f"Error downloading {file}: {e}")

print("Download process completed.")
