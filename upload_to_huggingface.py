"""Paste this entire script into a Colab cell after training, or run it with Python.

Uses HF_TOKEN from Colab Secrets/environment or your saved Hugging Face login;
otherwise prompts privately for a token with write access. No token belongs here.
"""
from getpass import getpass
from pathlib import Path
import json

# Settings — the training notebook exports this folder automatically.
MODEL_DIR = Path("female-authors/best_model")
REPO_ID = ""  # Blank: your authenticated username/FemaleAuthorsScratch.
PRIVATE = True  # New repositories are private; False creates a public repository.


def upload_model(model_dir=MODEL_DIR, repo_id=REPO_ID, private=PRIVATE, token=None):
    from huggingface_hub import HfApi, get_token

    model_dir = Path(model_dir)
    filenames = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards or any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
            raise ValueError("Invalid model shard index.")
        filenames += [index.name, *sorted(shards)]
    else:
        filenames.append("model.safetensors")
    missing = [name for name in filenames if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"No complete model export at {model_dir}: missing {', '.join(missing)}. Run training first.")
    filenames += [name for name in ("generation_config.json", "special_tokens_map.json",
                  "added_tokens.json", "vocab.json", "merges.txt", "README.md", "LICENSE")
                  if (model_dir / name).is_file()]

    token = token or get_token() or getpass("Hugging Face write token: ").strip()
    if not token:
        raise ValueError("A Hugging Face token with write access is required.")
    api = HfApi(token=token)
    repo_id = repo_id.strip() or f"{api.whoami()['name']}/FemaleAuthorsScratch"
    # Existing repositories retain their current visibility.
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(model_dir),
        allow_patterns=filenames,  # Only inference files, never optimizer state or corpus.
        commit_message="Upload trained FemaleAuthorsScratch model and tokenizer",
    )
    print(f"Model: https://huggingface.co/{repo_id}")
    print(f"Upload: {commit.commit_url}")
    return commit


if __name__ == "__main__":
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub>=1.5,<2"], check=True)
    upload_model()
