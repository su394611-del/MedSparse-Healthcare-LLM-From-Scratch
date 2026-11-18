"""Upload native .pt checkpoint and tokenizer, not an AutoModel-compatible HF model."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True)
    p.add_argument("--checkpoint", default="checkpoints/run/last.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--config", required=True)
    p.add_argument("--private", action="store_true")
    a = p.parse_args()
    paths = [Path(a.checkpoint), Path(a.tokenizer), Path(a.config)]
    for path in paths:
        if not path.is_file():
            p.error(f"Missing {path}")
    api = HfApi()
    api.create_repo(repo_id=a.repo_id, private=a.private, exist_ok=True)
    for path, target in zip(paths, ("model.pt", "tokenizer.json", "model_config.json")):
        api.upload_file(path_or_fileobj=str(path), path_in_repo=target, repo_id=a.repo_id)
    api.upload_folder(folder_path="mixtral_lab", path_in_repo="mixtral_lab",
                      repo_id=a.repo_id, ignore_patterns=["__pycache__/*"])
    api.upload_file(path_or_fileobj="MODEL_CARD.md", path_in_repo="README.md", repo_id=a.repo_id)
    print(f"Published native project files to https://huggingface.co/{a.repo_id}")
    print("Not Transformers AutoModel compatible; use the provided Python class and loader.")


if __name__ == "__main__":
    main()
