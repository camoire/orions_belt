import os
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
QWEN_REPO = "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"
QWEN_FILE = "qwen2.5-coder-7b-instruct-q5_k_m.gguf"
BLIP_REPO = "Salesforce/blip-image-captioning-base"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  No HF_TOKEN provided. Downloads may be slow or throttled.")
        print("   Set it like this before running:")
        print("   PowerShell:  $env:HF_TOKEN = 'hf_xxx'")
        print("   Bash:        export HF_TOKEN=hf_xxx")
    else:
        print("✓ Using Hugging Face token for authenticated download.")

    downloads = (
        (QWEN_REPO, BASE_DIR / "qwen_model", [QWEN_FILE]),
        (BLIP_REPO, BASE_DIR / "blip_model", None),
    )
    for repo_id, destination, patterns in downloads:
        print(f"\n=== Downloading {repo_id} to {destination} ===")
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            allow_patterns=patterns,
            token=token,
        )
        print(f"✓ {repo_id} downloaded.")

    print("\nModels are ready for local use and Docker builds.")

if __name__ == "__main__":
    main()
