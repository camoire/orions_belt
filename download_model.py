import os
from pathlib import Path
from huggingface_hub import snapshot_download

BASE_DIR = Path(__file__).resolve().parent
MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen_model",
    "Salesforce/blip-image-captioning-base": "blip_model",
}

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  No HF_TOKEN provided. Downloads may be slow or throttled.")
        print("   Set it like this before running:")
        print("   PowerShell:  $env:HF_TOKEN = 'hf_xxx'")
        print("   Bash:        export HF_TOKEN=hf_xxx")
    else:
        print("✓ Using Hugging Face token for authenticated download.")

    for model_name, directory in MODELS.items():
        destination = BASE_DIR / directory
        print(f"\n=== Downloading {model_name} to {destination} ===")
        snapshot_download(
            repo_id=model_name,
            local_dir=destination,
            token=token,
        )
        print(f"✓ {model_name} downloaded.")

    print("\nModels are ready for local use and Docker builds.")

if __name__ == "__main__":
    main()
