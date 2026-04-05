import os
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def main():
    print("=== Qwen Model Downloader ===")

    # Read HF token from environment
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⚠️  No HF_TOKEN provided. Downloads may be slow or throttled.")
        print("   Set it like this before running:")
        print("   PowerShell:  $env:HF_TOKEN = 'hf_xxx'")
        print("   Bash:        export HF_TOKEN=hf_xxx")
    else:
        print("✓ Using Hugging Face token for authenticated download.")

    print("\n=== Downloading model and tokenizer ===")

    # Trigger download into HF cache
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=token)

    print("✓ Download complete.")

    # Locate HF cache directory
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = None

    # Find the model folder inside the cache
    for item in hf_cache.iterdir():
        if item.name.startswith("models--Qwen--Qwen2.5-7B-Instruct"):
            model_dir = item
            break

    if model_dir is None:
        raise RuntimeError("❌ Could not locate downloaded model in Hugging Face cache.")

    print(f"✓ Found model directory in cache:\n  {model_dir}")

    # Destination inside your project
    dest_root = Path("qwen_model")
    dest_root.mkdir(exist_ok=True)

    target = dest_root / model_dir.name

    # Remove old copy if present
    if target.exists():
        print("⚠️ Existing model folder found. Removing it first...")
        shutil.rmtree(target)

    print(f"\n=== Copying model into project folder ===")
    print(f"→ {target}")

    shutil.copytree(model_dir, target)

    print("\n✓ Model copied successfully.")
    print("You can now build your Docker image without downloading the model again.")

if __name__ == "__main__":
    main()
