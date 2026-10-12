import json
from pathlib import Path
from prompt_builder import build_prompt
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "logs"
LOG_PATH.mkdir(exist_ok=True)
LOG_FILE = LOG_PATH / "interactions.jsonl"

DATASET_DIR = BASE_DIR / "dataset"
DATASET_DIR.mkdir(exist_ok=True)

def build_dataset():
    if not LOG_FILE.exists():
        print("No interaction logs found. Generate interactions before building a dataset.")
        return

    records = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            prompt = build_prompt(
                user_text=entry.get("user_text"),
                image_description=entry.get("image_description"),
                voice_text=entry.get("voice_text"),
                memory_items=[]  # memory not stored in logs
            )

            records.append({
                "prompt": prompt,
                "response": entry["model_output"]
            })

    if len(records) < 2:
        raise ValueError("At least two logged interactions are required to build train and validation sets.")

    val_size = max(1, round(len(records) * 0.1))
    val_size = min(val_size, len(records) - 1)
    train, val = train_test_split(records, test_size=val_size, random_state=42)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")

    with open(DATASET_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")

    print("Dataset built successfully.")

if __name__ == "__main__":
    build_dataset()
