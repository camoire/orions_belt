import json
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

LOG_PATH = BASE_DIR / "logs"
LOG_PATH.mkdir(exist_ok=True)
LOG_FILE = LOG_PATH / "interactions.jsonl"

def log_interaction(
    user_text: str | None,
    image_description: str | None,
    voice_text: str | None,
    model_output: str
):
    """
    Logs multimodal interactions in a structured JSONL format.
    Each line is a training-ready record.
    """

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user_text": user_text or "",
        "image_description": image_description or "",
        "voice_text": voice_text or "",
        "model_output": model_output,
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print("LOGGING ERROR:", e)
