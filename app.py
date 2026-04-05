# -----------------------------
# Standard Library Imports
# -----------------------------
import io
import os
import threading
from threading import Lock

# -----------------------------
# Third‑Party Libraries
# -----------------------------
import torch
import whisper
from fastapi import FastAPI, File, UploadFile, Body
from peft import PeftModel
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BlipForConditionalGeneration,
    BlipProcessor,
)
from pydantic import BaseModel, Field
from typing import Any, Optional

# -----------------------------
# Local Project Modules
# -----------------------------
from behavior_logger import log_interaction
from memory import MemoryStore
from prompt_builder import build_messages
from conversation_store import conversations
from train_lora import train_lora_adapter

app = FastAPI()

# -----------------------------
# Load Base Model + Tokenizer
# -----------------------------
BASE_MODEL = "/models/qwen_model"
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    local_files_only=True
)
model.eval()

# -----------------------------
# Load LoRA Adapter (optional)
# -----------------------------
LORA_PATH = "lora_adapter"

def try_load_lora(base_model):
    adapter_config = os.path.join(LORA_PATH, "adapter_config.json")
    adapter_model = os.path.join(LORA_PATH, "adapter_model.safetensors")

    if os.path.exists(adapter_config) and os.path.exists(adapter_model):
        print("Found LoRA adapter. Loading...")
        return PeftModel.from_pretrained(base_model, LORA_PATH)
    else:
        print("No LoRA adapter found. Running base model only.")
        return base_model

model = try_load_lora(model)

# -----------------------------
# Load Image Captioning Model
# -----------------------------
BLIP_MODEL_PATH = os.environ.get("BLIP_MODEL_PATH", "/models/blip_model")

processor = BlipProcessor.from_pretrained(
    BLIP_MODEL_PATH,
    local_files_only=True
)

blip_model = BlipForConditionalGeneration.from_pretrained(
    BLIP_MODEL_PATH,
    local_files_only=True
)
blip_model.eval()

# -----------------------------
# Load Whisper for Audio
# -----------------------------
whisper_model = whisper.load_model("base")

# -----------------------------
# Memory Store
# -----------------------------
memory = MemoryStore()

# -----------------------------
# Training state
# -----------------------------
training_state = {
    "running": False,
    "last_status": "idle"
}

# -----------------------------
# Helper: Generate from Messages (Qwen chat-template)
# -----------------------------
generation_lock = Lock()  # prevent concurrent generate() on same model instance

def generate_from_messages(messages: list[dict]) -> str:
    # Render messages using the model's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )  # Qwen docs recommend this pattern [1](https://qwen.readthedocs.io/en/v2.0/inference/chat.html)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with generation_lock:
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

    # Slice off the prompt tokens (recommended in Qwen docs) [1](https://qwen.readthedocs.io/en/v2.0/inference/chat.html)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    assistant_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    print("Generated text:", assistant_text)
    return assistant_text

# -----------------------------
# Helper: Generate Text
# -----------------------------
def generate_text(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        do_sample=True
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Strip the prompt from the output
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()

    return decoded

# -----------------------------
# Endpoint: Text Generation
# -----------------------------
@app.post("/generate")
async def generate(user_text: str = Body(...), session_id: str = Body("default")):
    session_messages = conversations.get(session_id)

    memory_items = memory.retrieve(user_text)
    build_messages(
        session_messages=session_messages,
        user_text=user_text,
        image_description=None,
        voice_text=None,
        memory_items=memory_items
    )

    output = generate_from_messages(session_messages)

    # Append assistant reply for the next turn (core multi-turn step) [1](https://qwen.readthedocs.io/en/v2.0/inference/chat.html)
    session_messages.append({"role": "assistant", "content": output})

    log_interaction(user_text, None, None, output)
    return {"session_id": session_id, "generated_text": output}

# -----------------------------
# Endpoint: Image Input
# -----------------------------
@app.post("/generate-image")
async def generate_from_image(file: UploadFile = File(...), session_id: str = "default"):
    session_messages = conversations.get(session_id)

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    caption_ids = blip_model.generate(**inputs, max_length=64)
    image_description = processor.decode(caption_ids[0], skip_special_tokens=True)

    build_messages(
        session_messages=session_messages,
        user_text=None,
        image_description=image_description,
        voice_text=None,
        memory_items=[]
    )

    output = generate_from_messages(session_messages)
    session_messages.append({"role": "assistant", "content": output})

    log_interaction(None, image_description, None, output)
    return {"session_id": session_id, "image_description": image_description, "generated_text": output}

# -----------------------------
# Endpoint: Voice Input
# -----------------------------
@app.post("/generate-voice")
async def generate_from_voice(file: UploadFile = File(...), session_id: str = "default"):
    session_messages = conversations.get(session_id)

    audio_bytes = await file.read()
    audio_path = "temp_audio.wav"
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    result = whisper_model.transcribe(audio_path)
    voice_text = result["text"]

    build_messages(
        session_messages=session_messages,
        user_text=None,
        image_description=None,
        voice_text=voice_text,
        memory_items=[]
    )

    output = generate_from_messages(session_messages)
    session_messages.append({"role": "assistant", "content": output})

    log_interaction(None, None, voice_text, output)
    return {"session_id": session_id, "transcription": voice_text, "generated_text": output}

# -----------------------------
# Endpoint: Reset Session
# -----------------------------
@app.post("/reset-session")
async def reset_session(session_id: str = Body(...)):
    conversations.clear(session_id)
    return {"status": "cleared", "session_id": session_id}

class MemoryAddRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    metadata: dict[str, Any] = Field(default_factory=dict)

class MemoryAddResponse(BaseModel):
    id: str

class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)

class MemorySearchItem(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: Optional[float] = None

class MemoryListResponse(BaseModel):
    items: list[dict[str, Any]]
    limit: int
    offset: int

@app.post("/memory/memory_add", response_model=MemoryAddResponse)
async def memory_add(req: MemoryAddRequest):
    memory_id = memory.add(req.text, req.metadata)
    return MemoryAddResponse(id=memory_id)

@app.post("/memory/memory_search", response_model=list[MemorySearchItem])
async def memory_search(req: MemorySearchRequest):
    results = memory.search(req.query, top_k=req.top_k)
    # results already shaped as dicts; coerce to Pydantic output
    return [MemorySearchItem(**r) for r in results]

@app.get("/memory/memory_list", response_model=MemoryListResponse)
async def memory_list(limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    items = memory.list_memories(limit=limit, offset=offset)
    return MemoryListResponse(items=items, limit=limit, offset=offset)

@app.delete("/memory/{memory_id}")
async def memory_delete(memory_id: str):
    # Optional: check existence first
    existing = memory.get(memory_id)
    if existing is None:
        return {"status": "not_found", "id": memory_id}
    memory.delete(memory_id)
    return {"status": "deleted", "id": memory_id}

# -----------------------------
# Endpoint: Train LoRA
# -----------------------------
@app.post("/train-now")
async def train_now():
    if training_state["running"]:
        return {"status": "training already in progress"}

    training_state["running"] = True
    training_state["last_status"] = "training started"

    def run_training():
        try:
            train_lora_adapter()
            training_state["last_status"] = "training complete"
        except Exception as e:
            training_state["last_status"] = f"error: {str(e)}"
        finally:
            training_state["running"] = False

    threading.Thread(target=run_training).start()

    return {"status": "training started"}

# -----------------------------
# Endpoint: Training status
# -----------------------------
@app.get("/training-status")
async def training_status():
    return {
        "running": training_state["running"],
        "status": training_state["last_status"]
    }

# -----------------------------
# Endpoint: Reload LoRA
# -----------------------------
@app.post("/reload-lora")
async def reload_lora():
    global model
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
    model = try_load_lora(base)
    return {"status": "LoRA reloaded (or skipped if none found)"}

# -----------------------------
# Endpoint: Train and Reload LoRA
# -----------------------------
@app.post("/train-and-reload")
async def train_and_reload():
    if training_state["running"]:
        return {"status": "training already in progress"}

    training_state["running"] = True
    training_state["last_status"] = "training started"

    def run_training_and_reload():
        try:
            train_lora_adapter()
            training_state["last_status"] = "training complete, reloading LoRA"

            # Reload LoRA adapter
            global model
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16)
            if os.path.exists(LORA_PATH):
                model = PeftModel.from_pretrained(base, LORA_PATH)
            training_state["last_status"] = "training complete and LoRA reloaded"

        except Exception as e:
            training_state["last_status"] = f"error: {str(e)}"
        finally:
            training_state["running"] = False

    threading.Thread(target=run_training_and_reload).start()

    return {"status": "training started"}