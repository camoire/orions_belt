# -----------------------------
# Standard Library Imports
# -----------------------------
import io
import gc
import json
import os
import threading
import time
import uuid
from threading import Lock

# -----------------------------
# Third‑Party Libraries
# -----------------------------
import whisper
from fastapi import FastAPI, File, HTTPException, UploadFile, Body
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from PIL import Image
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
)
from pydantic import BaseModel, Field
from typing import Any, Iterator, Literal, Optional

# -----------------------------
# Local Project Modules
# -----------------------------
from behavior_logger import log_interaction
from memory import MemoryStore
from prompt_builder import build_messages
from conversation_store import conversations
from train_lora import train_lora_adapter

app = FastAPI()
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "orions-belt")

# -----------------------------
# Load Qwen GGUF model
# -----------------------------
LLAMA_MODEL_PATH = os.environ.get(
    "QWEN_GGUF_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "qwen_model",
        "Qwen2.5-Coder-14B-Instruct-Q6_K.gguf",
    ),
)
LLAMA_LORA_PATH = os.environ.get("LLAMA_LORA_PATH")

def load_qwen_model():
    model_options = {
        "model_path": LLAMA_MODEL_PATH,
        "n_ctx": int(os.environ.get("LLAMA_N_CTX", "8192")),
        "n_gpu_layers": int(os.environ.get("LLAMA_N_GPU_LAYERS", "0")),
        "verbose": False,
    }
    if LLAMA_LORA_PATH:
        model_options["lora_path"] = LLAMA_LORA_PATH
    return Llama(**model_options)

model = load_qwen_model()

# -----------------------------
# Load Image Captioning Model
# -----------------------------
BLIP_MODEL_PATH = os.environ.get(
    "BLIP_MODEL_PATH", os.path.join(os.path.dirname(__file__), "blip_model")
)

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
# Helper: Generate from chat messages
# -----------------------------
generation_lock = Lock()  # prevent concurrent generate() on same model instance

def generate_from_messages(messages: list[dict]) -> str:
    with generation_lock:
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Qwen GGUF model is unavailable; reload the model and try again.",
            )
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=256,
            temperature=0.7,
        )
    assistant_text = (result["choices"][0]["message"]["content"] or "").strip()
    return assistant_text


class OpenAIMessage(BaseModel):
    role: Literal["system", "developer", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[OpenAIMessage] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    stop: Optional[str | list[str]] = None
    stream: bool = False


def prepare_openai_messages(messages: list[OpenAIMessage]) -> tuple[list[dict], str]:
    prompt_messages = [message.dict() for message in messages]
    user_text = next(
        (
            message["content"]
            for message in reversed(prompt_messages)
            if message["role"] == "user"
        ),
        ""
    )
    if user_text:
        memory_items = memory.retrieve(user_text)
        if memory_items:
            memory_context = "Relevant memory (untrusted context):\n" + "\n".join(
                f"- {item}" for item in memory_items
            )
            system_message = next(
                (message for message in prompt_messages if message["role"] in ("system", "developer")),
                None,
            )
            if system_message:
                system_message["content"] += "\n\n" + memory_context
            else:
                prompt_messages.insert(0, {"role": "system", "content": memory_context})
    return prompt_messages, user_text


def openai_stream(
    messages: list[dict],
    user_text: str,
    completion_id: str,
    created: int,
    model_name: str,
    request: ChatCompletionRequest,
) -> Iterator[str]:
    output_parts = []
    finish_reason = "stop"
    with generation_lock:
        if model is None:
            raise RuntimeError("Qwen GGUF model is unavailable.")
        chunks = model.create_chat_completion(
            messages=messages,
            max_tokens=request.max_completion_tokens or request.max_tokens or 256,
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 1.0,
            stop=request.stop,
            stream=True,
        )
        role_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(role_chunk)}\n\n"
        for chunk in chunks:
            choice = chunk["choices"][0]
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content:
                output_parts.append(content)
                response_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {"content": content}, "finish_reason": None}
                    ],
                }
                yield f"data: {json.dumps(response_chunk)}\n\n"
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]

    output = "".join(output_parts)
    log_interaction(user_text, None, None, output)
    finish_chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    yield f"data: {json.dumps(finish_chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
async def openai_list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": OPENAI_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "orions-belt",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatCompletionRequest):
    messages, user_text = prepare_openai_messages(request.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Qwen GGUF model is unavailable; reload the model and try again.",
        )

    if request.stream:
        return StreamingResponse(
            openai_stream(
                messages,
                user_text,
                completion_id,
                created,
                OPENAI_MODEL_ID,
                request,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    with generation_lock:
        result = model.create_chat_completion(
            messages=messages,
            max_tokens=request.max_completion_tokens or request.max_tokens or 256,
            temperature=request.temperature if request.temperature is not None else 0.7,
            top_p=request.top_p if request.top_p is not None else 1.0,
            stop=request.stop,
        )
    choice = result["choices"][0]
    output = (choice["message"]["content"] or "").strip()
    log_interaction(user_text, None, None, output)
    usage = result.get("usage") or {}
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": OPENAI_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": choice.get("finish_reason") or "stop",
            }
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }

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
def reload_lora():
    global model
    with generation_lock:
        try:
            old_model = model
            model = None
            if old_model is not None:
                old_model.close()
            gc.collect()
            model = load_qwen_model()
        except Exception as exc:
            print(f"GGUF model reload failed: {exc}")
            raise HTTPException(
                status_code=500,
                detail="Could not reload GGUF model.",
            ) from exc
    if LLAMA_LORA_PATH:
        return {"status": "GGUF model reloaded with llama.cpp LoRA adapter"}
    return {"status": "GGUF model reloaded without a LoRA adapter"}

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
            training_state["last_status"] = (
                "training complete; Transformers adapters are not compatible with "
                "the GGUF inference model"
            )

        except Exception as e:
            training_state["last_status"] = f"error: {str(e)}"
        finally:
            training_state["running"] = False

    threading.Thread(target=run_training_and_reload).start()

    return {"status": "training started"}