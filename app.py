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
from memory import MemoryStore, MemoryExtractor
from prompt_builder import build_messages
from conversation_store import conversations
from train_lora import train_lora_adapter

app = FastAPI()
OPENAI_MODEL_ID = os.environ.get("OPENAI_MODEL_ID", "").strip() or "orions-belt"

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
# Generation Lock & Memory Store
# -----------------------------
generation_lock = Lock()  # prevent concurrent generate() on same model instance

memory = MemoryStore()
memory_extractor = MemoryExtractor(model=model, generation_lock=generation_lock)

def extract_memories_background(user_text: str | None, assistant_text: str | None = None):
    if not user_text:
        return
    try:
        memory_extractor.extract_and_store(
            memory_store=memory,
            user_text=user_text,
            assistant_text=assistant_text,
            source="automatic_extraction",
        )
    except Exception as exc:
        print("Automatic memory extraction failed:", exc)

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
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: Optional[str | list[dict[str, Any]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    function_call: Optional[dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    messages: list[OpenAIMessage] = Field(..., min_length=1)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(default=None, ge=0, le=20)
    temperature: Optional[float] = Field(default=None, ge=0, le=2)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    n: Optional[int] = Field(default=1, ge=1, le=128)
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    stop: Optional[str | list[str]] = None
    stream: bool = False
    seed: Optional[int] = None
    response_format: Optional[dict[str, Any]] = None
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[str | dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    functions: Optional[list[dict[str, Any]]] = None
    function_call: Optional[str | dict[str, Any]] = None
    stream_options: Optional[dict[str, Any]] = None
    user: Optional[str] = None


def prepare_openai_messages(messages: list[OpenAIMessage]) -> tuple[list[dict], str]:
    prompt_messages = []
    for message in messages:
        item = (
            message.model_dump(exclude_none=True)
            if hasattr(message, "model_dump")
            else message.dict(exclude_none=True)
        )
        content = item.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") in ("text", "input_text") and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                else:
                    raise HTTPException(
                        status_code=400,
                        detail="Only text message content is supported by this model.",
                    )
            item["content"] = "".join(text_parts)
        prompt_messages.append(item)
    user_text = next(
        (
            message["content"]
            for message in reversed(prompt_messages)
            if message["role"] == "user" and isinstance(message.get("content"), str)
        ),
        ""
    )
    if user_text:
        # Retrieve top 5 memories ranked by relevance and importance score
        memory_items = memory.retrieve(user_text, top_k=5)
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


def openai_model_not_found(model_name: str):
    raise HTTPException(status_code=404, detail=f"The model '{model_name}' does not exist.")


def chat_completion_options(request: ChatCompletionRequest) -> dict[str, Any]:
    options: dict[str, Any] = {
        "max_tokens": request.max_completion_tokens or request.max_tokens or 256,
        "temperature": request.temperature if request.temperature is not None else 0.7,
        "top_p": request.top_p if request.top_p is not None else 1.0,
    }
    for field in (
        "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
        "n", "presence_penalty", "seed", "response_format", "tools",
        "tool_choice", "functions", "function_call",
    ):
        value = getattr(request, field)
        if value is not None:
            options[field] = value
    if request.stop is not None:
        options["stop"] = request.stop
    return options


def openai_stream(
    messages: list[dict],
    user_text: str,
    completion_id: str,
    created: int,
    model_name: str,
    request: ChatCompletionRequest,
) -> Iterator[str]:
    output_parts: dict[int, list[str]] = {}
    finish_reasons: dict[int, str] = {}
    with generation_lock:
        if model is None:
            raise RuntimeError("Qwen GGUF model is unavailable.")
        chunks = model.create_chat_completion(
            messages=messages, stream=True, **chat_completion_options(request)
        )
        for choice_index in range(request.n or 1):
            role_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": choice_index,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(role_chunk)}\n\n"
        for chunk in chunks:
            for choice in chunk["choices"]:
                choice_index = choice.get("index", 0)
                delta = choice.get("delta") or {}
                if delta.get("role") == "assistant":
                    delta = {key: value for key, value in delta.items() if key != "role"}
                if delta:
                    content = delta.get("content")
                    if content:
                        output_parts.setdefault(choice_index, []).append(content)
                    response_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [
                            {"index": choice_index, "delta": delta, "finish_reason": None}
                        ],
                    }
                    yield f"data: {json.dumps(response_chunk)}\n\n"
                if choice.get("finish_reason"):
                    finish_reasons[choice_index] = choice["finish_reason"]

    for choice_index in range(request.n or 1):
        output = "".join(output_parts.get(choice_index, []))
        log_interaction(user_text, None, None, output)
        threading.Thread(
            target=extract_memories_background,
            args=(user_text, output),
            daemon=True,
        ).start()
        finish_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": choice_index,
                "delta": {},
                "finish_reason": finish_reasons.get(choice_index, "stop"),
            }],
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


@app.get("/v1/models/{model_id}")
async def openai_retrieve_model(model_id: str):
    if model_id != OPENAI_MODEL_ID:
        openai_model_not_found(model_id)
    return {
        "id": OPENAI_MODEL_ID,
        "object": "model",
        "created": 0,
        "owned_by": "orions-belt",
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: ChatCompletionRequest):
    if request.model != OPENAI_MODEL_ID:
        openai_model_not_found(request.model)
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
            messages=messages, **chat_completion_options(request)
        )
    output_choices = result.get("choices", [])
    if not output_choices:
        raise HTTPException(status_code=502, detail="The model returned no completion choices.")
    output = (output_choices[0].get("message", {}).get("content") or "")
    output = output.strip() if isinstance(output, str) else ""
    log_interaction(user_text, None, None, output)
    threading.Thread(
        target=extract_memories_background,
        args=(user_text, output),
        daemon=True,
    ).start()
    usage = result.get("usage") or {}
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": OPENAI_MODEL_ID,
        "choices": [
            {
                "index": choice.get("index", index),
                "message": choice.get("message", {"role": "assistant", "content": output}),
                "finish_reason": choice.get("finish_reason") or "stop",
                **({"logprobs": choice["logprobs"]} if "logprobs" in choice else {}),
            }
            for index, choice in enumerate(output_choices)
        ],
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        },
    }


class CompletionRequest(BaseModel):
    model: str = Field(..., min_length=1)
    prompt: str | list[str] | list[int] | list[list[int]] = Field(..., min_length=1)
    best_of: Optional[int] = Field(default=None, ge=1, le=128)
    echo: bool = False
    frequency_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[int] = Field(default=None, ge=0, le=5)
    max_tokens: int = Field(default=16, ge=0)
    n: int = Field(default=1, ge=1, le=128)
    presence_penalty: Optional[float] = Field(default=None, ge=-2, le=2)
    seed: Optional[int] = None
    stop: Optional[str | list[str]] = None
    stream: bool = False
    suffix: Optional[str] = None
    temperature: Optional[float] = Field(default=1, ge=0, le=2)
    top_p: Optional[float] = Field(default=1, ge=0, le=1)
    user: Optional[str] = None


def text_completion_options(request: CompletionRequest) -> dict[str, Any]:
    options: dict[str, Any] = {
        "max_tokens": request.max_tokens,
        "temperature": request.temperature if request.temperature is not None else 1,
        "top_p": request.top_p if request.top_p is not None else 1,
        "n": request.n,
        "echo": request.echo,
    }
    for field in (
        "frequency_penalty", "presence_penalty", "logit_bias", "seed",
        "best_of", "logprobs", "suffix",
    ):
        value = getattr(request, field)
        if value is not None:
            options[field] = value
    if request.stop is not None:
        options["stop"] = request.stop
    return options


def completion_stream(
    request: CompletionRequest,
    prompts: list[str | list[int]],
    completion_id: str,
    created: int,
) -> Iterator[str]:
    with generation_lock:
        if model is None:
            raise RuntimeError("Qwen GGUF model is unavailable.")
        for prompt_index, prompt in enumerate(prompts):
            chunks = model.create_completion(
                prompt=prompt,
                **text_completion_options(request),
                stream=True,
            )
            for chunk in chunks:
                for choice in chunk["choices"]:
                    payload = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": OPENAI_MODEL_ID,
                        "choices": [{
                            "text": choice.get("text", ""),
                            "index": prompt_index * request.n + choice.get("index", 0),
                            "logprobs": choice.get("logprobs"),
                            "finish_reason": choice.get("finish_reason"),
                        }],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def openai_completions(request: CompletionRequest):
    if request.model != OPENAI_MODEL_ID:
        openai_model_not_found(request.model)
    raw_prompt = request.prompt
    if isinstance(raw_prompt, str):
        prompts: list[str | list[int]] = [raw_prompt]
    elif raw_prompt and isinstance(raw_prompt[0], str):
        prompts = raw_prompt
    elif raw_prompt and isinstance(raw_prompt[0], int):
        prompts = [raw_prompt]
    elif raw_prompt and isinstance(raw_prompt[0], list):
        prompts = raw_prompt
    else:
        raise HTTPException(status_code=400, detail="prompt must contain at least one prompt.")
    if request.best_of is not None and request.best_of < request.n:
        raise HTTPException(status_code=400, detail="best_of must be greater than or equal to n.")
    if request.stream and request.best_of not in (None, 1):
        raise HTTPException(status_code=400, detail="Streaming does not support best_of greater than 1.")
    completion_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Qwen GGUF model is unavailable; reload the model and try again.",
        )
    if request.stream:
        if request.best_of not in (None, 1):
            raise HTTPException(status_code=400, detail="Streaming does not support best_of greater than 1.")
        return StreamingResponse(
            completion_stream(request, prompts, completion_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    choices = []
    prompt_tokens = 0
    completion_tokens = 0
    output_count = 0
    with generation_lock:
        for prompt_index, prompt in enumerate(prompts):
            result = model.create_completion(
                prompt=prompt,
                **text_completion_options(request),
            )
            prompt_tokens += (result.get("usage") or {}).get("prompt_tokens", 0)
            completion_tokens += (result.get("usage") or {}).get("completion_tokens", 0)
            for choice_index, choice in enumerate(result.get("choices", [])):
                output_count += 1
                text = choice.get("text", "")
                choices.append({
                    "text": text,
                    "index": prompt_index * request.n + choice_index,
                    "logprobs": choice.get("logprobs"),
                    "finish_reason": choice.get("finish_reason") or "stop",
                })
                log_interaction(str(prompt), None, None, text)
    if not output_count:
        raise HTTPException(status_code=502, detail="The model returned no completion choices.")
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": OPENAI_MODEL_ID,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

# -----------------------------
# Endpoint: Text Generation
# -----------------------------
@app.post("/generate")
async def generate(user_text: str = Body(...), session_id: str = Body("default")):
    session_messages = conversations.get(session_id)

    memory_items = memory.retrieve(user_text, top_k=5)
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
    threading.Thread(
        target=extract_memories_background,
        args=(user_text, output),
        daemon=True,
    ).start()
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

    memory_items = memory.retrieve(image_description, top_k=5) if image_description else []
    build_messages(
        session_messages=session_messages,
        user_text=None,
        image_description=image_description,
        voice_text=None,
        memory_items=memory_items
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

    memory_items = memory.retrieve(voice_text, top_k=5) if voice_text else []
    build_messages(
        session_messages=session_messages,
        user_text=None,
        image_description=None,
        voice_text=voice_text,
        memory_items=memory_items
    )

    output = generate_from_messages(session_messages)
    session_messages.append({"role": "assistant", "content": output})

    log_interaction(None, None, voice_text, output)
    threading.Thread(
        target=extract_memories_background,
        args=(voice_text, output),
        daemon=True,
    ).start()
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
    importance: Optional[float] = Field(default=5.0, ge=1.0, le=10.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

class MemoryAddResponse(BaseModel):
    id: str
    status: str = "created"

class MemoryUpdateRequest(BaseModel):
    text: Optional[str] = Field(default=None, min_length=1, max_length=10_000)
    importance: Optional[float] = Field(default=None, ge=1.0, le=10.0)
    metadata: Optional[dict[str, Any]] = None

class MemorySearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)
    min_importance: Optional[float] = Field(default=1.0, ge=1.0, le=10.0)

class MemorySearchItem(BaseModel):
    id: Optional[str] = None
    text: str
    importance: Optional[float] = None
    score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    distance: Optional[float] = None

class MemoryListResponse(BaseModel):
    items: list[dict[str, Any]]
    limit: int
    offset: int
    total: int

class MemoryExtractRequest(BaseModel):
    text: Optional[str] = Field(default=None, max_length=10_000)
    user_text: Optional[str] = Field(default=None, max_length=10_000)
    assistant_text: Optional[str] = Field(default=None, max_length=10_000)
    auto_store: bool = True

class ExtractedMemoryItem(BaseModel):
    id: Optional[str] = None
    text: str
    importance: float

class MemoryExtractResponse(BaseModel):
    memories: list[ExtractedMemoryItem]
    count: int

@app.post("/memory/memory_add", response_model=MemoryAddResponse)
async def memory_add(req: MemoryAddRequest):
    memory_id = memory.add(req.text, metadata=req.metadata, importance=req.importance)
    return MemoryAddResponse(id=memory_id, status="created")

@app.post("/memory/memory_search", response_model=list[MemorySearchItem])
async def memory_search(req: MemorySearchRequest):
    results = memory.search(req.query, top_k=req.top_k, min_importance=req.min_importance or 1.0)
    return [MemorySearchItem(**r) for r in results]

@app.get("/memory/memory_list", response_model=MemoryListResponse)
async def memory_list(limit: int = 50, offset: int = 0):
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    items = memory.list_memories(limit=limit, offset=offset)
    total = memory.count()
    return MemoryListResponse(items=items, limit=limit, offset=offset, total=total)

@app.get("/memory/{memory_id}")
async def memory_get(memory_id: str):
    item = memory.get(memory_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    return item

@app.put("/memory/{memory_id}")
async def memory_update(memory_id: str, req: MemoryUpdateRequest):
    existing = memory.get(memory_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
    success = memory.update(
        memory_id=memory_id,
        text=req.text,
        metadata=req.metadata,
        importance=req.importance,
    )
    return {"status": "updated" if success else "failed", "id": memory_id}

@app.delete("/memory/{memory_id}")
async def memory_delete(memory_id: str):
    existing = memory.get(memory_id)
    if existing is None:
        return {"status": "not_found", "id": memory_id}
    memory.delete(memory_id)
    return {"status": "deleted", "id": memory_id}

@app.post("/memory/memory_extract", response_model=MemoryExtractResponse)
async def memory_extract(req: MemoryExtractRequest):
    target_user_text = req.user_text or req.text or ""
    if not target_user_text.strip():
        return MemoryExtractResponse(memories=[], count=0)
    if req.auto_store:
        stored = memory_extractor.extract_and_store(
            memory_store=memory,
            user_text=target_user_text,
            assistant_text=req.assistant_text,
            source="manual_extraction",
        )
        items = [ExtractedMemoryItem(**s) for s in stored]
    else:
        extracted = memory_extractor.extract(
            user_text=target_user_text,
            assistant_text=req.assistant_text,
        )
        items = [ExtractedMemoryItem(text=e["text"], importance=e["importance"]) for e in extracted]
    return MemoryExtractResponse(memories=items, count=len(items))

@app.get("/memory/stats")
async def memory_stats():
    total = memory.count()
    items = memory.list_memories(limit=200)
    avg_imp = 0.0
    if items:
        avg_imp = sum(m.get("importance", 5.0) for m in items) / len(items)
    return {
        "total_memories": total,
        "sample_size": len(items),
        "average_importance": round(avg_imp, 2),
    }

# -----------------------------
# Open WebUI Compatibility Endpoints
# -----------------------------
@app.get("/api/v1/memories")
async def openwebui_list_memories(limit: int = 50, offset: int = 0):
    return await memory_list(limit=limit, offset=offset)

@app.post("/api/v1/memories")
async def openwebui_add_memory(req: MemoryAddRequest):
    return await memory_add(req)

@app.delete("/api/v1/memories/{memory_id}")
async def openwebui_delete_memory(memory_id: str):
    return await memory_delete(memory_id)

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
            memory_extractor.model = model
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