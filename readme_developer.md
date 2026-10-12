# 📘 **Developer Onboarding Guide**

Welcome to **OrionsBelt**, a local AI assistant powered by:

- Qwen2.5-Coder-14B-Instruct Q6_K GGUF via llama.cpp
- FastAPI  
- A memory + behavior logging system  
- Docker for reproducible deployment  

This guide walks you through getting the system running locally and building the production container.

---

# 🧩 1. Prerequisites

Install the following on your host machine:

### **Required**
- **Python 3.11+**
- **Pip** (bundled with Python)
- **Docker Desktop**
- **~40 GB free disk space**

### **Recommended**
- **Hugging Face token** (dramatically speeds up model downloads)
- **Git** (for version control)

---

# 🔑 2. Configure Your Hugging Face Token

Set your token in your environment:

### PowerShell (Windows)
```
$env:HF_TOKEN = "hf_xxx"
```

### Bash / WSL / Linux / macOS
```
export HF_TOKEN="hf_xxx"
```

---

# 📦 3. Install Python Dependencies

From the project root:

```
pip install -r requirements.txt
```

---

# 📥 4. Download the Models (One‑Time Setup)

Run the helper script:

```
python download_model.py
```

This script downloads both required models into the project:

- `Qwen2.5-Coder-14B-Instruct-Q6_K.gguf` from `bartowski/Qwen2.5-Coder-14B-Instruct-GGUF` into `qwen_model/`
- BLIP image captioning model into `blip_model/`

The Qwen GGUF is loaded locally by llama.cpp. CPU inference is the default. Set `QWEN_GGUF_PATH` to select another GGUF file or `LLAMA_N_GPU_LAYERS` when using a GPU-enabled llama-cpp-python build.

After this step, your project will contain:

```
qwen_model/
    Qwen2.5-Coder-14B-Instruct-Q6_K.gguf
blip_model/
    config.json
    ...
```

---

# 🧪 5. Generate Training Data

Start the API locally with `uvicorn app:app --host 0.0.0.0 --port 8000` and send requests to `/generate`; the system logs interactions automatically. Stop the API after generating at least two interactions.
To convert logs into a training dataset:

```
python build_dataset.py
```

This produces:

```
dataset/train.jsonl
dataset/val.jsonl
```

The app still exposes `/train-now`, `/training-status`, and `/train-and-reload` for the legacy Transformers trainer. It trains a separate Hugging Face Qwen checkpoint; those PEFT adapters are not compatible with GGUF inference and are not loaded by the GGUF server.

---

# 🔄 6. Reload the GGUF Model

Once the FastAPI service is running, call:

```
POST http://localhost:8000/reload-lora
```

This reloads the GGUF model. To apply a LoRA adapter, set `LLAMA_LORA_PATH` to a llama.cpp-compatible GGUF adapter. Transformers PEFT adapter files from the legacy trainer are not compatible directly.

---

# 🛠️ 7. Build the Docker Image

After downloading the models:

```
docker build -t orions_belt .
```

The model files are mounted at runtime rather than copied into the image:

- No model downloads occur inside Docker
- The image does not contain the large GGUF or BLIP model files

---

# 🚀 8. Run the Container

```
docker run -p 8000:8000 \
  -v "$(pwd)/qwen_model:/app/qwen_model:ro" \
  -v "$(pwd)/blip_model:/app/blip_model:ro" \
  orions_belt
```

Open the API docs:

```
http://localhost:8000/docs
```

(You already have this open in your browser .)

---

# 🔧 9. Key API Endpoints

### **POST /generate**
Generate a response using the model + memory.

### **POST /train-now** and **GET /training-status**
Run and check the legacy Transformers training workflow. Its adapters are not used for GGUF inference.

### **POST /train-and-reload**
Runs legacy Transformers training; the resulting adapter is not loaded into the GGUF model.

### **POST /reload-lora**
Reloads the GGUF model and any configured llama.cpp-compatible adapter.

---

# 🧭 10. Typical Developer Workflow

1. Pull latest code
2. Run `download_model.py` (only once per machine)
3. Build the Docker image
4. Run the container and use the API

---

# 🎯 11. Troubleshooting

### **Model download is slow**
Set your HF token:
```
$env:HF_TOKEN = "hf_xxx"
```

### **Docker build takes too long**
Ensure you ran `download_model.py` first.

### **LoRA not loading**
`/reload-lora` accepts a llama.cpp-compatible GGUF adapter via `LLAMA_LORA_PATH`; Transformers PEFT adapter files are not compatible directly.

### **FastAPI not responding**
Ensure port 8000 isn’t in use.