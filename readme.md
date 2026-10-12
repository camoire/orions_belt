# 📘 **README.md — OrionsBelt Local AI Assistant**

## 🚀 Overview

OrionsBelt is a **local AI assistant** built with:

- **FastAPI** for serving responses  
- **Qwen2.5-Coder-14B-Instruct Q6_K GGUF** through llama.cpp  
- **A memory system** for contextual responses  
- **A behavior logger** for dataset generation  
- **Docker** for reproducible deployment  

This document outlines host requirements, model setup, and how to build and run the service.

---

# 🖥️ Host Machine Requirements

Your host machine must have the following installed:

### **1. Python 3.11+**
Required for:
- Running FastAPI locally  
- Downloading the model  
- Generating datasets  

Check version:
```
python --version
```

---

### **2. Pip + Virtual Environment (recommended)**

Install dependencies:
```
pip install -r requirements.txt
```

---

### **3. Hugging Face Token (recommended)**

A token dramatically speeds up model downloads and avoids throttling.

Set it in your environment:

**PowerShell (Windows):**
```
$env:HF_TOKEN = "hf_xxx"
```

**Bash / WSL / Linux / macOS:**
```
export HF_TOKEN="hf_xxx"
```

---

### **4. Disk Space and Memory Requirements**

The Q6_K GGUF model is about 12 GB. Allow additional space for BLIP, dependencies, and Docker layers. Inference also needs enough system RAM for the model and context; GPU offloading requires a CUDA-enabled llama-cpp-python build.

**Recommended free disk space: 40 GB**

---

### **5. Docker Desktop**

Required for:
- Building the production image  
- Running the FastAPI service in a container  

---

### **6. Git (optional)**

For version control and cloning the repo.

---

# 📁 Project Structure

```
OrionsBelt/
│
├── app.py                 # FastAPI app (generate, train-now, reload-lora)
├── train_lora.py          # LoRA training script
├── build_dataset.py       # Converts logs → training dataset
├── download_model.py      # Downloads Qwen GGUF + BLIP models
│
├── memory.py              # Memory store
├── behavior_logger.py     # Structured behavior logging
│
├── qwen_model/            # Local Qwen GGUF model (downloaded once)
├── blip_model/            # Local BLIP model (downloaded once)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 📥 Download the Models Once (Required Before Starting the API)

Run:

```
python download_model.py
```

This script downloads both models into the project:

- `Qwen2.5-Coder-14B-Instruct-Q6_K.gguf` from `bartowski/Qwen2.5-Coder-14B-Instruct-GGUF` into `qwen_model/`
- BLIP image captioning model into `blip_model/`

The Qwen inference path runs locally through llama.cpp. Set `QWEN_GGUF_PATH` to use a different GGUF file. CPU inference is the default; set `LLAMA_N_GPU_LAYERS` to configure GPU offloading when using a llama-cpp-python build with GPU support. `LLAMA_N_CTX` controls the context window and defaults to 8192 tokens.

---

# 🧪 Generate Training Data

Start the API locally so it can log interactions:

```
uvicorn app:app --host 0.0.0.0 --port 8000
```

Send some requests to `/generate`, then stop the server. The interactions are stored in `logs/interactions.jsonl`.

Generate dataset:

```
python build_dataset.py
```

The legacy Transformers trainer is separate from GGUF inference. Its PEFT adapters cannot be loaded by the GGUF server as-is; `/reload-lora` only accepts a llama.cpp-compatible GGUF adapter configured with `LLAMA_LORA_PATH`.

---

# 🔄 Reload LoRA Adapter Without Restarting

Once the container is running:

```
POST http://localhost:8000/reload-lora
```

This reloads the GGUF model, including a configured llama.cpp-compatible adapter if present.

---

# 🛠️ Build the Docker Image

After downloading the models:

```
docker build -t orions_belt .
```

The model files are mounted at runtime rather than copied into the image:

- No model downloads happen inside Docker
- The image does not contain the large GGUF or BLIP model files

---

# 🚀 Run the Container

```
docker run -p 8000:8000 \
  -v "$(pwd)/qwen_model:/app/qwen_model:ro" \
  -v "$(pwd)/blip_model:/app/blip_model:ro" \
  orions_belt
```

Open Swagger UI:

```
http://localhost:8000/docs
```

---

# 🔧 API Endpoints

### **POST /generate**
Generate a response using the model + memory.

### **POST /train-now**
Triggers the legacy Transformers LoRA training workflow. Its output cannot be loaded directly by GGUF inference.

### **POST /reload-lora**
Reloads the GGUF model and an optional llama.cpp-compatible adapter.

---

# 🎯 Summary

Your host machine must have:

- Python 3.11  
- Pip + venv  
- Hugging Face token (recommended for model downloads)  
- Docker Desktop  
- Git (optional)  
- ~40 GB free disk space  

Your workflow:

1. `python download_model.py`
2. `docker build -t orions_belt .`
3. `docker run -p 8000:8000 orions_belt`
4. Open `http://localhost:8000/docs`
