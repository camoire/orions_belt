# 📘 **README.md — OrionsBelt Local AI Assistant**

## 🚀 Overview

OrionsBelt is a **self‑improving local AI assistant** built with:

- **FastAPI** for serving responses  
- **Qwen2.5‑7B‑Instruct** as the base model  
- **LoRA adapters** for continual fine‑tuning  
- **A memory system** for contextual responses  
- **A behavior logger** for dataset generation  
- **Docker** for reproducible deployment  

This document outlines all **host machine requirements**, setup steps, and the full workflow for downloading the model, training LoRA adapters, and building the Docker image.

---

# 🖥️ Host Machine Requirements

Your host machine must have the following installed:

### **1. Python 3.11+**
Required for:
- Running FastAPI locally  
- Training LoRA adapters  
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

### **4. Disk Space Requirements**

Qwen2.5‑7B‑Instruct requires:

- 14–16 GB model weights  
- 3–5 GB tokenizer + metadata  
- 5–10 GB for LoRA training artifacts  
- 5–10 GB for Docker image layers  

**Recommended free space: 40 GB**

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
├── download_model.py      # Downloads model + copies into qwen_model/
│
├── memory.py              # Memory store
├── behavior_logger.py     # Structured behavior logging
│
├── lora_adapter/          # LoRA weights (after training)
├── qwen_model/            # Local copy of Qwen model (downloaded once)
│
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 📥 Download the Model Once (Required Before Docker Build)

Run:

```
python download_model.py
```

This script:

- Downloads Qwen2.5‑7B‑Instruct into your Hugging Face cache  
- Copies it into `qwen_model/`  
- Prepares it for Docker  

---

# 🧪 Training a LoRA Adapter

Generate dataset:

```
python build_dataset.py
```

Train:

```
python train_lora.py
```

Your trained adapter will appear in:

```
lora_adapter/
```

---

# 🔄 Reload LoRA Adapter Without Restarting

Once the container is running:

```
POST http://localhost:8000/reload-lora
```

This hot‑swaps the adapter instantly.

---

# 🛠️ Build the Docker Image

After the model + LoRA adapter are in place:

```
docker build -t orions_belt .
```

This build is fast because:

- No model downloads happen inside Docker  
- Everything is copied from your host  

---

# 🚀 Run the Container

```
docker run -p 8000:8000 orions_belt
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
Triggers LoRA training in the background.

### **POST /reload-lora**
Reloads the LoRA adapter without restarting the container.

---

# 🎯 Summary

Your host machine must have:

- Python 3.11  
- Pip + venv  
- Hugging Face token (recommended)  
- Docker Desktop  
- Git (optional)  
- ~40 GB free disk space  

Your workflow:

1. `python download_model.py`  
2. `python build_dataset.py`  
3. `python train_lora.py`  
4. `docker build -t orions_belt .`  
5. `docker run -p 8000:8000 orions_belt`  
6. Use `/reload-lora` to hot‑swap adapters  

