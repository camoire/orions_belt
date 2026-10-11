# 📘 **Developer Onboarding Guide**

Welcome to **OrionsBelt**, a self‑improving local AI assistant powered by:

- Qwen2.5‑7B‑Instruct  
- FastAPI  
- LoRA continual fine‑tuning  
- A memory + behavior logging system  
- Docker for reproducible deployment  

This guide walks you through everything you need to get the system running locally, train LoRA adapters, and build the production container.

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

# 📥 4. Download the Qwen Model (One‑Time Setup)

Run the helper script:

```
python download_model.py
```

This script downloads both required models into the project:

- Qwen2.5‑7B‑Instruct into `qwen_model/`
- BLIP image captioning model into `blip_model/`

The files are placed where the application and Docker image expect them. The script downloads model files without loading the full Qwen model into memory.

After this step, your project will contain:

```
qwen_model/
    models--Qwen--Qwen2.5-7B-Instruct/
        snapshots/<hash>/
            model.safetensors
            tokenizer.json
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

---

# 🧠 6. Train a LoRA Adapter

Run:

```
python train_lora.py
```

This creates:

```
lora_adapter/
    adapter_config.json
    adapter_model.safetensors
    ...
```

---

# 🔄 7. Hot‑Reload the LoRA Adapter (No Restart Needed)

Once the FastAPI service is running, call:

```
POST http://localhost:8000/reload-lora
```

This reloads the adapter instantly.

---

# 🛠️ 8. Build the Docker Image

After downloading the models, and optionally training a LoRA adapter:

```
docker build -t orions_belt .
```

This build is fast because:

- No model downloads occur inside Docker  
- Everything is copied from your host  

---

# 🚀 9. Run the Container

```
docker run -p 8000:8000 orions_belt
```

Open the API docs:

```
http://localhost:8000/docs
```

(You already have this open in your browser .)

---

# 🔧 10. Key API Endpoints

### **POST /generate**
Generate a response using the model + memory.

### **POST /train-now**
Triggers LoRA training in the background.

### **POST /reload-lora**
Reloads the LoRA adapter without restarting the container.

---

# 🧭 11. Typical Developer Workflow

1. Pull latest code  
2. Run `download_model.py` (only once per machine)  
3. Start FastAPI locally and generate interaction logs  
4. Run `build_dataset.py`  
5. Run `train_lora.py` (optional)  
6. Build and run the Docker image  
7. Call `/reload-lora` to load an updated adapter  

---

# 🎯 12. Troubleshooting

### **Model download is slow**
Set your HF token:
```
$env:HF_TOKEN = "hf_xxx"
```

### **Docker build takes too long**
Ensure you ran `download_model.py` first.

### **LoRA not loading**
Check that `lora_adapter/` exists and call `/reload-lora`.

### **FastAPI not responding**
Ensure port 8000 isn’t in use.