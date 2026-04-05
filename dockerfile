# ---------------------------------------------------------
# STAGE 1 — Builder: install Python deps only
# ---------------------------------------------------------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your full project (including LoRA adapter)
COPY . .

# ---------------------------------------------------------
# STAGE 2 — Runtime: copy model + LoRA + deps
# ---------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Copy installed Python libs + binaries
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy your application code
COPY . .

# Copy the model from your host into the image
# (You will place the model folder next to your Dockerfile)
COPY qwen_model/ /root/.cache/huggingface/hub/

# Copy LoRA adapter into the image
COPY lora_adapter/ /app/lora_adapter/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
