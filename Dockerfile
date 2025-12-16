FROM nvidia/cuda:12.6.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/root/FireRedTTS2

# ---- 1. System dependencies ----
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    espeak-ng \
    build-essential \
    wget \
    tree \
    python3.11 \
    python3.11-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python

# ---- 2. Python core deps (STRICT versions) ----
RUN pip install --upgrade pip setuptools wheel

RUN pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    soundfile \
    "numpy<2.0" \
    huggingface_hub \
    --extra-index-url https://download.pytorch.org/whl/cu126

# ---- 3. Clone FireRedTTS2 (OFFICIAL, unmodified) ----
RUN git clone https://github.com/FireRedTeam/FireRedTTS2.git /root/FireRedTTS2

# ---- 4. Install repo requirements (remove torch conflicts) ----
RUN cd /root/FireRedTTS2 && \
    sed -i '/^torch[>=<=]/d' requirements.txt && \
    sed -i '/^torchaudio/d' requirements.txt && \
    sed -i '/^torchvision/d' requirements.txt && \
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# ---- 5. Install FireRedTTS2 as package (no deps) ----
RUN cd /root/FireRedTTS2 && pip install --no-deps -e .

# ---- 6. Download pretrained model (build-time) ----
RUN git lfs install && \
    mkdir -p /root/FireRedTTS2/pretrained_models && \
    cd /root/FireRedTTS2 && \
    git clone https://huggingface.co/FireRedTeam/FireRedTTS2 pretrained_models/FireRedTTS2

# ---- 7. Copy FastAPI app ----
COPY app.py /root/app.py

WORKDIR /root

# ---- 8. Start FastAPI ----
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

