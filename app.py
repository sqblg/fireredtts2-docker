from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

app = FastAPI(title="FireRedTTS2 API")

# ---- Model Load (once) ----
repo_path = "/root/FireRedTTS2"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from fireredtts2.fireredtts2 import FireRedTTS2

MODEL_DIR = os.path.join(repo_path, "pretrained_models/FireRedTTS2")

tts = FireRedTTS2(
    pretrained_dir=MODEL_DIR,
    gen_type="dialogue",
    device="cuda"
)

# ---- Request schema ----
class TTSRequest(BaseModel):
    text_list: list[str]
    prompt_wav_list: list[str]
    prompt_text_list: list[str]
    temperature: float = 0.75
    topk: int = 20

# ---- API ----
@app.post("/generate")
def generate(req: TTSRequest):
    rec_wavs = tts.generate_dialogue(
        text_list=req.text_list,
        prompt_wav_list=req.prompt_wav_list,
        prompt_text_list=req.prompt_text_list,
        temperature=req.temperature,
        topk=req.topk
    )

    return {
        "sample_rate": 24000,
        "wav_tensor_shape": list(rec_wavs.shape)
    }

# ---- Entrypoint ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
