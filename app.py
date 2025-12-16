import os
import sys
import io
import glob
from typing import List, Optional

import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="FireRedTTS2 Prod API")

# ---- 模型加载（全局，只初始化一次）----
repo_path = "/root/FireRedTTS2"
if repo_path not in sys.path:
    sys.path.append(repo_path)

from fireredtts2.fireredtts2 import FireRedTTS2  # 官方包，不改

MODEL_DIR = os.path.join(repo_path, "pretrained_models/FireRedTTS2")

try:
    tts = FireRedTTS2(
        pretrained_dir=MODEL_DIR,
        gen_type="dialogue",  # 和你 Modal 脚本一致
        device="cuda"
    )
except Exception as e:
    # 这里抛出 500，方便你在后端看到错误
    raise RuntimeError(f"FireRedTTS2 初始化失败: {e}")


# ---- 请求体：脚本文本 + 可选参数 ----
class TTSRequest(BaseModel):
    # 你的后端已经拼好的播客脚本，每一条是类似 "[S1] ...", "[S2] ..." 的句子列表
    text_list: List[str]

    # 如果你想把对应的 prompt 文本也传进来（可选）
    prompt_text_list: Optional[List[str]] = None

    temperature: float = 0.75
    topk: int = 20


# ---- 核心接口：生成音频并把「音频字节」回传给你的后端 ----
@app.post("/generate")
async def generate(
    req: TTSRequest,
    speaker_wavs: Optional[List[UploadFile]] = File(
        None,
        description="可选：你后端上传的克隆声音文件列表，比如 S1.wav、S2.wav"
    ),
):
    """
    - 你的后端调用 /generate：
      - 用 multipart/form-data 传：
        - JSON 字段: text_list / prompt_text_list / temperature / topk
        - 文件字段: speaker_wavs[]=S1.wav, speaker_wavs[]=S2.wav ...
    - 本接口：
      1. 把上传的 wav 保存到 /tmp
      2. 调 FireRedTTS2.generate_dialogue
      3. 得到 torch.Tensor -> wav bytes
      4. 通过 StreamingResponse 把 wav 返回给你的后端
    - 你后端拿到 bytes 后：
      - 上传 Cloudflare R2
      - 更新 Supabase
    """
    try:
        # 1. 处理 prompt 声音
        prompt_wav_list: List[str] = []

        if speaker_wavs:
            # 使用网站后端传来的克隆声音
            for wav_file in speaker_wavs:
                contents = await wav_file.read()
                save_path = f"/tmp/{wav_file.filename}"
                with open(save_path, "wb") as f:
                    f.write(contents)
                prompt_wav_list.append(save_path)
        else:
            # 如果没有上传 prompt，就用仓库里自带的 FLAC 做 fallback
            flacs = glob.glob(os.path.join(repo_path, "**", "*.flac"), recursive=True)
            if len(flacs) < 2:
                raise HTTPException(status_code=500, detail="没有找到足够的 prompt 音频文件")
            # 简单取前 N 个，你也可以在后端控制 text_list 里 [S1]/[S2] 的数量
            prompt_wav_list = flacs[: len(set([line.split("[")[1] for line in req.text_list if "[" in line]))]

        # 2. 处理 prompt_text_list
        if req.prompt_text_list is not None:
            prompt_text_list = req.prompt_text_list
        else:
            # 如果你不想传，就给一个占位，模型只关心结构
            prompt_text_list = ["[S1] 示例", "[S2] 示例"]

        # 3. 调用 FireRedTTS2 生成
        rec_wavs = tts.generate_dialogue(
            text_list=req.text_list,
            prompt_wav_list=prompt_wav_list,
            prompt_text_list=prompt_text_list,
            temperature=req.temperature,
            topk=req.topk,
        )
