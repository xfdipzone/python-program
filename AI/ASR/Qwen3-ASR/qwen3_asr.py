# coding=utf-8
import torch
import os
from pathlib import Path
from qwen_asr import Qwen3ASRModel

"""
自动识别语音为文本（基于 Qwen3-ASR）

模型：
https://huggingface.co/Qwen/Qwen3-ASR-1.7B
高精度大参数版本，在复杂环境和难度文本下拥有极高的准确率，达到业界领先水平

https://huggingface.co/Qwen/Qwen3-ASR-0.6B
轻量化版本，在精度与推理效率之间取得极佳平衡，适合高并发部署

dependency packages
pip install -U qwen-asr
pip install accelerate
pip install transformers
pip install soundfile
pip install librosa
"""
# 加载模型
model = Qwen3ASRModel.from_pretrained(
    "Qwen/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# 音频文件目录
audio_dir = "data/audio"

# 支持的音频格式
audio_extensions = {
    ".wav",
    ".mp3",
}

# 音频文件
audio_files = [
    str(f)
    for f in Path(audio_dir).iterdir()
    if f.suffix.lower() in audio_extensions
]

# 文本输出目录
output_dir = "data/output"

# 创建文本输出目录
os.makedirs(output_dir, exist_ok=True)

# 每批次音频的数量（按机器配置能力调整）
batch_size = 2

# 批量识别音频转为文本
for i in range(0, len(audio_files), batch_size):
    batch = audio_files[i:i + batch_size]

    results = model.transcribe(
        audio=batch,
        language="Chinese"
    )

    # 将音频自动识别的文本保存
    for audio_file, r in zip(batch, results):
        file_path = os.path.join(output_dir, f"{Path(audio_file).stem}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(r.text)

        print(f"语音文件: {audio_file}")
        print(f"识别的文本文件：{file_path}")
        print(f"识别的文本内容：{r.text}\n")
