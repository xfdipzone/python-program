# coding=utf-8
import torch
from qwen_asr import Qwen3ForcedAligner

"""
将语音与文本强制对齐，返回词级/字级时间戳（基于 Qwen3-ForcedAligner）

模型：
https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
找到每个词/字在音频中的开始与结束时间

dependency packages
pip install -U qwen-asr
pip install accelerate
pip install transformers
pip install soundfile
pip install librosa
"""
# 加载模型
model = Qwen3ForcedAligner.from_pretrained(
    "Qwen/Qwen3-ForcedAligner-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# 音频文件
audio = "data/audio/soft_girl.mp3"

results = model.align(
    audio=audio,
    text="很高兴认识你哦！我刚刚去买了杯奶茶，要一起去长椅上坐坐吗？",
    language="Chinese",
)

for item in results[0]:
    print(f"{item.text} {item.start_time:.2f} - {item.end_time:.2f}")
