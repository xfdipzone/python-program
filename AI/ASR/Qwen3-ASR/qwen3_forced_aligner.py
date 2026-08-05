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

"""
很 0.24 - 0.40
高 0.40 - 0.56
兴 0.56 - 0.72
认 0.72 - 0.88
识 0.88 - 1.04
你 1.04 - 1.28
哦 1.28 - 1.60
我 2.08 - 2.08
刚 2.08 - 2.24
刚 2.24 - 2.40
去 2.40 - 2.56
买 2.56 - 2.72
了 2.72 - 2.80
杯 2.88 - 2.96
奶 2.96 - 3.20
茶 3.20 - 3.68
要 4.08 - 4.24
一 4.24 - 4.32
起 4.32 - 4.48
去 4.48 - 4.64
长 4.64 - 4.88
椅 4.88 - 4.96
上 4.96 - 5.20
坐 5.20 - 5.36
坐 5.36 - 5.52
吗 5.52 - 6.00
"""
