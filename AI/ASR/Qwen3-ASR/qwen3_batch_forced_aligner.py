# coding=utf-8
import torch
import json
import os
from pathlib import Path
from qwen_asr import Qwen3ForcedAligner

"""
批量将语音与文本强制对齐，返回词级/字级时间戳（基于 Qwen3-ForcedAligner）

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

# 音频文件目录
audio_dir = "data/audio"

# 音频文本目录
text_dir = "data/text"

# 音频与文本对齐的输出目录
output_dir = "data/alignment"

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

# 创建音频与文本对齐的输出目录
os.makedirs(output_dir, exist_ok=True)

# 批量对齐音频与文本
for audio_file in audio_files:
    # 文件名，不包含后缀
    name = Path(audio_file).stem

    # 对应文本
    text_file = os.path.join(text_dir, f"{name}.txt")

    if not os.path.exists(text_file):
        print(f"{text_file} 不存在，跳过")
        continue

    # 读取文本内容
    with open(text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        print(f"{text_file} 文本内容为空，跳过")
        continue

    results = model.align(
        audio=audio_file,
        text=text,
        language="Chinese",
    )

    # 保存音频与文本对齐数据
    alignment = []

    for item in results[0]:
        data = {
            "text": item.text,
            "start": round(item.start_time, 3),
            "end": round(item.end_time, 3)
        }

        alignment.append(data)

    # 保存到 JSON 文件
    output_file = os.path.join(output_dir, f"{name}_alignment.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(alignment, f, ensure_ascii=False, indent=2)

    # 输出保存的 JSON 文件内容
    print("\n" + "=" * 60)
    print(f"{output_file} 内容:")
    print("=" * 60)

    with open(output_file, "r", encoding="utf-8") as f:
        json_content = json.load(f)

    print(json.dumps(json_content, ensure_ascii=False, indent=2))

"""
音频与文本强制对齐结果
data/alignment/soft_girl_alignment.json
data/alignment/warm_bestie_alignment.json
data/alignment/warm_girl_alignment.json
"""
