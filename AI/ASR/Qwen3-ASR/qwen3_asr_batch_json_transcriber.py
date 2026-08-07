# coding=utf-8
import torch
import json
import os
from pathlib import Path
from qwen_asr import Qwen3ASRModel

"""
批量识别语音，将识别的内容更新到 JSON 文件（基于 Qwen3-ASR）

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

# JSON 源文件
json_file = "data/source.json"

# 音频目录
audio_dir = Path("data/audio")

# JSON 输出文件
output_json = "output/asr_result.json"

# 每批次处理的音频数量，根据 GPU 显存调整
# T4建议 2~4
# A10/A100 可以更大
batch_size = 2

# 创建输出目录
os.makedirs(
    Path(output_json).parent,
    exist_ok=True
)


# 读取 JSON 源文件
with open(json_file, "r", encoding="utf-8") as f:
    voices = json.load(f)


# 创建识别任务
tasks = []

# 遍历 JSON 每一个 item
for item in voices:
    # 已经识别过，直接跳过
    if item.get("text", "").strip():
        print(f"跳过已识别: {item['file']}")
        continue

    # 查找 JSON 内音频文件
    audio_file = audio_dir / item["file"]

    if not audio_file.exists():
        print(f"⚠️ 音频不存在: {audio_file}")
        item["text"] = ""
        continue

    tasks.append(
        (
            item,
            str(audio_file)
        )
    )


# 没有任务直接保存退出
if not tasks:
    print("没有需要识别的音频")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)

    exit()


# 批量识别
for i in range(0, len(tasks), batch_size):
    batch_tasks = tasks[
        i:i + batch_size
    ]

    audio_files = [
        x[1]
        for x in batch_tasks
    ]

    print("\n" + "=" * 60)
    print(f"正在识别 {i + 1}-{i + len(batch_tasks)} / {len(tasks)}")

    results = model.transcribe(
        audio=audio_files,
        language="Chinese"
    )

    # 更新 JSON对象
    for (item, audio_file), result in zip(batch_tasks, results):
        text = result.text.strip()
        item["text"] = text
        print(f"\n文件: {item['file']}")
        print(f"文本: {text}")


# 保存结果
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(voices, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("全部完成")
print(f"输出文件: {output_json}")


# 输出修改后的 JSON
print("\n" + "=" * 60)
print("修改后的 JSON 内容:")
print("=" * 60)

with open(output_json, "r", encoding="utf-8") as f:
    updated_json = json.load(f)

print(json.dumps(updated_json, ensure_ascii=False, indent=2))

"""
修改后的 JSON 内容:
[
  {
    "name": "Soft Girl",
    "name_cn": "温柔女孩",
    "description": "Adult, Female, Welcoming · Ideal for Vlog, Podcast",
    "description_cn": "成年女性，亲切温暖 · 适用于Vlog、播客",
    "file": "soft_girl.mp3",
    "text": "很高兴认识你哦！我刚刚去买了杯奶茶，要一起去长椅上坐坐吗？"
  },
  {
    "name": "Warm Bestie",
    "name_cn": "暖心闺蜜",
    "description": "Adult, Female, Crisp · Ideal for News, Explainer",
    "description_cn": "成年女性，声音清晰利落 · 适用于新闻、知识讲解",
    "file": "warm_bestie.mp3",
    "text": "看你这样子，估计明天会有点头疼。不过没关系，多喝点水，明天会好得快一点。"
  },
  {
    "name": "Warm Girl",
    "name_cn": "温暖女孩",
    "description": "Female, Soft · Ideal for Meditation, Storytelling",
    "description_cn": "女性，柔和温暖 · 适用于冥想、故事讲述",
    "file": "warm_girl.mp3",
    "text": "或许今天的你并不完美，或许你还在寻找那个属于自己的方向，但不要急，给自己一点时间吧。"
  }
]
"""
