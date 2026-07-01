# coding=utf-8
import os
import numpy as np
import librosa
import ChatTTS
import soundfile as sf
from IPython.display import Audio, display, Markdown

"""
通过 Chat-TTS 筛选出 Top 20 好听的年轻女声

pip install chattts
pip install librosa
pip install soundfile
pip install torch
pip install numpy
pip install ipython
"""
# 音色测试文字内容
TEST_TEXT = [
    "你好呀，我是你的语音助手，今天过得怎么样呢，希望我的声音你会喜欢。",
]

# 随机测试 100 种音色
VOICE_COUNT = 100

# 音色文件保存目录
OUTPUT_DIR = "voice_library"

# 创建保存目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 生成音色并计算平均基频
results = []

# 初始化并加载模型
chat = ChatTTS.Chat()
chat.load(compile=False)

for idx in range(VOICE_COUNT):
    print(f"\n进行音色生成 {idx:03d}")

    # 随机音色
    spk = chat.sample_random_speaker()

    # 保存 Speaker Embedding
    spk_file = os.path.join(
        OUTPUT_DIR,
        f"speaker_{idx:03d}.txt"
    )

    with open(spk_file, "w", encoding="utf-8") as f:
        f.write(spk)

    # 生成音频
    wavs = chat.infer(
        TEST_TEXT,
        params_infer_code=ChatTTS.Chat.InferCodeParams(
            spk_emb=spk,
            temperature=0.3,
            top_P=0.7,
            top_K=20
        )
    )

    audio = np.concatenate(wavs)

    if hasattr(audio, "shape") and len(audio.shape) > 1:
        audio = audio[0]

    # 保存试听音频
    wav_file = os.path.join(
        OUTPUT_DIR,
        f"speaker_{idx:03d}.wav"
    )

    sf.write(
        wav_file,
        audio,
        24000
    )

    # 提取基频
    try:
        # 限定了算法寻找人类说话音高的范围（ 60Hz 到 500Hz ）
        # 男性的基频通常在 85 ~ 180Hz
        # 女性通常在 165 ~ 255Hz，儿童或极高亢的声音会更高
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio.astype(np.float32),
            sr=24000,
            fmin=60,
            fmax=500
        )

        # 清洗数据（剔除无声/静音部分）
        valid_f0 = f0[~np.isnan(f0)]

        if len(valid_f0) > 0:
            mean_pitch = np.mean(valid_f0)
        else:
            mean_pitch = 0

    except Exception:
        mean_pitch = 0

    results.append(
        {
            "id": idx,
            "pitch": mean_pitch,
            "wav": wav_file,
            "spk": spk_file
        }
    )

# 按照音高排序（女性音高 > 男性）
results = sorted(
    results,
    key=lambda x: x["pitch"],
    reverse=True
)

top_candidates = results[:20]

print("Top 20 Female-like Speakers")
print("-" * 40)

for item in top_candidates:
    print(
        f"Speaker {item['id']:03d} "
        f"Pitch={item['pitch']:.1f}"
    )

# 试听 Top 20
for rank, item in enumerate(top_candidates):
    display(
        Markdown(
            f"### Rank {rank + 1} "
            f"Speaker {item['id']:03d} "
            f"(Pitch={item['pitch']:.1f})"
        )
    )

    display(
        Audio(
            item["wav"]
        )
    )
