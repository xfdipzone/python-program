# coding=utf-8
import os
import ChatTTS
import soundfile as sf
from IPython.display import Audio, display, Markdown

"""
Chat-TTS 音色库管理

dependency packages
pip install chattts
pip install soundfile
pip install ipython
"""
# 随机测试 50 种音色
VOICE_COUNT = 50

# 音色测试文字内容
TEST_TEXT = "你好呀，很高兴认识你。这是一个音色测试。今天天气不错，希望你能找到喜欢的声音。"

# 音色文件保存目录
OUTPUT_DIR = "voice_library"

# 创建保存目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化并加载模型
chat = ChatTTS.Chat()
chat.load(compile=False)

for idx in range(VOICE_COUNT):
    print(f"\n进行音色生成 {idx:02d}")

    # 随机音色
    spk = chat.sample_random_speaker()

    # 保存 Speaker Embedding
    speaker_file = os.path.join(
        OUTPUT_DIR,
        f"speaker_{idx:02d}.txt"
    )

    with open(speaker_file, "w", encoding="utf-8") as f:
        f.write(spk)

    # 生成试听音频
    wavs = chat.infer(
        [TEST_TEXT],
        params_infer_code=ChatTTS.Chat.InferCodeParams(
            spk_emb=spk,
            temperature=0.3,
            top_P=0.7,
            top_K=20
        )
    )

    audio = wavs[0]

    if hasattr(audio, "shape") and len(audio.shape) > 1:
        audio = audio[0]

    # 保存试听音频
    wav_file = os.path.join(
        OUTPUT_DIR,
        f"speaker_{idx:02d}.wav"
    )

    sf.write(
        wav_file,
        audio,
        24000
    )

    display(Markdown(f"### Speaker {idx:02d}"))
    display(Audio(audio, rate=24000))

print("\n音色库生成完成")
