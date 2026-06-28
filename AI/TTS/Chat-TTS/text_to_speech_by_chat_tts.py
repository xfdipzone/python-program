# coding=utf-8
import ChatTTS
import soundfile as sf
import os
from IPython.display import Audio, display, Markdown

"""
文本转为语音（基于 Chat TTS）
使用指定的 Speaker

dependency packages
pip install chattts
pip install soundfile
pip install torch
pip install numpy
pip install ipython
"""
# 定义音色库目录
voice_library_dir = "data/chat-tts_voice_library"

# 音色库中喜爱的声音
favorite_voices = {
    "成熟女性": "speaker_01",
    "女主持": "speaker_11",
    "性感女性": "speaker_17",
    "女教师": "speaker_21",
    "女主播": "speaker_22",
    "温柔女性": "speaker_26",
    "活泼女性": "speaker_28",
    "女医生": "speaker_31",
    "年轻女性": "speaker_39",
}

voice_name = favorite_voices["活泼女性"]

# 初始化并加载模型
chat = ChatTTS.Chat()
chat.load(compile=False)

with open(
    f"{voice_library_dir}/{voice_name}.txt",
    "r",
    encoding="utf-8"
) as f:
    spk = f.read()

# 文本内容
text = "[speed_3] 亲爱的主人，您好呀 [uv_break] 我叫小莹 [uv_break] 是您的专属小助手，[uv_break] 此刻正带着满满的暖意，给您送来这份问候呢。"

wavs = chat.infer(
    [text],
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

# 保存语音的目录
output_dir = "output_audio"

# 语音文件
file_path = os.path.join(output_dir, "output.wav")

# 检查保存的目录，如不存在则创建目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

sf.write(
    file_path,
    audio,
    24000
)

display(Markdown("### 🔊 生成的语音"))
display(Markdown(f"**语音文件已自动保存到：** `{file_path}`"))
display(Audio(audio, autoplay=False, rate=24000))
