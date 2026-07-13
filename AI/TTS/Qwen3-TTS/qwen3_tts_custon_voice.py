# coding=utf-8
import soundfile as sf
import os
from qwen_tts import Qwen3TTSModel
from IPython.display import Audio, display, Markdown

"""
文本转为语音（基于 Qwen3-TTS）
使用指定的 Speaker

dependency packages
pip install qwen-tts
pip install soundfile
pip install ipython
"""
# 加载模型
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda"
)

# 查看 Speaker
print(tts.get_supported_speakers())

# 使用的 Speaker
speaker = "vivian"

# 文本内容
text = "王总您好，提醒您一下，今天下午两点在三号会议室有一个关于新项目的推进会，各部门经理都会出席。会议资料我已经打印好放在您的办公桌左侧了，另外，五点钟您和李总还有一个视频会议。"

audio = tts.generate_custom_voice(
    text=text,
    language="chinese",
    speaker=speaker,
    instruct="声音甜美，亲切自然，像电台主播。"
)

waveform = audio[0][0]
sample_rate = audio[1]

# 保存语音的目录
output_dir = "output_audio"

# 检查保存的目录，如不存在则创建目录
os.makedirs(output_dir, exist_ok=True)

# 语音文件
file_path = os.path.join(output_dir, "custom_voice.wav")

# 保存语音
sf.write(
    file_path,
    waveform,
    sample_rate
)

display(Markdown(f"### 🎙️ **Speaker: {speaker}**"))
display(Markdown(f"语音文件已自动保存到： `{file_path}`"))
display(Audio(waveform, autoplay=False, rate=sample_rate))
