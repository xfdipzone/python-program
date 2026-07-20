# coding=utf-8
import soundfile as sf
import os
from qwen_tts import Qwen3TTSModel
from IPython.display import Audio, display, Markdown

"""
文本转为自然语言描述的语音（基于 Qwen3-TTS）
不需要参考音频，只需要用自然语言描述，生成一种全新的声音

模型：
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign

dependency packages
pip install -U qwen-tts
pip install soundfile
pip install ipython
"""
# 加载模型
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda"
)

# 文本内容集合
texts = [
    "王总您好，提醒您一下，今天下午两点在三号会议室有一个关于新项目的推进会，各部门经理都会出席。会议资料我已经打印好放在您的办公桌左侧了，另外，五点钟您和李总还有一个视频会议。",
    "其实最近，我总有一种特别强烈的无力感。就是觉得每天都在连轴转，做着差不多的人生选择，但好像根本不知道自己到底在图什么。",
    "听你这么一说，我突然觉得现在的努力都有了具象的意义。以前总觉得在这个大城市打拼挺迷茫的，但现在想到未来的那个家有你在，就一点都不怕了。",
    "难道我们要一辈子租房吗？你看看现在的房租，每年都在涨，随时可能被房东赶走。我们已经在一起五年了，你每次聊到结婚、聊到未来，都是“再等等”，你到底要让我等到什么时候？",
    "谢谢你参与了我的生活，包容我的小脾气，听我那些奇奇怪怪的碎碎念。在这个充满变数的世界里，你是我最确定、最安心的归宿。今后的日子，换我来好好守护你，好不好？"
]

# 保存语音的目录
output_dir = "data/output_audio"

# 检查保存的目录，如不存在则创建目录
os.makedirs(output_dir, exist_ok=True)

# 遍历文本内容集合，转为语音播放
for index, text in enumerate(texts, start=1):
    audio = tts.generate_voice_design(
        text=text,
        language="chinese",
        instruct="甜妹风格，声音软萌可爱，轻松自然，偶尔带一点笑意，像正在和朋友聊天。",
        max_new_tokens=1024,
        temperature=0.5,
        top_k=40,
        repetition_penalty=1.05
    )

    waveform = audio[0][0]
    sample_rate = audio[1]

    # 语音文件
    file_path = os.path.join(output_dir, f"voice_design_{index:02}.wav")

    # 保存语音
    sf.write(
        file_path,
        waveform,
        sample_rate
    )

    display(Markdown(f"### 🎙️ **Voice Design**"))
    display(Markdown(f"**文本内容：** {text}"))
    display(Markdown(f"语音文件已自动保存到： `{file_path}`"))
    display(Audio(waveform, autoplay=False, rate=sample_rate))
