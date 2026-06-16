# coding=utf-8
import edge_tts
import nest_asyncio
from IPython.display import Audio, display

"""
文本转为语音（基于 Microsoft Edge TTS）

dependency packages
pip install edge-tts
pip install nest_asyncio
"""
nest_asyncio.apply()

# 语音
VOICE = "zh-CN-XiaoyiNeural"

# 文本转为语音
async def speak_async(text):
    # 在 edge_tts 中可以加入 rate 参数来调整语速（例如 rate="+10%" 变快，rate="-10%" 变慢）
    communicate = edge_tts.Communicate(text, VOICE)
    audio_bytes = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes += chunk["data"]

    # rate（采样率）edge-tts 默认采样率为 24kHz
    display(Audio(audio_bytes, autoplay=False, rate=24000))

# 文本内容
text = "王总您好，提醒您一下，今天下午两点在三号会议室有一个关于新项目的推进会，各部门经理都会出席。会议资料我已经打印好放在您的办公桌左侧了，另外，五点钟您和李总还有一个视频会议。"

# 转换为语音
await speak_async(text)
