# coding=utf-8
import edge_tts
import nest_asyncio
from IPython.display import Audio, display, Markdown

"""
文本转为语音（基于 Microsoft Edge TTS）

dependency packages
pip install edge-tts
pip install nest_asyncio
"""
nest_asyncio.apply()

"""
Edge-TTS 支持的语音

普通话（zh-CN）
女声：
zh-CN-XiaoxiaoNeural（晓晓：最经典的通用女声，声音温柔、情感丰富）
zh-CN-XiaoyiNeural（晓伊：声音年轻、更有活力）

男声：
zh-CN-YunyangNeural（云扬：新闻播报风，大气沉稳，非常适合读小说或做旁白）
zh-CN-YunjianNeural（云健：偏向商务、影视解说风格）

地方方言：
粤语（zh-HK）
zh-HK-HiuGaaiNeural（女）
zh-HK-WanLungNeural（男）

台湾腔（zh-TW）
zh-TW-HsiaoChenNeural（晓臻：甜美台湾女声）
zh-TW-YunJheNeural（云哲：台湾男声）
"""

# 声音标注
VOICE_LABELS = {
    "zh-CN-XiaoxiaoNeural": "普通话-晓晓（女）：最经典的通用女声，声音温柔、情感丰富",
    "zh-CN-XiaoyiNeural": "普通话-晓伊（女）：声音年轻、更有活力，适合青春、时尚或活泼风格的旁白",
    "zh-CN-YunyangNeural": "普通话-云扬（男）：新闻播报风，大气沉稳，非常适合读小说或做旁白",
    "zh-CN-YunjianNeural": "普通话-云健（男）：偏向商务、影视解说风格",
    "zh-HK-HiuGaaiNeural": "粤语-晓佳（女）：声音温和自然、极具亲和力，适合日常与情感类文本",
    "zh-HK-WanLungNeural": "粤语-云龙（男）：声音相对沉稳、专业、有力，适合新闻播报与商务演示",
    "zh-TW-HsiaoChenNeural": "台湾腔-晓臻（女）：甜美台湾女声，语调柔和细腻，适合生活流与温柔对白",
    "zh-TW-YunJheNeural": "台湾腔-云哲（男）：标准台湾男声，语气温和自然，适合日常对话与电台广播"
}

# 文本转为语音
async def speak_async(text, voice_labels):
    for voice, label in voice_labels.items():
        # 在 edge_tts 中可以加入 rate 参数来调整语速（例如 rate="+10%" 变快，rate="-10%" 变慢）
        communicate = edge_tts.Communicate(text, voice)
        audio_bytes = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_bytes += chunk["data"]

        # 语音标签
        display(Markdown(f"**当前使用的语音:** `{label}`"))

        # rate（采样率）edge-tts 默认采样率为 24kHz(24000)
        display(Audio(audio_bytes, autoplay=False))

# 文本内容
text = "王总您好，提醒您一下，今天下午两点在三号会议室有一个关于新项目的推进会，各部门经理都会出席。会议资料我已经打印好放在您的办公桌左侧了，另外，五点钟您和李总还有一个视频会议。"

# 转换为各种声音的语音
await speak_async(text, VOICE_LABELS)
