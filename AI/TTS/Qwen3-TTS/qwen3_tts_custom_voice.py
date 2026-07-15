# coding=utf-8
import soundfile as sf
import os
from qwen_tts import Qwen3TTSModel
from IPython.display import Audio, display, Markdown

"""
文本转为语音（基于 Qwen3-TTS）
使用指定的 Speaker

模型：
https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
3.9GB，显存需求 6-8 GB，推理速度较慢，音质更自然，更稳定，情绪表达丰富，长文本稳定性高，支持 Instruction 控制

https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
1.8GB，显存需求 2-4 GB，推理速度很快，音质很好，情绪表达一般，长文本稳定性弱，不支持 Instruction 控制

dependency packages
pip install -U qwen-tts
pip install soundfile
pip install ipython
"""
# 加载模型
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda"
)

# 文本内容
text = "王总您好，提醒您一下，今天下午两点在三号会议室有一个关于新项目的推进会，各部门经理都会出席。会议资料我已经打印好放在您的办公桌左侧了，另外，五点钟您和李总还有一个视频会议。"

# 保存语音的目录
output_dir = "data/output_audio"

# 检查保存的目录，如不存在则创建目录
os.makedirs(output_dir, exist_ok=True)

# 获取 Speakers 列表
speakers = tts.get_supported_speakers()

# 遍历所有 Speaker 播放
for speaker in speakers:
    try:
        audio = tts.generate_custom_voice(
            text=text,
            language="chinese",
            speaker=speaker,
            instruct="声音甜美，温柔、亲切、像秘书。",
            max_new_tokens=1024,
            temperature=0.7,
            top_k=20,
            repetition_penalty=1.1
        )

        waveform = audio[0][0]
        sample_rate = audio[1]

        # 语音文件
        file_path = os.path.join(output_dir, f"{speaker}_custom_voice.wav")

        # 保存语音
        sf.write(
            file_path,
            waveform,
            sample_rate
        )

        display(Markdown(f"### 🎙️ **Speaker: {speaker}**"))
        display(Markdown(f"语音文件已自动保存到： `{file_path}`"))
        display(Audio(waveform, autoplay=False, rate=sample_rate))

    except Exception as e:
        print(f"{speaker} 生成失败：{e}")

"""
| Speaker  | 性别 | 音色的特点               | 适合场景
| -------------------------------------------------------------------------
| vivian   | 女性 | 温柔、成熟、亲和，发音标准 | 客服、秘书、AI 助手（很受欢迎）
| serena   | 女性 | 偏年轻、甜美、活泼        | 导航、教育、儿童内容
| sohee    | 女性 | 偏轻柔自然               | 陪伴机器人、聊天
| ono_anna | 女性 | 日年轻女生自然亲切        | 日语 AI 助手、日语学习、游戏 NPC
| ryan     | 男性 | 青年男声，温和自然        | AI 助手、智能家居
| eric     | 男性 | 稳重、成熟               | 商务、企业播报
| aiden    | 男性 | 年轻有活力               | 对话机器人
| dylan    | 男性 | 磁性一些，语速自然        | 有声内容
| uncle_fu | 男性 | 中年感明显低沉厚重        | 长辈、广播、讲故事
"""
