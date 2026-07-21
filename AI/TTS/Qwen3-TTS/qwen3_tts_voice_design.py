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
texts = {
    "情感独白": "其实最近，我总觉得生活过得特别快，每天都在忙碌，却很少有时间认真停下来想一想，自己真正想要的到底是什么。",
    "恋人对白": "谢谢你一直陪在我的身边，包容我的任性，也鼓励我勇敢面对生活中的困难。未来还有很长的路，希望我们能够一起慢慢走下去。",
    "安慰别人": "没关系，这一次没有做好并不代表以后也会失败。每个人都会经历低谷，只要继续努力，总会迎来属于自己的机会，我相信你一定可以做到。",
    "生气质问": "为什么每次遇到问题，你都选择沉默？我们明明可以一起面对，可你总是什么都不说。这样的结果，真的就是你想看到的吗？",
    "开心分享": "太好了，我终于通过了这次考试！虽然准备的过程特别辛苦，但看到结果的时候，一切努力都变得值得了，真的特别开心。",
}

# 自然语言描述集合
instructs = {
    "温暖治愈姐姐": "温柔知性的姐姐，二十五岁左右，声音柔和，语气耐心，有亲和力，像深夜电台主持人。",
    "元气少女": "甜妹风格，声音软萌可爱，轻松自然，偶尔带一点笑意，像正在和朋友聊天。",
    "高冷御姐": "御姐风格，气质优雅声音，声音柔美细腻，带一点高贵感，成熟自信，讲话从容，富有魅力，但不过度夸张。",
    "客服姐姐": "客服小姐姐，礼貌热情，声音亲切自然，语气真诚，耐心解答用户问题。",
    "幼儿园老师": "幼儿园老师，声音温柔耐心，富有感染力，语气充满鼓励和关爱。",
    "心理咨询师": "心理咨询师，语速缓慢，声音柔和，有安抚情绪的力量，富有同理心。",
    "情感恋人": "恋人之间聊天，声音温柔甜美，带一点撒娇语气，充满关心和陪伴感，自然真诚，富有情绪变化。",
    "ASMR风格女生": "ASMR风格女生，声音非常轻柔，贴近耳边轻声讲话，语速缓慢，令人放松。",
}

# 遍历文本内容集合，转为语音播放
for label, text in texts.items():

    # 保存语音的目录
    output_dir = f"data/output_audio/{label}"

    # 检查保存的目录，如不存在则创建目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历自然语言描述集合
    for name, instruct in instructs.items():
        audio = tts.generate_voice_design(
            text=text,
            language="chinese",
            instruct=instruct,
            max_new_tokens=1024,
            temperature=0.5,
            top_k=40,
            repetition_penalty=1.05
        )

        waveform = audio[0][0]
        sample_rate = audio[1]

        # 语音文件
        file_path = os.path.join(
            output_dir, f"voice_design_{name}.wav")

        # 保存语音
        sf.write(
            file_path,
            waveform,
            sample_rate
        )

        display(Markdown(f"### 🎙️ **Voice Design** {label} - {name}"))
        display(Markdown(f"**文本内容：** {text}"))
        display(Markdown(f"语音文件已自动保存到： `{file_path}`"))
        display(Audio(waveform, autoplay=False, rate=sample_rate))

    display(Markdown("---"))
