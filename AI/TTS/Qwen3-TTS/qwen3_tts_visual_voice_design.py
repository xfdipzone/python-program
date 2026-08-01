# coding=utf-8
import os
import gradio as gr
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from datetime import datetime
from zoneinfo import ZoneInfo

"""
可视化文本转为自然语言描述的语音（基于 Qwen3-TTS Voice Design）
不需要参考音频，只需要用自然语言描述，生成一种全新的声音

dependency packages
pip install -U qwen-tts
pip install soundfile
pip install gradio
"""
# 输出目录
OUTPUT_DIR = "data/output_audio"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 初始化模型
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0"
)

# 供 Gradio 调用的核心转换函数
def text_to_speech_qwen3_tts(
    text,
    instruct,
    temperature,
    top_k,
    repetition_penalty,
):
    if not text.strip():
        raise gr.Error("请输入文本！")

    if not instruct.strip():
        raise gr.Error("请输入声音风格描述！")

    try:
        audio = tts.generate_voice_design(
            text=text,
            language="chinese",
            instruct=instruct,
            max_new_tokens=1024,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )

        waveform = audio[0][0]
        sample_rate = audio[1]

        # 年月日_时分秒_三位毫秒
        timestamp = datetime.now(
            ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")[:-3]

        file_path = os.path.join(
            OUTPUT_DIR,
            f"{timestamp}.wav"
        )

        # 保存音频到本地
        sf.write(
            file_path,
            waveform,
            sample_rate
        )

        return file_path, file_path

    except Exception as e:
        raise gr.Error(f"合成失败：{str(e)}")


# 构建 Gradio 网页界面
with gr.Blocks(title="Qwen3-TTS") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Voice Design 自然语言描述语音合成")
    gr.Markdown("使用自然语言描述生成语音，可在线试听，也可下载 wav 文件。")

    with gr.Row():

        with gr.Column():
            # 预设默认值
            input_text = gr.Textbox(
                label="输入文本",
                value="很高兴认识你哦，我刚刚去买了杯奶茶，要一起去长椅上坐坐吗？",
                lines=6
            )

            # 声音风格描述
            instruct_text = gr.Textbox(
                label="声音风格描述",
                value="恋人之间聊天，声音温柔甜美，带一点撒娇语气，充满关心和陪伴感，自然真诚，富有情绪变化。",
                lines=2
            )

            with gr.Accordion("高级参数", open=False):

                # 采样温度（Temperature）
                temperature_slider = gr.Slider(
                    minimum=0,
                    maximum=1.5,
                    value=0.5,
                    step=0.05,
                    label="采样温度（Temperature）"
                )

                # 候选数量（Top-K）
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=40,
                    step=1,
                    label="候选数量（Top-K）"
                )

                # 重复惩罚（Repetition Penalty）
                repetition_slider = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.05,
                    step=0.05,
                    label="重复惩罚（Repetition Penalty）"
                )

            btn = gr.Button("⚡ 开始合成语音", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="🎧 在线试听", type="filepath")
            file_output = gr.File(label="📥 下载保存 WAV 音频文件")

    # 绑定按钮点击事件
    btn.click(
        fn=text_to_speech_qwen3_tts,
        inputs=[
            input_text,
            instruct_text,
            temperature_slider,
            topk_slider,
            repetition_slider
        ],
        outputs=[
            audio_output,
            file_output
        ]
    )


# 启动
demo.launch(inline=True, share=True)
