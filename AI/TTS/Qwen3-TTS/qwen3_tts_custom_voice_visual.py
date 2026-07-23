# coding=utf-8
import os
import gradio as gr
import soundfile as sf
from qwen_tts import Qwen3TTSModel

"""
可视化文本转为语音（基于 Qwen3-TTS Custom Voice）

dependency packages
pip install -U qwen-tts
pip install soundfile
pip install gradio
"""
# 输出目录
OUTPUT_DIR = "data/output_audio"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Speaker 列表
SPEAKER_LABELS = {
    "vivian（女）- 温柔、成熟、亲和发音标准": "vivian",
    "serena（女）- 偏年轻、甜美、活泼": "serena",
    "sohee（女）- 偏轻柔，自然": "sohee",
    "ono_anna（女）- 日年轻女生自然亲切": "ono_anna",
    "ryan（男）- 青年男声，温和自然": "ryan",
    "eric（男）- 稳重成熟坚定": "eric",
    "aiden（男）- 年轻且有活力": "aiden",
    "dylan（男）- 磁性一些，语速自然": "dylan",
    "uncle_fu（男）- 中年感明显低沉厚重": "uncle_fu"
}

# 初始化模型
tts = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda"
)

# 供 Gradio 调用的核心转换函数
def text_to_speech_qwen3_tts(
    text,
    speaker_display_name,
    instruct,
    temperature,
    top_k,
    repetition_penalty,
):
    if not text.strip():
        raise gr.Error("请输入文本！")

    try:
        # 用户选择的音色
        speaker = SPEAKER_LABELS[speaker_display_name]

        audio = tts.generate_custom_voice(
            text=text,
            language="chinese",
            speaker=speaker,
            instruct=instruct.strip() or None,
            max_new_tokens=1024,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty
        )

        waveform = audio[0][0]
        sample_rate = audio[1]

        file_path = os.path.join(
            OUTPUT_DIR,
            f"{speaker}.wav"
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
    gr.Markdown("# 🎙️ Qwen3-TTS Custom Voice 多音色语音合成")
    gr.Markdown("支持官方 Speaker + Instruction 控制，可在线试听，也可下载 wav 文件。")

    with gr.Row():

        with gr.Column():
            # 预设默认值
            input_text = gr.Textbox(
                label="输入文本",
                value="很高兴认识你哦，我刚刚去买了杯奶茶，要一起去长椅上坐坐吗？",
                lines=6
            )

            # 选择音色
            speaker_dropdown = gr.Dropdown(
                label="音色",
                choices=list(SPEAKER_LABELS.keys()),
                value="vivian（女）- 温柔、成熟、亲和发音标准"
            )

            with gr.Accordion("高级参数", open=False):

                # 声音风格描述
                instruct_box = gr.Textbox(
                    label="声音风格描述",
                    value="声音甜美，温柔、亲切、像秘书。",
                    lines=2
                )

                # 采样温度（Temperature）
                temperature_slider = gr.Slider(
                    minimum=0,
                    maximum=1.5,
                    value=0.7,
                    step=0.05,
                    label="采样温度（Temperature）"
                )

                # 候选数量（Top-K）
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    label="候选数量（Top-K）"
                )

                # 重复惩罚（Repetition Penalty）
                repetition_slider = gr.Slider(
                    minimum=1.0,
                    maximum=2.0,
                    value=1.1,
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
            speaker_dropdown,
            instruct_box,
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
