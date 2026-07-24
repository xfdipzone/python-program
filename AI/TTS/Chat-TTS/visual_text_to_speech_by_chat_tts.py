# coding=utf-8
import ChatTTS
import soundfile as sf
import os
import gradio as gr

"""
可视化文本转语音（基于 ChatTTS）

dependency packages
pip install chattts
pip install soundfile
pip install gradio
pip install torch
"""
# 音色库目录
VOICE_LIBRARY_DIR = "data/chat-tts_voice_library"

# 输出目录
OUTPUT_DIR = "data/output_audio"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 音色列表
VOICE_LABELS = {
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

# 初始化模型
chat = ChatTTS.Chat()
chat.load(compile=False)

# 文本转语音
def text_to_speech_chat_tts(
    text,
    voice_display_name,
    temperature,
    top_p,
    top_k,
):
    if not text.strip():
        raise gr.Error("请输入文本！")

    try:
        # 用户选择的音色
        speaker_name = VOICE_LABELS[voice_display_name]

        speaker_path = os.path.join(
            VOICE_LIBRARY_DIR,
            f"{speaker_name}.txt"
        )

        if not os.path.exists(speaker_path):
            raise gr.Error(f"找不到音色文件：{speaker_path}")

        with open(
            speaker_path,
            "r",
            encoding="utf-8"
        ) as f:
            spk = f.read()

        wavs = chat.infer(
            [text],
            params_infer_code=ChatTTS.Chat.InferCodeParams(
                spk_emb=spk,
                temperature=temperature,
                top_P=top_p,
                top_K=top_k
            )
        )

        audio = wavs[0]

        if hasattr(audio, "shape") and len(audio.shape) > 1:
            audio = audio[0]

        file_path = os.path.join(
            OUTPUT_DIR,
            f"{speaker_name}.wav"
        )

        sf.write(
            file_path,
            audio,
            24000
        )

        return file_path, file_path

    except Exception as e:
        raise gr.Error(f"合成失败：{str(e)}")


# Gradio 页面
with gr.Blocks(title="ChatTTS 可视化语音合成器") as demo:
    gr.Markdown("# 🎙️ ChatTTS 可视化语音合成器")
    gr.Markdown("基于 ChatTTS 的本地语音生成，可选择不同 Speaker 音色，支持试听与下载。")

    with gr.Row():

        with gr.Column():
            # 预设默认值
            input_text = gr.Textbox(
                label="请输入要转换的文本",
                value="[speed_3] 亲爱的主人，您好呀 [uv_break] 我叫小莹 [uv_break] 是您的专属小助手，[uv_break] 此刻正带着满满的暖意，给您送来这份问候呢。",
                placeholder="请输入文本...",
                lines=6
            )

            # 选择音色
            voice_dropdown = gr.Dropdown(
                label="选择音色",
                choices=list(VOICE_LABELS.keys()),
                value="活泼女性"
            )

            with gr.Accordion("高级参数", open=False):

                # 随机性（Temperature）
                temperature_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.3,
                    step=0.05,
                    label="随机性（Temperature）"
                )

                # 采样范围（Top P）
                top_p_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.7,
                    step=0.05,
                    label="采样范围（Top P）"
                )

                # 候选数量（Top K）
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    label="候选数量（Top K）"
                )

            btn = gr.Button("⚡ 开始合成语音", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="🎧 在线试听", type="filepath")
            file_output = gr.File(label="📥 下载保存 WAV 音频文件")

    # 绑定按钮点击事件
    btn.click(
        fn=text_to_speech_chat_tts,
        inputs=[
            input_text,
            voice_dropdown,
            temperature_slider,
            top_p_slider,
            top_k_slider
        ],
        outputs=[
            audio_output,
            file_output
        ]
    )


# 启动
demo.launch(inline=True, share=True)
