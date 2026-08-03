# coding=utf-8
import os
import gradio as gr
import soundfile as sf
import threading
from qwen_tts import Qwen3TTSModel
from datetime import datetime
from zoneinfo import ZoneInfo

"""
可视化文本转为克隆的语音（基于 Qwen3-TTS Voice Clone）
使用用户提供的音色文件提取特征

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
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0"
)

# 并发锁
tts_lock = threading.Lock()

# 供 Gradio 调用的核心转换函数
def text_to_speech_qwen3_tts(
    text,
    reference_audio,
    reference_text,
    temperature,
    top_k,
    repetition_penalty,
    voice_clone_cache
):
    if not text.strip():
        raise gr.Error("请输入文本！")

    if reference_audio is None:
        raise gr.Error("请上传参考声音！")

    if not reference_text.strip():
        raise gr.Error("请输入参考声音的文本！")

    # 初始化
    if voice_clone_cache is None:
        voice_clone_cache = {
            "audio": None,
            "text": None,
            "prompt": None
        }

    try:
        # 判断是否需要重新生成 voice clone prompt
        need_create_prompt = (
            voice_clone_cache.get("prompt") is None
            or voice_clone_cache.get("audio") != reference_audio
            or voice_clone_cache.get("text") != reference_text
        )

        with tts_lock:
            if need_create_prompt:
                # 分析语音特征生成 prompt（复用）
                voice_clone_prompt = tts.create_voice_clone_prompt(
                    ref_audio=reference_audio,
                    ref_text=reference_text,
                )

                voice_clone_cache = {
                    "audio": reference_audio,
                    "text": reference_text,
                    "prompt": voice_clone_prompt
                }

            audio = tts.generate_voice_clone(
                text=text,
                language="chinese",
                voice_clone_prompt=voice_clone_cache["prompt"],
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

        return file_path, file_path, voice_clone_cache

    except Exception as e:
        raise gr.Error(f"合成失败：{str(e)}")


# 构建 Gradio 网页界面
with gr.Blocks(title="Qwen3-TTS") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Voice Clone 克隆语音合成")
    gr.Markdown("使用克隆的音色生成语音，可在线试听，也可下载 wav 文件。")

    voice_clone_state = gr.State(
        {
            "audio": None,
            "text": None,
            "prompt": None
        }
    )

    with gr.Row():

        with gr.Column():
            # 预设默认值
            input_text = gr.Textbox(
                label="输入文本",
                value="很高兴认识你哦，我刚刚去买了杯奶茶，要一起去长椅上坐坐吗？",
                lines=6
            )

            # 参考声音文件
            reference_audio = gr.File(
                label="上传参考声音",
                file_types=[".wav"],
                type="filepath"
            )

            # 参考声音文本
            reference_text = gr.Textbox(
                label="参考声音文本",
                value="",
                lines=2
            )

            with gr.Accordion("高级参数", open=False):

                # 采样温度（Temperature）
                temperature_slider = gr.Slider(
                    minimum=0,
                    maximum=1.5,
                    value=0.3,
                    step=0.05,
                    label="采样温度（Temperature）"
                )

                # 候选数量（Top-K）
                topk_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=10,
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
            reference_audio,
            reference_text,
            temperature_slider,
            topk_slider,
            repetition_slider,
            voice_clone_state
        ],
        outputs=[
            audio_output,
            file_output,
            voice_clone_state
        ]
    )


# 启动
demo.queue(max_size=20).launch(inline=True, share=True)
