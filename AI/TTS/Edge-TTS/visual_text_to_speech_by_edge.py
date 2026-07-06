# coding=utf-8
import edge_tts
import os
import gradio as gr

"""
可视化文本转为语音（基于 Microsoft Edge TTS）

dependency packages
pip install edge-tts
pip install gradio
pip install uvicorn
"""
# 声音标注
VOICE_LABELS = {
    "普通话-晓晓（女）：温柔、情感丰富": "zh-CN-XiaoxiaoNeural",
    "普通话-晓伊（女）：年轻活力、活泼风格": "zh-CN-XiaoyiNeural",
    "普通话-云扬（男）：新闻播报风、大气沉稳": "zh-CN-YunyangNeural",
    "普通话-云健（男）：商务、影视解说": "zh-CN-YunjianNeural",
    "粤语-晓佳（女）：温和自然、极具亲和力": "zh-HK-HiuGaaiNeural",
    "粤语-晓曼（女）：知性沉稳、清晰演示": "zh-HK-HiuMaanNeural",
    "粤语-云龙（男）：专业有力、新闻演示": "zh-HK-WanLungNeural",
    "台湾腔-晓臻（女）：甜美柔和、生活对白": "zh-TW-HsiaoChenNeural",
    "台湾腔-云哲（男）：语气温和、电台广播": "zh-TW-YunJheNeural"
}

# 文本转为语音
async def _fetch_single_audio_bytes(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    audio_bytes = bytearray()

    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])

    return bytes(audio_bytes)


# 供 Gradio 调用的核心转换函数
async def text_to_speech_edge(text, voice_display_name, output_dir="data/output_audio"):
    if not text.strip():
        return None, "请输入文字！"

    # 检查并创建保存目录
    os.makedirs(output_dir, exist_ok=True)

    # 根据用户在网页选择的声音风格，获取实际的 Edge-TTS 音色 ID
    actual_voice_id = VOICE_LABELS[voice_display_name]

    # 运行异步合成任务
    try:
        audio_bytes = await _fetch_single_audio_bytes(text, actual_voice_id)

        # 保存音频到本地
        file_path = os.path.join(output_dir, f"{actual_voice_id}.mp3")
        with open(file_path, "wb") as f:
            f.write(audio_bytes)

        return file_path, file_path

    except Exception as e:
        raise gr.Error(f"Edge-TTS 合成失败: {str(e)}")


# 构建 Gradio 网页界面
with gr.Blocks(title="Edge-TTS 语音合成器") as demo:
    gr.Markdown("# 🎙️ Microsoft Edge-TTS 语音多音色合成器")
    gr.Markdown("借助微软 Edge 神经网络语音，无需显卡，秒级生成高质量、自然逼真的真人语音。")

    with gr.Row():
        with gr.Column():
            # 预设默认值
            input_text = gr.Textbox(
                label="请输入要转换的文本",
                value="王总您好，提醒您一下，今天下午两点在三号会议室有一个关于新项目的推进会，各部门经理都会出席。会议资料我已经打印好放在您的办公桌左侧了，另外，五点钟您和李总还有一个视频会议。",
                placeholder="在此输入文本内容...",
                lines=5
            )

            # 选择声音风格
            voice_dropdown = gr.Dropdown(
                label="选择声音风格 (包含普通话、粤语、台湾腔)",
                choices=list(VOICE_LABELS.keys()),
                value="普通话-晓伊（女）：年轻活力、活泼风格"  # 默认选择你活力女声
            )

            btn = gr.Button("⚡ 开始合成语音", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="🎧 在线试听播放", type="filepath")
            file_output = gr.File(label="📥 下载保存 MP3 音频文件")

    # 绑定按钮点击事件
    btn.click(
        fn=text_to_speech_edge,
        inputs=[input_text, voice_dropdown],
        outputs=[audio_output, file_output]
    )


# 启动
demo.launch(inline=True, share=True)
