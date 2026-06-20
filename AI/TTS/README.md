# 文本转语音 TTS (Text To Speech)

**文本转语音（Text-to-Speech, 简称 TTS）** 是一种将计算机中的文本信息（Text）转化为自然流畅的语音信号（Speech）的人工智能技术

它赋予了机器 “说话” 的能力，是人机交互（HCI）与自然语言处理（NLP）领域的重要组成部分

核心处理流程：

```text
[文本输入] ──> 1. 前端处理 (NLP) ──> 2. 声学模型 (Acoustic) ──> 3. 声码器 (Vocoder) ──> [语音输出]
```

**前端处理 (NLP)：** 负责文本归一化、分词、多音字消歧，并将文本转化为音素序列

**声学模型 (Acoustic)：** 将音素序列映射为表达声音特征的梅尔谱图（Mel-spectrogram）

**声码器 (Vocoder)：** 将梅尔谱图还原、合成为高质量的原始音频波形（WAV/MP3）

## Microsoft Edge TTS

**edge_tts** 并不是微软官方发布的离线 SDK，而是一个开源的第三方库（调用需要联网）

工作原理：模拟 Microsoft Edge 浏览器的“大声朗读”（Read Aloud）功能

通过 WebSocket 协议直接向微软的云端 TTS（从文本到语音）服务器发送请求

- [基于 Microsoft Edge TTS 实现的文本转语音](./text_to_speech_by_edge.py)

  基于 Microsoft Edge TTS 模型，实现文本转语音功能，可选择适合的语音声音播放（男声/女生，国语/粤语/台语）

- [基于 Microsoft Edge TTS 实现的文本并发转多种音色语音](./parallel_text_to_speech_by_edge.py)

  基于 Microsoft Edge TTS 模型，实现文本并发转多种音色语音功能，包括（男声/女声，国语/粤语/台语）多种音色的语音
