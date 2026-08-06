# 自动语音识别 ASR (Automatic Speech Recognition)

**ASR** 是 **Automatic Speech Recognition** 的缩写，即自动语音识别。简单来说，它是一种将人类说话的语音信号转化为文本文字的技术（通常称为 "语音转文字" 或 "语音转写"）

## Qwen3-ASR

**Qwen3-ASR** 是通义千问（Qwen）团队推出的多语言自动语音识别大模型系列。它继承了 Qwen3-Omni 架构对音频数据的深层理解能力，能够高效准确地将音频内容转写为文本

- [基于 Qwen3-ASR 实现自动语音识别](./Qwen3-ASR/qwen3_asr.py)

  基于 Qwen3-ASR 实现自动语音识别，支持批量识别，并输出识别后的文本内容

- [基于 Qwen3-ASR 实现语音与文本强制对齐](./Qwen3-ASR/qwen3_forced_aligner.py)

  基于 Qwen3-ASR `Qwen3-ForcedAligner-0.6B` 模型，实现语音与文本强制对齐，返回词级/字级时间戳
