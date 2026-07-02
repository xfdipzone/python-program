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

- [基于 Microsoft Edge TTS 实现的文本转语音](./Edge-TTS/text_to_speech_by_edge.py)

  基于 Microsoft Edge TTS 模型，实现文本转语音功能，可选择适合的语音声音播放（男声/女声，国语/粤语/台语）

- [基于 Microsoft Edge TTS 实现的文本并发转多种音色语音](./Edge-TTS/parallel_text_to_speech_by_edge.py)

  基于 Microsoft Edge TTS 模型，实现文本并发转多种音色语音功能，包括（男声/女声，国语/粤语/台语），支持播放与保存

- [获取 Microsoft Edge TTS 所有中文语音音色](./Edge-TTS/edge_tts_voices_list.py)

  获取 Microsoft Edge TTS 所有中文声音列表，包括（男声/女声，国语/粤语/台语/辽宁话/陕西话）

---

## Chat TTS

**ChatTTS** 是一款专门为**对话场景**（如长文本朗读、小说播客、游戏 NPC 对话）设计的开源语音合成（TTS）模型

它最大的特点和优势可以概括为以下三点：

- 超自然的对话感： 与传统机械、生硬的 TTS 不同，ChatTTS 能够生成极其逼真的笑声、叹气声、语气词（如“呃”、“啊”），以及非常自然的停顿

- 多说话人与声音控制： 支持上万种不同音色的预测和控制，可以细粒度地调节说话人的语速、语气和情感

- 开源与高效： 对学术界和民间开发者开源，且在主流消费级显卡（GPU）上就能实现快速推理，非常适合二次开发

简而言之，它是目前开源领域里让机器说话最像“真人聊天”的模型之一

**注意：** 不同的文本内容，有可能输出的语音不一样（男声变女声，声音换了另一个等）

因此不适合使用在交互对话的场景，适合在没有上下文关联的一整段文本播放

- [Chat-TTS 音色库管理](./Chat-TTS/speaker_manager.py)

  基于 Chat-TTS 随机生成 50 个音色，用于语音合成，方便对应不同的使用场景

- [基于 Chat-TTS 实现的文本转语音](./Chat-TTS/text_to_speech_by_chat_tts.py)

  基于 Chat-TTS 模型，实现文本转语音功能，可选择音色库的语音声音播放 (Chat-TTS voice library)

- [基于 Chat-TTS 筛选出 Top 20 好听的年轻女声](./Chat-TTS/filter_top20_female_speaker.py)

  基于 Chat-TTS 随机生成 100 个音色，然后通过音高评分，筛选出 Top 20 好听的年轻女声

  限定了算法寻找人类说话音高的范围（ 60Hz 到 500Hz ）

  男性的基频通常在 85 ~ 180Hz，女性通常在 165 ~ 255Hz，儿童或极高亢的声音会更高
