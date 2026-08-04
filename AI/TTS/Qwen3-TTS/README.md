# Qwen3-TTS

**Qwen3-TTS** 是阿里巴巴通义千问团队于 2026 年开源的新一代文本转语音（Text-to-Speech，TTS）模型系列

支持高质量语音合成、声音克隆（Voice Clone）、声音设计（Voice Design）以及自然语言控制音色、情绪、语速和语气等能力

在保证接近真人发音效果的同时，实现了超低延迟流式推理，并支持中文、英文、日文、韩文等 10 种主流语言

适用于 AI 语音助手、数字人、有声书、视频配音及智能客服等多种应用场景，是目前开源 TTS 领域中兼具音质、可控性与语音克隆能力的代表性模型之一

## 功能示例列表

- [基于 Qwen3-TTS 实现的文本转语音](./qwen3_tts_custom_voice.py)

  基于 Qwen3-TTS 模型，实现文本转语音功能，可选择适合的语音声音播放（Custom Voice）

- [基于 Qwen3-TTS 实现的文本转克隆的语音](./qwen3_tts_voice_clone.py)

  基于 Qwen3-TTS 模型，使用用户提供的音色文件提取特征，实现文本转克隆的语音功能（Voice Clone）

- [基于 Qwen3-TTS 实现的文本转自然语言描述的语音](./qwen3_tts_voice_design.py)

  基于 Qwen3-TTS 模型，不需要参考音频，只需要用自然语言描述，生成一种全新的声音

  实现文本转自然语言描述的语音功能（Voice Design）

- [基于 Qwen3-TTS 实现的可视化文本转语音](./qwen3_tts_visual_custom_voice.py)

  基于 Qwen-3-TTS 模型，实现可视化文本转语音功能，可输入文本内容，选择音色库的语音声音播放（Custom Voice）

  支持声音风格描述（Instruct），采样温度（Temperature），候选数量（Top-K），重复惩罚（Repetition Penalty）调整

- [基于 Qwen3-TTS 实现的可视化文本转自然语言描述的语音](./qwen3_tts_visual_voice_design.py)

  基于 Qwen-3-TTS 模型，不需要参考音频，只需要用自然语言描述，生成一种全新的声音

  实现可视化文本转自然语言描述的语音功能（Voice Design）

  支持声音风格描述（Instruct），采样温度（Temperature），候选数量（Top-K），重复惩罚（Repetition Penalty）调整

- [基于 Qwen3-TTS 实现的可视化文本转克隆的语音](./qwen3_tts_visual_voice_clone.py)

  基于 Qwen-3-TTS 模型，使用用户提供的音色文件提取特征，实现可视化文本转克隆的语音功能（Voice Clone）

  支持采样温度（Temperature），候选数量（Top-K），重复惩罚（Repetition Penalty）调整
