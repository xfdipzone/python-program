# Microsoft Edge TTS

**edge_tts** 并不是微软官方发布的离线 SDK，而是一个开源的第三方库（调用需要联网）

工作原理：模拟 Microsoft Edge 浏览器的“大声朗读”（Read Aloud）功能

通过 WebSocket 协议直接向微软的云端 TTS（从文本到语音）服务器发送请求

## 项目列表

- [基于 Microsoft Edge TTS 实现的文本转语音](./text_to_speech_by_edge.py)

  基于 Microsoft Edge TTS 模型，实现文本转语音功能，可选择适合的语音声音播放（男声/女声，国语/粤语/台语）

- [基于 Microsoft Edge TTS 实现的文本并发转多种音色语音](./parallel_text_to_speech_by_edge.py)

  基于 Microsoft Edge TTS 模型，实现文本并发转多种音色语音功能，包括（男声/女声，国语/粤语/台语），支持播放与保存

- [基于 Microsoft Edge TTS 实现的可视化文本转语音](./visual_text_to_speech_by_edge.py)

  基于 Microsoft Edge TTS 模型，实现可视化文本转语音功能，可输入文本内容，选择适合的语音声音播放（男声/女声，国语/粤语/台语）

  支持语速，音量，音调调整

- [获取 Microsoft Edge TTS 所有中文语音音色](./edge_tts_voices_list.py)

  获取 Microsoft Edge TTS 所有中文声音列表，包括（男声/女声，国语/粤语/台语/辽宁话/陕西话）
