# AI 项目说明

## 初始化

[项目初始化说明文档](./docs/INITIALIZATION.md)

本地环境初始化（虚拟环境、Jupyter、环境变量）、Google Colab 配置以及模型 token 编码

---

## AI 项目列表

### Base（基础应用）| [Explore](./Base/)

**定位：** 基础 AI 应用，展示 AI 一些基本功能

---

### Models（模型）| [Explore](./Models/)

**定位：** 热门模型，包括 OpenAI, Kimi, Gemini 等 AI 平台支持的具体模型列表

- [OpenAI GPT 模型列表](./Models/gpt_models.py)

  获取 OpenAI GPT 支持的模型列表

- [Google Gemini 模型列表](./Models/google_gemini_models.py)

  获取 Google Gemini 支持的模型列表

- [Kimi 模型列表](./Models/kimi_models.py)

  获取 Kimi 支持的模型列表

---

### Embedding（向量嵌入）| [Explore](./Embedding/)

**定位：** 语义向量化、机器学习、文本挖掘相关应用

- [文本分类 Text Classification](./Embedding/TextClassification/)

  文本分类模型训练与测试（随机森林，逻辑回归，梯度提升决策树等）

- [文本聚类 Text Clustering](./Embedding/TextClustering/)

  文本聚类算法（K-Means）测试，包括聚类主题总结，n_init，轮廓系数分析，可视化聚类分布等

- [零样本文本分类 Zero Shot Classification](./Embedding/ZeroShotClassification/)

  不进行任何样本训练（零样本），只基于模型计算 Embedding 相似度来实现文本分类

---

### CorpusSearch（语料库检索）| [Explore](./CorpusSearch/)

**定位：** RAG（检索增强生成），通过整合私有语料库，使 AI 能够突破通用知识的限制，针对特定文档内容进行精准回答

- [文本类语料库检索](./CorpusSearch/TextCorpusSearch/)

  对文本类语料库检索，针对文本内容进行精准回答

---

### TTS（文本转语音）| [Explore](./TTS/)

**定位：** 将计算机中的文本信息（Text）转化为自然流畅的语音信号（Speech）

- [Microsoft Edge TTS](./TTS/Edge-TTS/)

  基于 Microsoft Edge TTS 实现文本转语音

  微软开源的第三方库，通过 WebSocket 协议直接向微软的云端 TTS（从文本到语音）服务器发送请求

- [Chat-TTS](./TTS/Chat-TTS/)

  基于 Chat-TTS 实现文本转语音

  专门为对话场景（如长文本朗读、小说播客、游戏 NPC 对话）设计的开源语音合成（TTS）模型

- [Qwen3-TTS](./TTS/Qwen3-TTS/)

  基于 Qwen3-TTS 实现文本转语音

  支持高质量语音合成、声音克隆（Voice Clone）、声音设计（Voice Design）以及自然语言控制音色、情绪、语速和语气等能力

---

### ASR（自动语音识别）| [Explore](./ASR/)

**定位：** 将人类说话的语音信号转化为文本文字的技术（通常称为 "语音转文字" 或 "语音转写"）

---

### ChatBot（机器人）| [Explore](./ChatBot/)

**定位：** 机器人相关 AI 应用，包括交互设计、垂直领域应用、图形化界面

- [01. 游戏百科聊天机器人](./ChatBot/game_conversation_chatbot.py)

  基于 Gradio 实现对话界面的游戏百科聊天机器人

---

### AI 工具（常用工具集合）| [Explore](./Tools/)

**定位：** AI 项目常用工具集合
