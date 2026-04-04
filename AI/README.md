# AI 项目说明

## 初始化

[项目初始化说明文档](./docs/INITIALIZATION.md)

本地环境初始化（虚拟环境、Jupyter、环境变量）、Google Colab 配置以及模型 token 编码

---

## 项目列表

### [Base](./Base)

- [AI 写情信](./Base/write_love_letter.py)

  让 AI 写一封情信

- [AI 查询产品价格范围](./Base/product_price_range.py)

  让 AI 预测一件产品的销售价格范围

- [AI 情感分析评论](./Base/sentiment_analysis.py)

  让 AI 根据正面与负面的评论例子，判断给出的评论是正面还是负面

- [AI 客服](./Base/customer_service.py)

  让 AI 根据提示，返回客服回复用户的内容（随机多种回复文案）

- [AI 聊天机器人 V1](./Base/chat_robot_v1.py)

  让 AI 根据用户提问与对话上下文回答问题

  版本 1，所有提问与回答都会作为下次提问的上下文数据

- [AI 游戏百科全书](./Base/game_encyclopedia.py)

  让 AI 身份是游戏百科全书，与用户进行对话，回答用户提出的游戏问题

- [OpenAI GPT 模型编码](./Base/gpt_models_encoding.py)

  获取 OpenAI GPT 模型使用的编码

- [AI 检测文字内容是否符合法规](./Base/moderation.py)

  检测文字内容是否符合法规

  包含 `hate`, `hate/threatening`, `self-harm`, `sexual`, `sexual/minors`, `violence`, `violence/graphic` 则不符合法规

- [AI 文本插入器](./Base/text_inserter.py)

  让 AI 根据前文与后文，插入中间文本内容

- [AI 产品数据生成器](./Base/product_data_generator.py)

  让 AI 根据需求，生成产品数据用于测试

---

### [Models](./Models)

- [OpenAI GPT 模型列表](./Models/gpt_models.py)

  获取 OpenAI GPT 支持的模型列表

- [Google Gemini 模型列表](./Models/google_gemini_models.py)

  获取 Google Gemini 支持的模型列表

- [Kimi 模型列表](./Models/kimi_models.py)

  获取 Kimi 支持的模型列表

---

### [Embedding](./Embedding)

- [文本分类 Text Classification](./Embedding/TextClassification/)

  文本分类模型训练与测试（随机森林，逻辑回归，梯度提升决策树等）

- [文本聚类 Text Clustering](./Embedding/TextClustering/)

  文本聚类算法（K-Means）测试，包括聚类主题总结，n_init，轮廓系数分析，可视化聚类分布等

- [基于 Sentence Transformers 实现文本转向量](./Embedding/sentence_transformers_embedding.py)

- [基于 Hugging Face Embedding 实现文本转向量](./Embedding/hugging_face_embedding.py)

- [基于 Embedding 实现情感分析](./Embedding/sentiment_analysis.py)

- [测试基于 Embedding 计算的评论分级准确率](./Embedding/comment_level.py)

---

### [CorpusSearch](./CorpusSearch)

提供语料库，让 AI 根据语料库内容回答问题

- [版本 1，基于 llama-index 实现](./CorpusSearch/read_corpus_search.py)

- [版本 2，基于 client.chat.completions.create 接口实现](./CorpusSearch/read_corpus_search_v2.py)

---

### [ChatBot](./ChatBot)

- [01. 游戏百科聊天机器人](./ChatBot/game_conversation_chatbot.py)

  基于 Gradio 实现对话界面的游戏百科聊天机器人
