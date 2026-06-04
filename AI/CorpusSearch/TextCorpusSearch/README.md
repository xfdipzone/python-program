# 文本语料库检索（Text Corpus Search）

通过整合私有文本语料库，使 AI 能够突破通用知识的限制，针对特定文档内容进行精准回答

## 文本语料库检索

- [基于 AI 大语言模型的文本语料库搜索](./text_corpus_search_by_AI_model.py)

  基于 AI 大语言模型，例如 GPT, DeepSeek, Kimi 等，对文本语料库进行读取，生成检索索引，再根据用户问题进行精准回答

- [基于 AI Chat 的文本语料库搜索](./text_corpus_search_by_AI_chat.py)

  基于 AI Chat，例如 GPT, DeepSeek, Kimi 等，将文本语料与要回答的问题作为 prompt 提交给 AI，让 AI 根据语料进行精准回答
