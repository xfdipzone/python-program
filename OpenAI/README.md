# OpenAI 项目说明

## 测试环境初始化

使用 `Google Colab` 环境测试

[https://colab.research.google.com/](<https://colab.research.google.com/>)

创建 `*.ipynb` 文件保存测试代码

```shell
# 安装依赖包
!pip install openai numpy tiktoken

# OPENAI_API_KEY 存放在环境变量中
%env OPENAI_API_KEY=[填写您的 OpenAI API Key]

# 读取环境变量
import os
os.environ.get("OPENAI_API_KEY")
```

```shell
# OPENAI_API_KEY 存放在 Google Colab Secret 中
# 读取 Secret
from google.colab import userdata
userdata.get("OPENAI_API_KEY")
```

---

## Model 使用的编码

参考：[https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb](<https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>)

使用 tiktoken 统计 token 数量时，需要使用与 Model 一样的编码

**获取 Model 使用的编码：**

例如 Model=`gpt-4o-mini`，则使用下面代码获取使用的编码

```python3
encoding = tiktoken.encoding_for_model('gpt-4o-mini')
```

**设置 tiktoken 编码：**

例如 Model=`gpt-4o-mini`，则 `tiktoken` 配置以下的编码

```python3
encoding = tiktoken.get_encoding("o200k_base")
```

---

## 项目列表

[AI 写情信](./write_love_letter.py)

让 AI 写一封情信

[AI 查询产品价格范围](./product_price_range.py)

让 AI 预测一件产品的销售价格范围

[AI 情感分析评论](./sentiment_analysis.py)

让 AI 根据正面与负面的评论例子，判断给出的评论是正面还是负面

[AI 客服](./customer_service.py)

让 AI 根据提示，返回客服回复用户的内容（随机多种回复文案）

[AI 聊天机器人 V1](./chat_robot_v1.py)

让 AI 根据用户提问与对话上下文回答问题

版本 1，所有提问与回答都会作为下次提问的上下文数据

[AI 游戏百科全书](./game_encyclopedia.py)

让 AI 身份是游戏百科全书，与用户进行对话，回答用户提出的游戏问题

[OpenAI GPT 模型列表](./gpt_models.py)

获取 OpenAI GPT 支持的模型列表

[OpenAI GPT 模型编码](./gpt_models_encoding.py)

获取 OpenAI GPT 模型使用的编码

[OpenAI 检测文字内容是否符合法规](./moderation.py)

检测文字内容是否符合法规

包含 `hate`, `hate/threatening`, `self-harm`, `sexual`, `sexual/minors`, `violence`, `violence/graphic` 则不符合法规

[AI 文本插入器](./text_inserter.py)

让 AI 根据前文与后文，插入中间文本内容

[AI 产品数据生成器](./product_data_generator.py)

让 AI 根据需求，生成产品数据用于测试

[AI 根据语料库回答问题](./CorpusSearch)

提供语料库，让 AI 根据语料库内容回答问题

- [版本 1，基于 llama-index 实现](./CorpusSearch/read_corpus_search.py)

- [版本 2，基于 client.chat.completions.create 接口实现](./CorpusSearch/read_corpus_search_v2.py)

[Embedding](./Embedding)

- [基于 Sentence Transformers 实现文本转向量](./Embedding/sentence_transformers_embedding.py)

- [基于 Hugging Face Embedding 实现文本转向量](./Embedding/hugging_face_embedding.py)

- [基于 Embedding 实现情感分析](./Embedding/sentiment_analysis.py)

- [测试基于 Embedding 计算的评论分级准确率](./Embedding/comment_level.py)

- [基于随机森林算法（Random Forest）训练预测新闻类别](./Embedding/news_grouping_by_random_forest.py)

- [基于逻辑回归算法（Logistic Regression）训练预测新闻类别](./Embedding/news_grouping_by_logistic_regression.py)

- [基于逻辑回归算法（Logistic Regression）V2 训练预测新闻类别](./Embedding/news_grouping_by_logistic_regression_v2.py)

- [基于梯度提升决策树算法（LightGBM）训练预测新闻类别](./Embedding/news_grouping_by_lightgbm.py)

- [基于随机森林算法（Random Forest）训练预测新闻类别模型评估](./Embedding/news_grouping_by_random_forest_model_evaluation.py)

- [基于梯度提升决策树算法（LightGBM）训练预测新闻类别模型评估](./Embedding/news_grouping_by_lightgbm_model_evaluation.py)

- [训练预测新闻类别特征主成分分析（PCA）](./Embedding/news_grouping_pca.py)

- [随机森林模型（Random Forest）重要特征分析](./Embedding/news_grouping_random_forest_important_features.py)

- [逻辑回归模型（Logistic Regression）重要特征分析](./Embedding/news_grouping_logistic_regression_important_features.py)

- [梯度提升决策树模型（LightGBM）重要特征分析](./Embedding/news_grouping_lightgbm_important_features.py)

---

## 聊天机器人

[01. 游戏百科聊天机器人](./ChatBot/game_conversation_chatbot.py)

基于 Gradio 实现对话界面的游戏百科聊天机器人
