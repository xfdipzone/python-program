# OpenAI 项目说明

## 测试环境初始化

使用 `Google Colab` 环境测试

[https://colab.research.google.com/](<https://colab.research.google.com/>)

创建 `*.ipynb` 文件保存测试代码

```shell
!pip install openai numpy tiktoken
%env OPENAI_API_KEY=[填写您的 OpenAI API Key]
```

Model 使用的编码

参考：[https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb](<https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb>)

使用 tiktoken 统计 token 数量时，需要使用与 Model 一样的编码

例如 Model=`gpt-4o-mini`，则 `tiktoken` 配置以下的编码

```python3
encoding = tiktoken.get_encoding("o200k_base")
```

---

## 项目列表

[01. AI 写情信](./write_letter.py)

让 AI 写一封情信

[02. AI 查询产品价格范围](./product_price_range.py)

让 AI 预测一件产品的销售价格范围

[03. AI 情感分析评论](./sentiment_analysis.py)

让 AI 根据正面与负面的评论例子，判断给出的评论是正面还是负面

[04. AI 客服](./customer_service.py)

让 AI 根据提示，返回客服回复用户的内容（随机多种回复文案）

[05. AI 对话机器人](./game_encyclopedia.py)

让 AI 身份是游戏百科全书，与用户进行对话，回答用户提出的游戏问题

[06. OpenAI 模型列表](./models.py)

获取 OpenAI 支持的模型列表

[07. OpenAI 检测文字内容是否符合法规](./moderation.py)

检测文字内容是否符合法规

包含 `hate`, `hate/threatening`, `self-harm`, `sexual`, `sexual/minors`, `violence`, `violence/graphic` 则不符合法规

[08. AI 文本插入器](./text_inserter.py)

让 AI 根据前文与后文，插入中间文本内容

[09. AI 产品数据生成器](./product_data_generator.py)

让 AI 根据需求，生成产品数据用于测试

---

## 聊天机器人

[01. 游戏百科聊天机器人](./ChatBot/game_conversation_chatbot.py)

基于 Gradio 实现对话界面的游戏百科聊天机器人
