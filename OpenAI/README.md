# OpenAI 项目说明

## 测试环境初始化

使用 `Google Colab` 环境测试

[https://colab.research.google.com/](<https://colab.research.google.com/>)

创建 `*.ipynb` 文件保存测试代码

```shell
!pip install openai numpy
%env OPENAI_API_KEY=[填写您的 OpenAI API Key]
```

---

## 项目列表

[01. AI 写情信](./write_letter.py)

让 AI 写一封情信

[02. AI 查询产品价格范围](./product_price_range.py)

让 AI 预测一件产品的销售价格范围

[03. AI 情感分析评论](./sentiment_analysis.py)

让 AI 判断评论是好评还是差评

[04. AI 客服](./customer_service.py)

让 AI 根据提示，返回客服回复用户的内容（随机多种回复文案）
