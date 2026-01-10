# coding=utf-8
import tiktoken

"""
OpenAI GPT Model 使用的编码

dependency packages
pip install tiktoken
"""

# 获取 OpenAI GPT Model 对应的 TikToken 编码
model = "gpt-4o-mini"
encoding = tiktoken.encoding_for_model(model)
print("Model: %s\nTikToken Encoding: %s" % (model, encoding.name))
