# coding=utf-8
import tiktoken

"""
OpenAI GPT Model 使用的编码

dependency packages
pip install tiktoken
"""

# 获取 OpenAI GPT Models 对应的 TikToken 编码
models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "davinci-002", "gpt-5"]

for model in models:
    encoding = tiktoken.encoding_for_model(model)
    print("Model: %s\nTikToken Encoding: %s\n" % (model, encoding.name))
