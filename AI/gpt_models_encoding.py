# coding=utf-8
import tiktoken
import pandas as pd

"""
OpenAI GPT Model 使用的编码

dependency packages
pip install tiktoken
pip install pandas
"""

# 获取 OpenAI GPT Models 对应的 TikToken 编码
models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "davinci-002", "gpt-5"]

data = []

for model in models:
    encoding = tiktoken.encoding_for_model(model)
    row = [model, encoding.name]
    data.append(row)

# 将数据写入 pandas data frame
df = pd.DataFrame(data, columns=['model', 'encoding'])

print("OpenAI GPT Models Encoding\n")
display(df[['model', 'encoding']])
