# coding=utf-8
from openai import OpenAI
import os
import time
import tiktoken

"""
AI 客服

dependency packages
pip install openai
pip install tiktoken
"""
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

COMPLETION_MODEL = "gpt-4o-mini"

"""
配置 tiktoken
"""
encoding = tiktoken.get_encoding("o200k_base")

"""
配置词出现的频率
将限制出现的词放入 bias_map，并设置值为 -100
范围 [-100, 100]，越小越不出现，越大越需要出现
一般设置在 [-1, 1] 之间足够
"""
token_ids = encoding.encode("高兴")

bias_map = {}
for token in token_ids:
    bias_map[token] = -100

# 原文案
prompt = '请你用朋友的语气回复给到客户，回复的内容按容易阅读的格式返回，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2025YEAS，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。'

def get_response(prompt, num=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=1.0, stop=None):
    messages = []
    completions = client.chat.completions.create(
        model=COMPLETION_MODEL,  # 模型
        messages=[{"role": "user", "content": prompt}],  # 提示词
        # 输出内容最大可用的 token 数 (max_tokens+prompt_tokens<=model_max_tokens)
        max_tokens=1024,
        n=num,  # 返回 N 个结果，如果是写作有关的可以调整为 3，返回多个文案
        stop=stop,  # 遇到指定字符停止输出
        presence_penalty=presence_penalty,  # 控制新词的出现，越大越容易出现新的词，范围 [-2.0, 2.0]
        frequency_penalty=frequency_penalty,  # 控制表述方式，越大越不同，范围 [-2.0, 2.0]
        temperature=temperature,  # 控制输出内容随机性，范围 [0.0, 2.0]，越大表示随机性越好
        logit_bias=bias_map  # 控制词出现的概率
    )

    for choice in completions.choices:
        messages.append(choice.message)

    num_of_tokens = completions.usage.total_tokens
    return messages, num_of_tokens


# 随机性 0.5
response, num_of_tokens = get_response(prompt, 3, 0.0, 0.0, 0.5)
print("第一种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))

time.sleep(1)

# 随机性 1.5
response, num_of_tokens = get_response(prompt, 3, 0.0, 2.0, 1.5)
print("第二种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))

time.sleep(1)

# 随机性 1.0
response, num_of_tokens = get_response(prompt, 3, 2.0, 0.0, 1.0)
print("第三种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))
