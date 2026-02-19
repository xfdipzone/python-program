# coding=utf-8
from openai import OpenAI
from google.colab import userdata
import time
import tiktoken
import shutil

"""
AI 客服

dependency packages
pip install openai
pip install tiktoken
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

"""
配置 tiktoken
设置分词使用的编码
"""
encoding = tiktoken.get_encoding("o200k_base")

"""
配置词出现的频率
将限制出现的词放入 bias_map，并设置值为 -100
范围 [-100, 100]，越小越不出现，越大越需要出现
一般设置在 [-1, 1] 之间足够
"""
words = {
    "高兴": -100,
    "抱歉": -5,
    "物流": -100
}

# 获取限制词配置
def get_bias_map(words):
    bias_map = {}
    for word, bias_value in words.items():
        token_ids = encoding.encode(word)
        for token in token_ids:
            bias_map[token] = bias_value
    return bias_map


bias_map = get_bias_map(words)

# 输出一条分隔线
def print_line(char='─', width=None):
    """输出一条分隔线，默认字符 ─，默认宽度=终端列宽"""
    if width is None:
        width = shutil.get_terminal_size().columns
    print(char * width + '\n')


# 原文案
prompt = "请你用朋友的语气回复给到客户，回复的内容按容易阅读的格式返回，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2025YEAS，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。"

# 根据提示词生成客服回复内容
def get_response(prompt, num=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=1.0, stop=None):
    responses = []

    completions = client.chat.completions.create(
        # 模型
        model=COMPLETION_MODEL,

        # 提示词
        messages=[{"role": "user", "content": prompt}],

        # 输出内容最大可用的 token 数 (max_tokens+prompt_tokens<=model_max_tokens)
        max_tokens=1024,

        # 返回 N 个结果，如果是写作有关的可以调整为 3，返回多个文案
        n=num,

        # 遇到指定字符停止输出
        stop=stop,

        # 控制新词的出现，越大越容易出现新的词，范围 [-2.0, 2.0]
        presence_penalty=presence_penalty,

        # 控制表述方式，越大越不同，范围 [-2.0, 2.0]
        frequency_penalty=frequency_penalty,

        # 控制输出内容随机性，范围 [0.0, 2.0]，越大表示随机性越好（Kimi 范围 [0.0, 1.0]）
        temperature=temperature,

        # 控制词出现的概率（Kimi 不支持此参数）
        # logit_bias=bias_map
    )

    for choice in completions.choices:
        responses.append(choice.message)

    num_of_tokens = completions.usage.total_tokens
    return responses, num_of_tokens


# 随机性 0.3
response, num_of_tokens = get_response(prompt, 3, 0.0, 0.0, 0.3)
print("第一种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))

print_line('=')
time.sleep(1)

# 随机性 1.0
response, num_of_tokens = get_response(prompt, 3, 0.0, 2.0, 1.0)
print("第二种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))

print_line('=')
time.sleep(1)

# 随机性 0.5
response, num_of_tokens = get_response(prompt, 3, 2.0, 0.0, 0.5)
print("第三种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))
