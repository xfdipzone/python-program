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

COMPLETION_MODEL = "kimi-k2-thinking"

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

# 系统提示词
system_prompt = """你是一个文字内容优化的工具，可以根据用户给出的文字进行优化。
被优化后的内容中的词汇会有一些控制规则
需要被控制的词汇都会有一个值，值的范围是 [-100, 100]，其中 -100 表示一定不能出现在内容中，100 表示一定要出现在内容中
这个值越小，越小概率出现，越大，越多概率出现。
"""

# 根据提示词生成客服回复内容，并返回消耗的 token 数量
def get_response(prompt, words, num=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=1.0, stop=None):
    # 词汇控制
    words_prompt = "被控制的词汇列表如下：\n"
    for word, bias_value in words.items():
        words_prompt += f"{word} : {bias_value}\n"

    responses = []
    messages = [
        {"role": "system", "content": system_prompt + words_prompt},
        {"role": "user", "content": prompt}
    ]

    try:
        completions = client.chat.completions.create(
            # 模型
            model=COMPLETION_MODEL,

            # 提示词
            messages=messages,

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
    except Exception as e:
        raise e

    for choice in completions.choices:
        responses.append(choice.message)

    num_of_tokens = completions.usage.total_tokens
    return responses, num_of_tokens


# 随机性 0.3
response, num_of_tokens = get_response(prompt, words, 3, 0.0, 0.0, 0.3)
print("第一种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))

print_line('=')
time.sleep(1)

# 随机性 1.0
response, num_of_tokens = get_response(prompt, words, 3, 0.0, 2.0, 1.0)
print("第二种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))

print_line('=')
time.sleep(1)

# 随机性 0.5
response, num_of_tokens = get_response(prompt, words, 3, 2.0, 0.0, 0.5)
print("第三种参数配置：(消耗的 token 数量：%d)\n" % num_of_tokens)
for index, answer in enumerate(response):
    print("version %d: %s\n\n" % (index + 1, answer.content))


"""
第一种参数配置：(消耗的 token 数量：1914)

version 1: 亲，你好呀！

你的订单已经发货啦，正在路上飞奔向你呢~ ✨

**订单号：** 2025YEAS
**预计送达：** 3天内

最近天气不太好，路上耽搁了一些时间，比预计的晚了一点，真是不好意思呀。不过包裹已经顺利出发，很快就会到你手上啦！

感谢你的支持和耐心等待，希望你喜欢我们的商品！有问题随时找我哦~


version 2: 亲～

告诉您一个好消息，您的订单已经发货在路上了哦！订单号是 **2025YEAS**。

正常情况下，大概3天左右就能送到您那儿啦。不过最近天气不太好，可能会让快递小哥稍微慢一点点，比预计时间稍长一些，还请您多多包涵呀！

真的特别感谢您选购我们的商品，希望您会喜欢！

有问题随时找我哦～


version 3: 亲，你的包裹已经飞奔在路上啦！✨

订单号：2025YEAS

预计3天内就能送到你手上～不过最近天气不太好，路上比原计划慢了一些，不好意思呀！

感谢你的选择和信任，希望你会喜欢！有任何问题随时找我哦～


================================================================================

第二种参数配置：(消耗的 token 数量：1477)

version 1: 嘿，亲爱的！😊

你的宝贝已经在飞奔来找你啦～订单号是 **#2024YEAS**，请查收！

因为最近天气不太好快递小哥路上耽搁了点儿时间不过别担心正常情况下再过个两三天你就能亲手接到它你们那儿应该也快能看到了谢谢你选了我们家店哦期待收到后你的好评！

一路顺风~~


version 2: Hi！亲～先跟你说个情况哦 ☺️

你的包裹今天已经嗖嗖地出发啦！虽然最近天公不作美有点儿小状况(刮风下雨的)，但不用担心～我特地帮你盯着让仓库优先发出了呢！正常情况快的话明天，慢点儿也就3天内能到你那儿咯✨

这是你可以随时查询的小单号：**2025YEAS**，有任何问题都能找我帮忙查哟！

真的很感谢你选择我们家东西 ❤️
希望你收到时会喜欢💕


version 3: 亲亲😊

你的包裹已经飞奔在路上啦！单号是：**2025YEAS**

最近天气确实不太给力，让这趟路程比预计多了点时间⏰~不过放心哈，再有个**2-3天**稳稳就能到你手上！

真的特别感谢你选择我们家宝贝🙏🏻这一路的风雨无阻只为奔向你✨


================================================================================

第三种参数配置：(消耗的 token 数量：1610)

version 1: 亲，你好呀！

你的包裹已经飞奔在路上啦～订单号是 **2025YEAS**，记得查收哦！

不过最近天气有点调皮，路上可能会耽搁点时间，大概**3天内**能送到你手上。让你久等了，真的不好意思呀！

谢谢你这么有眼光选了我们家的东西，希望你会喜欢！

有问题随时找我聊～


version 2: 亲，
你的订单已经发货啦！订单号是 **2025YEAS**，目前正在火速赶往你那里哦～

因为最近天气不太好，运输时间比预计的长了一点点，还请多多包涵。不过别担心，宝贝预计在**3天内**就能送到你手上啦！

非常感谢你选择我们的商品，希望你会喜欢！有什么疑问随时找我哈～


version 3: 亲，您的订单已经发货啦！

**订单号**：2025YEAS
**预计送达**：3天内送到您手上

包裹正在快马加鞭赶往您那儿，不过最近天气不太给力，路上比平时慢了一些，还望理解哈～

感谢您选购我们的宝贝，希望您会喜欢！
"""
