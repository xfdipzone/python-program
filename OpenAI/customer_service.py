# coding=utf-8
from openai import OpenAI
import os
import time

client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

COMPLETION_MODEL = "gpt-4o-mini"

prompt = '请你用朋友的语气回复给到客户，并称他为“亲”，他的订单已经发货在路上了，预计在3天之内会送达，订单号2025YEAS，我们很抱歉因为天气的原因物流时间比原来长，感谢他选购我们的商品。'

def get_response(prompt, temperature = 1.0, stop=None):
    completions = client.chat.completions.create(
        model=COMPLETION_MODEL, # 模型
        messages=[{"role": "user", "content": prompt}], # 提示词
        max_tokens=1024, # 输出内容最大可用的 token 数 (max_tokens+prompt_token<=model_max_tokens)
        n=1, # 返回 1 个结果，如果是写作有关的可以调整为 3，返回多个文案
        stop=stop, # 遇到指定字符停止输出
        temperature=temperature, # 控制输出内容随机性，范围 0 ~ 2.0，越大表示随机性越好
    )
    message = completions.choices[0].message
    return message.content

# 随机性 0.5
print("第一种回答：" + get_response(prompt, 0.5))
time.sleep(1)

# 随机性 1.0
print("第二种回答：" + get_response(prompt, 1.0))
time.sleep(1)

# 随机性 0.0
print("第三种回答：" + get_response(prompt, 0.0))
