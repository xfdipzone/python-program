# coding=utf-8
from openai import OpenAI
import os

"""
AI 文本插入器

dependency packages
pip install openai
"""
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# 提示词
prompt = """你是一个文本插入器，根据前文与后文的内容，补充中间的文本内容。
你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内"""

# 前文
prefix = """在这个快节奏的现代社会中，我们每个人都面临着各种各样的挑战和困难。
在这些挑战和困难中，有些是由外部因素引起的，例如经济萧条、全球变暖和自然灾害等。"""

# 后文
suffix = """面对这些挑战和困难，我们需要采取积极的态度和行动来克服它们。
这意味着我们必须具备坚韧不拔的意志和创造性思维，以及寻求外部支持的能力。
只有这样，我们才能真正地实现自己的潜力并取得成功。"""

"""
文本插入器
根据前文与后文，补充中间的内容
"""
def insert_text(prefix, suffix):
    completions = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": prefix},
            {"role": "user", "content": suffix}
        ],
        max_tokens=1024,
        n=1,
        stop=None
    )

    return completions.choices[0].message.content


print("前文：\n%s\n" % prefix)
print("AI 插入的文本：\n%s\n" % insert_text(prefix, suffix))
print("后文：\n%s" % suffix)
