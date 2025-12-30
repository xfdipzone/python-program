# coding=utf-8
from openai import OpenAI
from google.colab import userdata

"""
AI 写情信

dependency packages
pip install openai
"""
client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

COMPLETION_MODEL = "gpt-4o-mini"

completion = client.chat.completions.create(
    model=COMPLETION_MODEL,
    store=False,  # true: 保存结果 false: 不保存结果
    messages=[
        {"role": "user", "content": "write a lover letter about ai, use chinese language."}
    ]
)

print(completion.choices[0].message.content)
