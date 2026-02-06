# coding=utf-8
from openai import OpenAI
from google.colab import userdata

"""
AI 写情信

dependency packages
pip install openai
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

completion = client.chat.completions.create(
    model=COMPLETION_MODEL,
    messages=[
        {"role": "user", "content": "write a lover letter about ai, use chinese language."}
    ]
)

print(completion.choices[0].message.content)
