# coding=utf-8
from openai import OpenAI
import os

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],
)

completion = client.chat.completions.create(
  model="gpt-4o-mini",
  store=False, # true: 保存结果 false: 不保存结果
  messages=[
    {"role": "user", "content": "write a lover letter about ai, use chinese language."}
  ]
)

print(completion.choices[0].message)
