# coding=utf-8
from openai import OpenAI
from google.colab import userdata

"""
AI 查询产品价格范围

dependency packages
pip install openai
"""
client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

COMPLETION_MODEL = "gpt-4o-mini"

prompt = """
Consideration product : 轩辕剑三外传天之痕steam

1. Compose human readable product title used on Amazon in english within 20 words.
2. Write 5 selling points for the products in Amazon.
3. Evaluate a price range for this product in U.S.

Output the result in json format with three properties called title, selling_points and price_range, use chinese language.
"""

def get_response(prompt):
    completions = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        n=1,
        stop=None,
        temperature=0.0,
    )
    message = completions.choices[0].message
    return message.content


print(get_response(prompt))
