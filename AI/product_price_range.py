# coding=utf-8
from openai import OpenAI
from google.colab import userdata

"""
AI 查询产品价格范围

dependency packages
pip install openai
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

prompt = """
Consideration product : 轩辕剑三外传天之痕steam

1. Compose human readable product title used on Amazon in english within 20 words.
2. Write 5 selling points for the products in Amazon.
3. Evaluate a price range for this product in U.S.

Output the result in json format with three properties called title, selling_points and price_range, use chinese language.
"""

def get_response(prompt):
    try:
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
    except Exception as e:
        print(e)
        return e


print(get_response(prompt))


"""
{
  "title": "轩辕剑三外传：天之痕 - 经典角色扮演游戏",
  "selling_points": [
    "1. 深受玩家喜爱的经典角色扮演游戏，丰富的剧情和角色发展。",
    "2. 独特的中国风画面和音乐，带来沉浸式的游戏体验。",
    "3. 精心设计的战斗系统，策略与技巧的完美结合。",
    "4. 探索广阔的游戏世界，解锁隐藏的秘密和支线任务。",
    "5. 支持Steam平台，享受便捷的游戏管理和更新服务。"
  ],
  "price_range": "20美元至40美元"
}
"""
