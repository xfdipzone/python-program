# coding=utf-8
from openai import OpenAI
import os

"""
AI 情感分析评论

dependency packages
pip install openai
"""
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

COMPLETION_MODEL = "gpt-4o-mini"

# 提示词，给出正面与负面例子
prompts = """
判断一下用户的评论情感上是正面的还是负面的
评论：买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质
情感：正面

评论：随意降价，不予价保，服务态度差
情感：负面
"""

def get_response(prompt):
    completions = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.0
    )
    message = completions.choices[0].message
    return message.content


# 正面的评论
good_case = prompts + """
评论：外形外观：苹果审美一直很好，金色非常漂亮
拍照效果：14pro升级的4800万像素真的是没的说，太好了，
运行速度：苹果的反应速度好，用上三五年也不会卡顿的，之前的7P用到现在也不卡
其他特色：14pro的磨砂金真的太好看了，不太高调，也不至于没有特点，非常耐看，很好的
情感：
"""

print(get_response(good_case))

# 负面的评论
bad_case = prompts + """
评论：信号不好电池也不耐电不推荐购买
情感：
"""

print(get_response(bad_case))

# 正面的评论
good_restaurant = prompts + """
评论：这家餐馆太好吃了，一点都不糟糕
情感：
"""

print(get_response(good_restaurant))

# 负面的评论
bad_restaurant = prompts + """
评论：这家餐馆太糟糕了，一点都不好吃
情感：
"""

print(get_response(bad_restaurant))
