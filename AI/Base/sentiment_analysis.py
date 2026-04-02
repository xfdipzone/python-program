# coding=utf-8
from openai import OpenAI
from google.colab import userdata

"""
AI 情感分析评论

dependency packages
pip install openai
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

# 提示词，给出正面与负面例子
prompts = """
判断一下用户的评论情感上是正面的还是负面的
评论：买的银色版真的很好看，一天就到了，晚上就开始拿起来完系统很丝滑流畅，做工扎实，手感细腻，很精致哦苹果一如既往的好品质
情感：正面

评论：随意降价，不予价保，服务态度差
情感：负面
"""

def get_response(prompt):
    try:
        completions = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.0
        )
    except Exception as e:
        raise e

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

print(get_response(good_case) + "\n")

# 负面的评论
bad_case = prompts + """
评论：信号不好电池也不耐电不推荐购买
情感：
"""

print(get_response(bad_case) + "\n")

# 正面的评论
good_restaurant = prompts + """
评论：这家餐馆太好吃了，一点都不糟糕
情感：
"""

print(get_response(good_restaurant) + "\n")

# 负面的评论
bad_restaurant = prompts + """
评论：这家餐馆太糟糕了，一点都不好吃
情感：
"""

print(get_response(bad_restaurant) + "\n")


"""
情感：正面

这个评论中，用户对苹果产品的外形外观、拍照效果、运行速度等方面都给予了积极的评价，并且提到了“苹果审美一直很好”、“4800万像素真的是没的说，太好了”、“苹果的反应速度好，用上三五年也不会卡顿的”等正面表述。同时，用户还特别提到了“14pro的磨砂金真的太好看了”，表达了对产品外观的喜爱。整体来看，这个评论的情感倾向是正面的。

情感：负面

这个评论提到了信号不好和电池不耐用的问题，并且明确表示不推荐购买，所以整体情感是负面的。

情感：正面

分析：
1. 评论中提到“这家餐馆太好吃了”，表示用户对餐馆的食物质量感到满意，这是一个正面的评价。
2. 评论中提到“一点都不糟糕”，表示用户认为餐馆没有任何不好的地方，这也是一个正面的评价。
综合以上两点，可以判断这条评论的情感倾向是正面的。

情感：负面

这个评论中使用了“太糟糕了”和“一点都不好吃”这样的负面词汇，表达了用户对这家餐馆的不满和失望，所以情感倾向是负面的。
"""
