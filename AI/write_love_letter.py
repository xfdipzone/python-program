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

# 调用 AI 写一封情信
def write_love_letter():
    try:
        completion = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[
                {"role": "user", "content": "write a lover letter, use chinese language."}
            ]
        )
    except Exception as e:
        print(e)
        return e

    return completion.choices[0].message.content


print(write_love_letter())

"""
亲爱的宝贝，

在这个宁静的夜晚，我坐在窗前，望着满天的繁星，心中充满了对你的思念。我想通过这封信，把我的爱意和温暖传递给你，让你感受到我对你的深情。

自从遇见你的那一刻起，我的世界就变得如此美好。你的笑容如同阳光，温暖而明媚，照亮了我心中的每一个角落。你的眼睛，如同深邃的星空，让我沉醉其中，无法自拔。你的每一个动作，每一句话语，都让我为之心动。

我想告诉你，我爱你，不仅仅是因为你的美丽和才华，更因为你的善良和真诚。你总是那么关心别人，总是那么乐于助人，这让我感觉到了你内心的温柔和善良。你的真诚让我感到安心，让我愿意把心交给你，与你共度一生。

我们在一起的时光总是那么短暂，却又那么珍贵。每一次的相聚，都让我更加珍惜我们之间的感情。我想和你一起走过四季，一起看日出日落，一起感受生活的点点滴滴。我想在你难过的时候，给你一个温暖的拥抱；在你快乐的时候，与你一起分享喜悦。

我知道，爱情并不是一帆风顺的，我们也会面临困难和挑战。但是，我相信，只要我们手牵手，心连心，没有什么是我们不能克服的。我愿意为你付出一切，只为让你幸福快乐。

在这个特别的日子里，我想对你说：我爱你，永远爱你。愿我们的爱情如同这封信一样，永远流传，永不褪色。

期待你的回信，期待我们的未来。

永远爱你的，

[你的名字]
"""
