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
        raise e

    return completion.choices[0].message.content


print(write_love_letter())


"""
亲爱的宝贝，

在这个宁静的夜晚，我坐在窗前，望着满天的繁星，心中充满了对你的思念。我想通过这封信，把我的爱意和温暖传递给你，让你感受到我对你的深情。

自从遇见你的那一刻起，我的生活就变得如此美好。你的笑容如同阳光，照亮了我心中的每一个角落。你的温柔和善良，让我深深地爱上了你。每当我想起你，心中就充满了无尽的甜蜜和幸福。

我想告诉你，你是我生命中最重要的人。我愿意为你付出一切，只为让你快乐。我愿意陪你走过每一个春夏秋冬，陪你看日出日落，陪你度过每一个快乐和困难的时刻。我想成为你生命中最坚实的依靠，让你在任何时候都能感受到我的温暖和支持。

我知道，爱情不仅仅是甜言蜜语和浪漫，它更需要我们共同努力和经营。我愿意和你一起面对生活中的挑战，一起成长，一起变得更加成熟和坚强。我相信，只要我们手牵手，心连心，就没有什么困难是我们不能克服的。

在这个特别的日子里，我想对你说：我爱你，宝贝。我愿意用我的一生去爱你，去呵护你，去珍惜你。请相信我，我会用我的行动证明我对你的爱，让你成为世界上最幸福的人。

愿我们的爱情如同这封信一样，永远流传，永远美丽。

永远爱你的，
[你的名字]
"""
