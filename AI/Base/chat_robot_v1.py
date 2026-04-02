# coding=utf-8
from openai import OpenAI
from google.colab import userdata

"""
AI 聊天机器人（V1）

dependency packages
pip install openai
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

print("你好，我是一个聊天机器人，请你提出你的问题吧！")

# 记录这次对话的所有问题与回答（上下文）
questions = []
answers = []

# 根据这次对话的问题与回答（上下文），创建提示词
def generate_prompt():
    prompt = ''
    nums = len(answers)
    # 已回答的问题与答案（上下文）
    for i in range(nums):
        prompt += "\n Q:" + questions[i]
        prompt += "\n A:" + answers[i]

    # 这次提问的问题
    prompt += "\n Q:" + questions[nums] + "\n A:"

    return prompt

# 向 AI 提问
def ask(prompt, stop=None):
    try:
        completions = client.chat.completions.create(
            model=COMPLETION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            n=1,
            stop=stop,
            temperature=0.5,
        )
    except Exception as e:
        raise e

    message = completions.choices[0].message
    return message.content

# 创建聊天机器人
def chat_robot():
    while True:
        user_input = input("请输入你的问题：> ")
        questions.append(user_input)

        # 判断是否结束对话
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("GoodBye!")
            break

        # 根据上下文创建 prompt
        prompt = generate_prompt()

        # 向 AI 提问
        answer = ask(prompt)

        print(answer + "\n")

        # 保存回答
        answers.append(answer)


# 启动聊天机器人
chat_robot()


"""
你好，我是一个聊天机器人，请你提出你的问题吧！
请输入你的问题：> 你是谁？
你好！我是一个人工智能助手，擅长中英文对话。我在这里帮助你解答问题、提供信息。需要我的协助吗？

请输入你的问题：> 如何制作白切鸡？
制作白切鸡是一道简单但需要技巧的菜肴，以下是基本的步骤：

1. **准备材料**：一只新鲜的鸡（建议选择肉质较嫩的品种，如三黄鸡），姜、葱、料酒、盐等调料。

2. **处理鸡肉**：将鸡清洗干净，去除内脏和多余的脂肪。可以在鸡的内外涂抹一些盐，帮助入味。

3. **煮水**：在锅中加入足够的水，放入姜片、葱段和料酒，大火煮沸。

4. **烫鸡**：将鸡放入沸水中，用勺子不断地将热水浇在鸡身上，使其均匀受热。这个过程大约需要3-5分钟。

5. **煮鸡**：将鸡完全浸入水中，转小火，保持水微开不翻滚的状态，煮大约20-30分钟（具体时间根据鸡的大小调整）。煮的过程中，可以用筷子插入鸡腿最厚的部位，如果没有血水流出，说明鸡已经熟了。

6. **冷却**：将煮好的鸡迅速放入冰水中冷却，这样可以使鸡肉更加紧实，皮更加脆嫩。

7. **切块**：将冷却后的鸡切块，摆放在盘子中。

8. **调味**：制作蘸料，可以是简单的姜葱酱油，也可以根据个人口味加入其他调料。

9. **享用**：将切好的鸡块与蘸料一起上桌，即可享用。

白切鸡的关键在于火候的控制和鸡肉的冷却过程，这两者都会影响最终的口感。希望这些步骤能帮助你制作出美味的白切鸡！

请输入你的问题：>  蚝油牛肉呢？
制作蚝油牛肉是一道简单快捷的家常菜，以下是基本的步骤：

1. **准备材料**：牛肉（最好选择里脊肉或者牛柳，因为这些部位的肉质比较嫩），蚝油，生抽，老抽，糖，淀粉，姜，蒜，葱，胡椒粉，料酒，盐。

2. **处理牛肉**：将牛肉切成薄片或者条状，放入碗中，加入料酒、生抽、胡椒粉、盐和少量淀粉，用手抓匀，腌制10-15分钟。

3. **准备调料**：在一个小碗中，加入蚝油、生抽、老抽（上色用）、糖和适量的水，搅拌均匀，备用。

4. **热锅凉油**：在锅中加入适量的油，油热后先放入姜片和蒜末爆香。

5. **炒牛肉**：将腌制好的牛肉倒入锅中，快速翻炒至牛肉变色，大约7-8成熟。

6. **加入调料**：倒入之前调好的蚝油汁，快速翻炒均匀，让牛肉片均匀裹上调料。

7. **收汁**：调至中小火，盖上锅盖，让牛肉在蚝油汁中焖煮1-2分钟，让牛肉更加入味。

8. **出锅前**：撒上葱花，翻炒均匀后即可出锅。

9. **装盘**：将炒好的蚝油牛肉装盘，即可上桌享用。

蚝油牛肉的关键在于牛肉的腌制和快速翻炒，这样可以保持牛肉的嫩滑口感。希望这些步骤能帮助你制作出美味的蚝油牛肉！

请输入你的问题：> bye
GoodBye!
"""
