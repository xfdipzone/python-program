# coding=utf-8
from openai import OpenAI
from google.colab import userdata

"""
AI 聊天机器人（V1）

dependency packages
pip install openai
"""
client = OpenAI(api_key=userdata.get("OPENAI_API_KEY"))

COMPLETION_MODEL = "gpt-4o-mini"

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
        print(e)
        return e

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
