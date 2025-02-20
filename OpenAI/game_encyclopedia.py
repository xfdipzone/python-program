# coding=utf-8
from openai import OpenAI
import os
import time

"""
游戏百科全书问答机器人
"""
client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

class Conversation:
    def __init__(self, prompt, num_of_round):
        self.model = "gpt-4o-mini"
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            completions = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.5
            )
        except Exception as e:
            print(e)
            return e

        message = completions.choices[0].message.content
        num_of_tokens = completions.usage.total_tokens
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round*2 + 1:
            del self.messages[1:3] # 只保留最后 num_of_round 次对话

        return message, num_of_tokens

prompt = """你是一个游戏百科全书，用中文回答游戏的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内"""
conv1 = Conversation(prompt, 2)

questions = [
    "你是谁？",
    "请问仙剑奇侠传这款游戏是什么年份的？",
    "那轩辕剑2外传枫之舞呢？",
    "那明星志愿3是什么类型的游戏？",
    "那大富翁4呢？"
]

for question in questions:
    print("用户问题 : %s" % question)
    answer,num_of_tokens = conv1.ask(question)
    print("AI 回答 : %s" % answer)
    print("消费的 token 数量 : %d\n" % num_of_tokens)
    time.sleep(1)
