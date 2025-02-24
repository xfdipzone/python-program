# coding=utf-8
from openai import OpenAI
import os
import time
import logging
import tiktoken

"""
游戏百科全书问答机器人
"""
client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

"""
配置日志
在 Google CoLab 中使用时，必须设置 force=True
"""
logging.basicConfig(level=logging.INFO, filemode='a', filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s', force=True)

"""
配置 tiktoken
Model 使用的编码
参考 https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
"""
encoding = tiktoken.get_encoding("o200k_base")

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
            # 超过 num_of_round 次对话后，执行历史对话内容总结
            summarized = self.summarize()
            logging.info(summarized)

            # 重置 messages 内容为历史对话内容总结
            self.messages = [
                {"role": "system", "content": self.prompt + "\n\n" + summarized}
            ]

        return message, num_of_tokens

    # 总结历史对话内容
    def summarize(self):
        # 历史对话内容
        history = ""
        for message in self.messages:
            history = history + message["role"] + ": " + message["content"] + "\n"

        try:
            completions = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": history + "\n\n请用100字以内总结一下上面 User 和 Assistant 聊了些什么：\n"}
                ],
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.0
            )
        except Exception as e:
            print(e)
            return e

        summarized = completions.choices[0].message.content
        return summarized

prompt = """你是一个游戏百科全书，用中文回答游戏的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内"""
conv1 = Conversation(prompt, 2)

# 使用 tiktoken 统计 token 消耗数量
prompt_count = len(encoding.encode(prompt))
print("prompt : %s\nTikToken 统计 : prompt 消耗 %d token\n" % (prompt, prompt_count))

questions = [
    "你是谁？",
    "请问仙剑奇侠传这款游戏是什么年份的？",
    "那轩辕剑2外传枫之舞呢？",
    "那明星志愿3是什么类型的游戏？",
    "那大富翁4呢？",
    "古剑奇谭一共有多少部作品？"
]

questions_count = len(questions)

for index, question in enumerate(questions):
    print("用户问题 : %s" % question)
    answer,num_of_tokens = conv1.ask(question)
    print("AI 回答 : %s" % answer)
    print("消耗的 token 数量 : %d" % num_of_tokens)

    # 使用 tiktoken 统计 token 消耗数量
    question_count = len(encoding.encode(question))
    answer_count = len(encoding.encode(answer))
    total_count = question_count + answer_count
    print("TikToken统计 : 问题消耗 %d token，回答消耗 %d token，总共消耗 %d token\n" % (question_count, answer_count, total_count))

    # 每次对话间隔 20s，避免被限流
    if index < questions_count-1:
        time.sleep(20)
