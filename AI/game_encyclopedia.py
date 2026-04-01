# coding=utf-8
from openai import OpenAI
from google.colab import userdata
import os
import time
import logging
import tiktoken

"""
游戏百科全书问答机器人

dependency packages
pip install openai
pip install tiktoken
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

"""
配置日志
在 Google CoLab 中使用时，必须设置 force=True
"""
logging.basicConfig(level=logging.INFO, filemode='a', filename='app.log',
                    format='%(asctime)s - %(levelname)s - %(message)s', force=True)

"""
配置 tiktoken
Model 使用的编码
参考 https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
"""
encoding = tiktoken.get_encoding("cl100k_base")

# 对话管理器类
class Conversation:
    # 初始化模型，设置系统提示词和保存的对话轮数限制
    def __init__(self, model, prompt, num_of_round):
        self.model = model
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": self.prompt})

    # 调用 AI 回答问题，将问题与回答追加到 messages 中保存
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
            raise e

        message = completions.choices[0].message.content
        num_of_tokens = completions.usage.total_tokens
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round * 2 + 1:
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
            history = history + message["role"] + \
                ": " + message["content"] + "\n"

        try:
            completions = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": history +
                        "\n\n请用100字以内总结一下上面 User 和 Assistant 聊了些什么：\n"}
                ],
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.0
            )
        except Exception as e:
            raise e

        summarized = completions.choices[0].message.content
        return summarized


# 系统提示词
prompt = """你是一个游戏百科全书，用中文回答游戏的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内"""

# 初始化对话管理器
COMPLETION_MODEL = "moonshot-v1-8k"
conv = Conversation(COMPLETION_MODEL, prompt, 2)

# 使用 tiktoken 统计 token 消耗数量
prompt_count = len(encoding.encode(prompt))
print("prompt : %s\nTikToken 统计 : prompt 消耗 %d token\n" %
      (prompt, prompt_count))

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
    answer, num_of_tokens = conv.ask(question)
    print("AI 回答 : %s" % answer)
    print("消耗的 token 数量 : %d" % num_of_tokens)

    # 使用 tiktoken 统计 token 消耗数量
    question_count = len(encoding.encode(question))
    answer_count = len(encoding.encode(answer))
    total_count = question_count + answer_count
    print("TikToken统计 : 问题消耗 %d token，回答消耗 %d token，总共消耗 %d token\n" %
          (question_count, answer_count, total_count))

    # 每次对话间隔 1s，避免被限流
    if index < questions_count - 1:
        time.sleep(1)


"""
prompt : 你是一个游戏百科全书，用中文回答游戏的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内
TikToken 统计 : prompt 消耗 67 token

用户问题 : 你是谁？
AI 回答 : 你好！我是你的游戏百科助手，随时准备回答你的游戏相关问题。
消耗的 token 数量 : 66
TikToken统计 : 问题消耗 5 token，回答消耗 34 token，总共消耗 39 token

用户问题 : 请问仙剑奇侠传这款游戏是什么年份的？
AI 回答 : 仙剑奇侠传最初发行于1995年。
消耗的 token 数量 : 95
TikToken统计 : 问题消耗 24 token，回答消耗 18 token，总共消耗 42 token

用户问题 : 那轩辕剑2外传枫之舞呢？
AI 回答 : 轩辕剑2外传枫之舞发行于1998年。
消耗的 token 数量 : 128
TikToken统计 : 问题消耗 19 token，回答消耗 21 token，总共消耗 40 token

用户问题 : 那明星志愿3是什么类型的游戏？
AI 回答 : 明星志愿3是一款模拟养成类游戏。
消耗的 token 数量 : 99
TikToken统计 : 问题消耗 18 token，回答消耗 20 token，总共消耗 38 token

用户问题 : 那大富翁4呢？
AI 回答 : 大富翁4是一款经典的棋盘式的策略经营游戏。
消耗的 token 数量 : 125
TikToken统计 : 问题消耗 12 token，回答消耗 31 token，总共消耗 43 token

用户问题 : 古剑奇谭一共有多少部作品？
AI 回答 : 古剑奇谭系列目前共有四部作品。
消耗的 token 数量 : 153
TikToken统计 : 问题消耗 17 token，回答消耗 19 token，总共消耗 36 token
"""
