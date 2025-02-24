# coding=utf-8
from openai import OpenAI
import os
import time
import gradio as gr

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
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round*2 + 1:
            # 超过 num_of_round 次对话后，执行历史对话内容总结
            summarized = self.summarize()

            # 重置 messages 内容为历史对话内容总结
            self.messages = [
                {"role": "system", "content": self.prompt + "\n\n" + summarized}
            ]

        return message

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
conv = Conversation(prompt, 2)

def answer(question, history=[]):
    history.append(question)
    response = conv.ask(question)
    history.append(response)
    responses = [(u,b) for u,b in zip(history[::2], history[1::2])]
    return responses, history, ""

with gr.Blocks(theme=gr.themes.Soft(), css="#chatbot{height:300px} .overflow-y-auto{height:500px}") as demo:
    gr.Markdown('<h1 style="text-align: center;">游戏百科聊天机器人</h1>')
    chatbot = gr.Chatbot(elem_id="chatbot")
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="请输入你的问题")

    txt.submit(answer, [txt, state], [chatbot, state, txt])

demo.launch()
