# coding=utf-8
from openai import OpenAI
from google.colab import userdata
import time
import gradio as gr
import warnings

"""
游戏百科全书问答机器人
hugging face space: https://huggingface.co/spaces/fdizone/game-conversation-chatbot
hugging face app: https://fdizone-game-conversation-chatbot.hf.space/

dependency packages
pip install openai
pip install gradio
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

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
            print(e)
            return e

        message = completions.choices[0].message.content
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round * 2 + 1:
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
            print(e)
            return e

        summarized = completions.choices[0].message.content
        return summarized


# 屏蔽 DeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 系统提示词
prompt = """你是一个游戏百科全书，用中文回答游戏的问题。你的回答需要满足以下要求:
1. 你的回答必须是中文
2. 回答限制在100个字以内"""

# 初始化对话管理器
COMPLETION_MODEL = "moonshot-v1-8k"
conv = Conversation(COMPLETION_MODEL, prompt, 2)

# 定义提问按钮的方法
# 调用对话管理器，获取问题的回答
# 组装问题与回答到 history
def answer(question, history=[]):
    history.append({"role": "user", "content": question})
    response = conv.ask(question)
    history.append({"role": "assistant", "content": response})
    return history, ""


# 构建 Gradio 界面
with gr.Blocks(theme=gr.themes.Soft(), css="#chatbot{height:300px} .overflow-y-auto{height:500px}") as demo:
    gr.Markdown('<h1 style="text-align: center;">游戏百科聊天机器人</h1>')
    chatbot = gr.Chatbot(elem_id="chatbot", type="messages", allow_tags=False)
    state = gr.State([])

    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="请输入你的问题")

    txt.submit(answer, [txt, state], [chatbot, txt])

demo.launch(
    share=True,
    debug=True
)
