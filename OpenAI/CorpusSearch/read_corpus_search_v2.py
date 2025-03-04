# coding=utf-8
from openai import OpenAI
import os

"""
根据提供的语料库，回答用户提出的问题
调用 client.chat.completions.create 接口实现
"""
client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

"""
读取文件
"""
def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        print("%s not exists\n" % file_path)
        return ''

"""
语料库
"""
class Corpus:
    def __init__(self, corpus):
        self.model = "gpt-4o-mini"
        self.corpus = corpus

    def answer(self, prompt):
        try:
            completions = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "请你根据下面这些内容回答问题，如果内容中没有与问题相关的内容，请回答 '不知道'\n问题：" + prompt + "\n\n" + self.corpus}
                ],
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.5
            )
        except Exception as e:
            print(e)
            return e

        message = completions.choices[0].message
        return message.content

content = read_file('data/mr_fujino/mr_fujino.txt')
corpus = Corpus(content)

questions = [
    "请问鲁迅先生在日本学习医学的老师是谁？",
    "鲁迅先生去哪里学的医学？",
    "西游记中有什么主要的人物？"
]

for index, question in enumerate(questions):
    response = corpus.answer(question)
    print("问题 %d: %s\n回答：%s\n" % (index+1, question, response))
