# coding=utf-8
from openai import OpenAI
from google.colab import userdata
import os

"""
根据提供的语料库，使用 AI Chat 分析，回答用户提出的问题

dependency packages
pip install openai
"""
client = OpenAI(
    api_key=userdata.get("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1"
)

COMPLETION_MODEL = "moonshot-v1-8k"

# 读取文件
def read_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return file.read()
    else:
        print("%s not exists\n" % file_path)
        return ''


# 语料库查询器类
class corpusSearcher:
    # 初始化
    def __init__(self, model, corpus):
        self.model = model
        self.corpus = corpus

    # 回答问题
    def answer(self, prompt):
        try:
            completions = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": "请你根据下面这些内容简短回答问题（不超过 80 字），如果内容中没有与问题相关的内容，请回答 '不知道'\n问题：" +
                        prompt + "\n\n" + self.corpus}
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


# 初始化语料库
text_corpus = read_file('./data/mr_fujino/mr_fujino.txt')
corpus_searcher = corpusSearcher(COMPLETION_MODEL, text_corpus)

# 执行查询
questions = [
    "鲁迅先生在日本学习医学的老师是谁？",
    "作者在离开仙台时，对藤野先生说了什么谎话？他为什么要说这个谎话？",
    "文章中提到了哪两处鲁迅在仙台期间“受到优待”或“被特殊照顾”的例子？",
    "日本爱国青年学生为什么要给作者写匿名信？匿名信的开头引用了谁的什么句子？",
    "是什么具体的事件促使作者做出了“不学医学”、离开仙台的决定？这一事件对他的思想产生了怎样的冲击？",
    "作者保存的藤野先生修改过的讲义最终去向如何？现在作者手头还留有什么关于藤野先生的纪念物？",
    "西游记中有什么主要的人物？"
]

for index, question in enumerate(questions):
    response = corpus_searcher.answer(question)
    print(f"问题{index + 1}: {question}")
    print(f"回答{index + 1}: {response}\n")
