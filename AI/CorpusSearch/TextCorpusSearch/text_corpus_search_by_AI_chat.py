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
    api_key=userdata.get("API_KEY"),
    base_url=userdata.get("AI_BASE_URL")
)

COMPLETION_MODEL = userdata.get("AI_DEFAULT_MODEL")

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
                temperature=0.5,
                extra_body={"thinking": {"type": "disabled"}}
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

"""
问题1: 鲁迅先生在日本学习医学的老师是谁？
回答1: 藤野先生（藤野严九郎）。

问题2: 作者在离开仙台时，对藤野先生说了什么谎话？他为什么要说这个谎话？
回答2: 谎称去学生物学；为安慰藤野先生，因见他凄然。

问题3: 文章中提到了哪两处鲁迅在仙台期间“受到优待”或“被特殊照顾”的例子？
回答3: 不知道

问题4: 日本爱国青年学生为什么要给作者写匿名信？匿名信的开头引用了谁的什么句子？
回答4: 日本爱国青年学生因怀疑作者成绩是藤野先生泄露题目所得，给作者写匿名信。匿名信开头引用了《新约》中“你改悔罢！”的句子（经托尔斯泰引用）。

问题5: 是什么具体的事件促使作者做出了“不学医学”、离开仙台的决定？这一事件对他的思想产生了怎样的冲击？
回答5: 事件是作者在讲堂看时事影片时，目睹中国人围观枪毙中国人的场景，日本学生拍掌欢呼“万岁”。这使他深感屈辱，认识到医学救不了麻木的国民，思想受到强烈冲击，决定弃医从文。

问题6: 作者保存的藤野先生修改过的讲义最终去向如何？现在作者手头还留有什么关于藤野先生的纪念物？
回答6: 讲义在七年前迁居时遗失；现在作者手头还留有藤野先生的照相。

问题7: 西游记中有什么主要的人物？
回答7: 不知道
"""
