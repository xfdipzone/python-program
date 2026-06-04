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

"""
问题1: 鲁迅先生在日本学习医学的老师是谁？
回答1: 鲁迅先生在日本学习医学的老师是藤野严九郎。

问题2: 作者在离开仙台时，对藤野先生说了什么谎话？他为什么要说这个谎话？
回答2: 作者在离开仙台时对藤野先生说了一个谎话，他告诉藤野先生他想改学生物学，先生教给他的医学知识还有用。他之所以说这个谎话，是因为看到藤野先生有些凄然，想要安慰他。

问题3: 文章中提到了哪两处鲁迅在仙台期间“受到优待”或“被特殊照顾”的例子？
回答3: 文章中提到鲁迅在仙台期间“受到优待”或“被特殊照顾”的两个例子是：1. 学校不收他的学费；2. 几个职员为他的食宿操心。

问题4: 日本爱国青年学生为什么要给作者写匿名信？匿名信的开头引用了谁的什么句子？
回答4: 日本爱国青年学生给作者写匿名信是因为怀疑作者在解剖学试验中作弊，得到了藤野先生泄露的题目。匿名信的开头引用了《新约》上的句子“你改悔罢！”，这句话也被托尔斯泰引用过。

问题5: 是什么具体的事件促使作者做出了“不学医学”、离开仙台的决定？这一事件对他的思想产生了怎样的冲击？
回答5: 作者做出“不学医学”、离开仙台的决定是因为在观看枪毙中国人的电影时，周围的中国人欢呼“万岁”，这种欢呼让他感到刺耳，使他的思想发生了变化。这一事件对他的思想产生了深刻的冲击，让他对中国人的麻木和冷漠感到失望，从而决定放弃医学，离开仙台。

问题6: 作者保存的藤野先生修改过的讲义最终去向如何？现在作者手头还留有什么关于藤野先生的纪念物？
回答6: 作者保存的藤野先生修改过的讲义最终遗失了，现在作者手头留有藤野先生的一张照相作为纪念物。

问题7: 西游记中有什么主要的人物？
回答7: 不知道
"""
