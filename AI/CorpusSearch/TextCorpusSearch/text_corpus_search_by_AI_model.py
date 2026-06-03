# coding=utf-8
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from google.colab import userdata
from huggingface_hub import login
import os

"""
根据提供的语料库，使用 AI 大语言模型分析，回答用户提出的问题

dependency packages
pip install llama-index
pip install llama-index-embeddings-huggingface
pip install llama-index-readers-file
pip install llama-index-llms-openai-like
"""
# Login HuggingFace Hub
login(token=userdata.get("HF_TOKEN"))

# 加载 Hugging Face 的嵌入模型
embed_model = HuggingFaceEmbedding(
    model_name="google/embeddinggemma-300M",
    embed_batch_size=32
)

# 初始化大语言模型
llm = OpenAILike(
    model="moonshot-v1-8k",
    api_key=userdata.get("KIMI_API_KEY"),
    api_base="https://api.moonshot.cn/v1",
    is_chat_model=True,
    context_window=8192,  # 显式声明上下文窗口，省去手动覆盖属性的麻烦
    is_function_calling_model=False
)

# 配置全局的大语言模型（LLM）
Settings.llm = llm
Settings.embed_model = embed_model

# 源数据目录
source_dir = "./data/mr_fujino"

# 索引目录
index_dir = "./data/index_mr_fujino"

# 判断索引是否存在
if not os.path.exists(index_dir):
    print("未检测到本地索引，开始构建...")

    # 加载数据并构建索引
    documents = SimpleDirectoryReader(source_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)

    # 持久化索引到磁盘
    index.storage_context.persist(persist_dir=index_dir)
else:
    print("检测到本地索引，直接读取...")

    # 从磁盘中读取索引
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    index = load_index_from_storage(storage_context)

# 查询引擎
query_engine = index.as_query_engine()

# 执行查询
questions = [
    "鲁迅先生在日本学习医学的老师是谁？",
    "作者在离开仙台时，对藤野先生说了什么谎话？他为什么要说这个谎话？",
    "文章中提到了哪两处鲁迅在仙台期间“受到优待”或“被特殊照顾”的例子？",
    "日本爱国青年学生为什么要给作者写匿名信？匿名信的开头引用了谁的什么句子？",
    "是什么具体的事件促使作者做出了“不学医学”、离开仙台的决定？这一事件对他的思想产生了怎样的冲击？",
    "作者保存的藤野先生修改过的讲义最终去向如何？现在作者手头还留有什么关于藤野先生的纪念物？"
]

for index, question in enumerate(questions):
    # 优化问题，增加 prompt 说明
    prompted_question = f"{question} （请用一段话简短回答，不超过 80 字）"

    response = query_engine.query(prompted_question)
    print(f"问题{index + 1}: {question}")
    print(f"回答{index + 1}: {response}\n")

"""
问题1: 鲁迅先生在日本学习医学的老师是谁？
回答1: 鲁迅先生在日本学习医学的老师是藤野先生。

问题2: 作者在离开仙台时，对藤野先生说了什么谎话？他为什么要说这个谎话？
回答2: 作者对藤野先生谎称他想学习生物学，因为他看到藤野先生有些悲哀，所以撒了个谎来安慰他。

问题3: 文章中提到了哪两处鲁迅在仙台期间“受到优待”或“被特殊照顾”的例子？
回答3: 鲁迅在仙台期间，学校不收他的学费，并且几个职员还为他食宿操心，这是两处受到优待的例子。

问题4: 日本爱国青年学生为什么要给作者写匿名信？匿名信的开头引用了谁的什么句子？
回答4: 日本爱国青年学生给作者写匿名信是因为他们怀疑作者在解剖学考试中作弊，信的开头引用了《新约》中的句子“你改悔罢！”。

问题5: 是什么具体的事件促使作者做出了“不学医学”、离开仙台的决定？这一事件对他的思想产生了怎样的冲击？
回答5: 作者因受到不实指控和歧视，以及观看日本战胜俄国的影片中中国人被枪毙的场景，感到愤慨和无力，这些事件促使他放弃学医并离开仙台，对他的思想产生了深刻的冲击，使他更加坚定地反对不公和追求正义。

问题6: 作者保存的藤野先生修改过的讲义最终去向如何？现在作者手头还留有什么关于藤野先生的纪念物？
回答6: 作者将藤野先生修改过的讲义退还给了学生会干事，现在手头留有藤野先生赠予的一张写有“惜别”的照相作为纪念物。
"""
