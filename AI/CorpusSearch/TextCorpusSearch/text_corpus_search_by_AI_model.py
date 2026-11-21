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
    model=userdata.get("AI_DEFAULT_MODEL"),
    api_key=userdata.get("API_KEY"),
    api_base=userdata.get("AI_BASE_URL"),
    is_chat_model=True,
    context_window=8192,  # 显式声明上下文窗口，省去手动覆盖属性的麻烦
    is_function_calling_model=False,

    # 关闭模型思考模式
    additional_kwargs={
        "extra_body": {
            "thinking": {
                "type": "disabled"
            }
        }
    }
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
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact"
)

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
    # 优化问题，增加 prompt 说明
    prompted_question = f"{question} （请用一段话简短回答，不超过 80 字，如果内容中没有与问题相关的内容，请回答 '不知道'）"

    response = query_engine.query(prompted_question)
    print(f"问题{index + 1}: {question}")
    print(f"回答{index + 1}: {response}\n")

"""
问题1: 鲁迅先生在日本学习医学的老师是谁？
回答1: 鲁迅在仙台医学专门学校学习时的老师是藤野先生（藤野严九郎），他教解剖学等课程，曾用红笔为鲁迅添改讲义，并赠送写有“惜别”的照片。

问题2: 作者在离开仙台时，对藤野先生说了什么谎话？他为什么要说这个谎话？
回答2: 作者说要去学生物学，其实并未决意，因见藤野先生凄然，便说这谎话安慰他。

问题3: 文章中提到了哪两处鲁迅在仙台期间“受到优待”或“被特殊照顾”的例子？
回答3: 学校不收学费，几个职员为他的食宿操心。

问题4: 日本爱国青年学生为什么要给作者写匿名信？匿名信的开头引用了谁的什么句子？
回答4: 日本爱国青年学生因怀疑作者解剖学成绩是藤野先生泄露题目所得，便写匿名信讥刺他。信的开头引用了托尔斯泰给俄国和日本皇帝信中的句子：“你改悔罢！”

问题5: 是什么具体的事件促使作者做出了“不学医学”、离开仙台的决定？这一事件对他的思想产生了怎样的冲击？
回答5: 看电影时，中国人围观同胞被枪毙却拍掌欢呼，这刺耳的一幕使作者决定不学医学、离开仙台；它让他深感弱国子民的麻木，思想受到强烈冲击。

问题6: 作者保存的藤野先生修改过的讲义最终去向如何？现在作者手头还留有什么关于藤野先生的纪念物？
回答6: 讲义在七年前迁居时因书箱毁坏而遗失，找运送局寻也无回信；现在手头只留有藤野先生的照相，挂在寓居东墙上。

问题7: 西游记中有什么主要的人物？
回答7: 不知道
"""
