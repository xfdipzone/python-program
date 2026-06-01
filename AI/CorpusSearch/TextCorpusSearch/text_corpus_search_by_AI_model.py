# coding=utf-8
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from google.colab import userdata
from huggingface_hub import login
import os

"""
根据提供的语料库，回答用户提出的问题

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

# 执行查询
question = "鲁迅先生在日本学习医学的老师是谁？"
query_engine = index.as_query_engine()
response = query_engine.query(question)

print(f"问题：{question}")
print(f"回答：{response}")
