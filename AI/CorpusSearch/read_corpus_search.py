# coding=utf-8
"""
根据提供的语料库，回答用户提出的问题

dependency packages
pip install llama-index
pip install langchain
pip install transformers
pip install sentence-transformers
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-huggingface
pip install icetk
pip install cpm_kernels
pip install torch>=2.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

此方法需要 OpenAI Quota
"""
import openai
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

# 设置 OpenAI API 密钥
openai.api_key = os.environ.get("OPENAI_API_KEY")

# 定义 embedding 模型
embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# 加载数据并构建索引
documents = SimpleDirectoryReader('./data/mr_fujino').load_data()
index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# 持久化索引到磁盘
index.storage_context.persist(persist_dir='index_mr_fujino')

# 从磁盘中读取索引
storage_context = StorageContext.from_defaults(persist_dir="./index_mr_fujino")
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()

# 执行查询
response = query_engine.query("鲁迅先生在日本学习医学的老师是谁？")
print(response)
