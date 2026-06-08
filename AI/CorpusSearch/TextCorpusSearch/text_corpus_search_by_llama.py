# coding=utf-8
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig
from google.colab import userdata
from huggingface_hub import login
import torch
import os
import warnings
import logging

"""
根据提供的语料库，使用本地大语言模型（unsloth/Llama-3.2-3B-Instruct）分析，回答用户提出的问题

dependency packages
pip install llama-index-core
pip install llama-index-embeddings-huggingface
pip install llama-index-llms-huggingface
pip install llama-index-readers-file
pip install bitsandbytes
pip install accelerate

优化安装（分三个单元格运行，安装完之后重启一次会话）
!pip install transformers==4.48.3 huggingface_hub==0.28.1 accelerate==1.3.0 bitsandbytes==0.45.0 llama-index-core llama-index-embeddings-huggingface llama-index-llms-huggingface llama-index-readers-file --quiet 2>/dev/null
!pip install git+https://github.com/huggingface/transformers.git huggingface_hub>=1.0 --upgrade --quiet 2>/dev/null
!pip install -U bitsandbytes --quiet
"""
# Login HuggingFace Hub
login(token=userdata.get("HF_TOKEN"))

# 加载 Hugging Face 的嵌入模型
embed_model = HuggingFaceEmbedding(
    model_name="google/embeddinggemma-300M",
    embed_batch_size=32
)

# 屏蔽系统警告
warnings.filterwarnings("ignore")

# 设置只打印程序崩溃的日志
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("llama_index").setLevel(logging.CRITICAL)

# 让 transformers 库只打印 ERROR，屏蔽 INFO 与 WARNING
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# 配置 4-bit 量化，让模型在 T4 GPU 运行流畅
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 定义使用的模型
model_id = "unsloth/Llama-3.2-3B-Instruct"

# 初始化大语言模型
llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={
        "temperature": 0.3,
        "do_sample": True,
        "repetition_penalty": 1.2
    },
    model_kwargs={"quantization_config": quantization_config},  # 注入量化配置
    # 屏蔽 clean_up_tokenization_spaces 提示
    tokenizer_kwargs={"clean_up_tokenization_spaces": False},
    device_map="auto",
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
    index = load_index_from_storage(storage_context, embed_model=embed_model)

# 查询引擎
query_engine = index.as_query_engine()

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
    prompted_question = f"{question} （请用一段中文简短回答，不超过 80 字，如果内容中没有与问题相关的内容，请回答 '不知道'）"

    response = query_engine.query(prompted_question)
    print(f"问题{index + 1}: {question}")
    print(f"回答{index + 1}: {response}\n")
