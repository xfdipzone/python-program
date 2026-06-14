# coding=utf-8
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, Settings, PromptTemplate
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
        "temperature": 0.1,
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

# 定义 RAG 提示词模版 (Llama-3 Template)
text_qa_template = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "你是一个严谨的回答专家。请根据提供的上下文回答问题。不超过 80 字。若无相关内容请说'不知道'。<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "【参考上下文】\n{context_str}\n\n"
    "【用户问题】\n{query_str}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

text_qa_template = PromptTemplate(text_qa_template)

# 查询引擎
query_engine = index.as_query_engine(
    text_qa_template=text_qa_template
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
    response = query_engine.query(question)
    print(f"问题{index + 1}: {question}")
    print(f"回答{index + 1}: {response}\n")

"""
问题1: 鲁迅先生在日本学习医学的老师是谁？
回答1: 藤野先生

问题2: 作者在离开仙台时，对藤野先生说了什么谎话？他为什么要说这个谎话？
回答2: 答案：作者在离开仙台时，对藤野先生说的是：“我想去学生物学，先生教给我的学问，也还有用的。”

问题3: 文章中提到了哪两处鲁迅在仙台期间“受到优待”或“被特殊照顾”的例子？
回答3: 据《mr_fujino》中的描述，鲁迅在仙台期间接受了以下两处特殊照顾：

1. 学校不收取学费
2. 一个客店（可能是监狱附近的）同时也是囚犯们的饭食供应者，尽管如此，鲁迅仍能获得舒适的生活条件和良好的饮食。

问题4: 日本爱国青年学生为什么要给作者写匿名信？匿名信的开头引用了谁的什么句子？
回答4: 答案：日本爱国青年学生给作者写匿名信是为了表示反感谢，而匿名信的开头引用的是《新约》的句子"你改悔罢!"

问题5: 是什么具体的事件促使作者做出了“不学医学”、离开仙台的决定？这一事件对他的思想产生了怎样的冲击？
回答5: 这是一个关于作者与老师 mr. Fujino 的故事。这个事件发生的是，当作者看到他被称赞的文章和其他人在看枪毙中国人时，感到非常反感。这让他意识到自己一直以来对日本的态度可能是不正确的。他开始质疑自己是否只是因为自满而接受了mr.Fujino的教育，而不是真正理解其中的价值。

问题6: 作者保存的藤野先生修改过的讲义最终去向如何？现在作者手头还留有什么关于藤野先生的纪念物？
回答6: 藤野先生修改过的讲义最终被送到了某个地方，但是具体地点未知。作者手头还剩下一些关于藤野先生的纪念物，如他的照相和一些讲义。

问题7: 西游记中有什么主要的人物？
回答7: 这个问题与原文无关。
"""
