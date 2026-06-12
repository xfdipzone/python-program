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
根据提供的语料库，使用本地大语言模型（Qwen/Qwen2.5-3B-Instruct）分析，回答用户提出的问题

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
model_id = "Qwen/Qwen2.5-3B-Instruct"

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
    model_kwargs={
        "quantization_config": quantization_config,  # 注入量化配置
        "trust_remote_code": True  # 允许加载 Qwen 的远程代码
    },
    tokenizer_kwargs={"trust_remote_code": True},
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

# 定义 RAG 提示词模版 (ChatML)
text_qa_template = (
    "<|im_start|>system\n"
    "你是一个严谨的问答助手。请仅根据下文内容，用一句话简短回答问题（不超过80字）。如果文中没提到，请直接回答“不知道”。<|im_end|>\n"
    "<|im_start|>user\n"
    "相关文章片段：\n{context_str}\n\n"
    "请回答问题：{query_str}<|im_end|>\n"
    "<|im_start|>assistant\n"
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
回答2: 他说的是“我想去学生物学”，因为他觉得藤野先生可能会难过，所以他撒了个谎说是自己真的想去学习生物。
原因是因为藤野先生可能感到伤心或沮丧，为了安慰他，他就编造了一条理由说自己确实想要转学到研究生物科学方面。这句话体现了他对老师的尊重以及关心老师的心情。

问题3: 文章中提到了哪两处鲁迅在仙台期间“受到优待”或“被特殊照顾”的例子？
回答3: 在日本留学时期得到仙台日本医专同学会的帮助租了一间宿舍。在北京景山公园附近的一家客店里住宿得到了管理者的关照。

问题4: 日本爱国青年学生为什么要给作者写匿名信？匿名信的开头引用了谁的什么句子？
回答4: 匿名信的原因是因为对鲁迅先生的文章中提及日本人解剖实验的内容感到不满；
开头引用的是《新约》中的“你改悔罢”的句子；来自托尔斯泰。

问题5: 是什么具体的事件促使作者做出了“不学医学”、离开仙台的决定？这一事件对他的思想产生了怎样的冲击？
回答5: 这件事让作者认识到中国的弱国身份以及中国人民的愚昧无知，从而改变了他对祖国未来的看法。知道了自己原来的成绩并不是靠自己的努力取得后，再加上看电影上的日俄战争影片，看到竟然也有中国人围观并为之鼓掌的画面，这让作者感到非常震惊与痛苦，因此萌生了弃医从文的想法。这一事件对他思想产生的巨大冲击让他意识到必须改变现状才能拯救国家，所以他选择放弃医学专业转学到东京帝国大学攻读文学。

问题6: 作者保存的藤野先生修改过的讲义最终去向如何？现在作者手头还留有什么关于藤野先生的纪念物？
回答6: 现在的样子还在悬挂在北京寓处的东墙上。剩下的东西是我收藏着的三本订正的笔记。

问题7: 西游记中有什么主要的人物？
回答7: 不知道
"""
